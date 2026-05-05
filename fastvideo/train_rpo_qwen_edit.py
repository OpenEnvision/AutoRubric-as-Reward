# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import asyncio
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_qwenimage_edit_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_hf_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from fastvideo.utils.fsdp_util_qwenimage import fsdp_wrapper, FSDPConfig
from contextlib import contextmanager
from safetensors.torch import save_file
import copy

from rubric_pipeline import Judger

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

class FSDP_EMA:
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0, 
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0, 
                options=restore_options
            )

def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    if rank == 0 and ema_handler is not None:
        ema_checkpoint_path = os.path.join(output_dir, f"checkpoint-ema-{step}-{epoch}")
        os.makedirs(ema_checkpoint_path, exist_ok=True)
        weight_path = os.path.join(ema_checkpoint_path ,
                                   "diffusion_pytorch_model.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, weight_path)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(ema_checkpoint_path, "config.json")
        # save dict as json
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        #torch.save(ema_handler.ema_state_dict_rank0, os.path.join(ema_checkpoint_path, "ema_model.pt"))
        main_print(f"--> EMA checkpoint saved at {ema_checkpoint_path}")


def validate_rpo_args(args):
    if not args.use_arr:
        raise ValueError("qwen-edit RPO only supports the ARR VLM judge. Please pass --use_arr.")
    if not args.arr_config_path:
        raise ValueError("--arr_config_path is required for ARR reward.")
    if args.num_generations != 2:
        raise ValueError("RPO is pairwise and requires --num_generations 2.")
    if not args.use_group:
        raise ValueError("RPO expects --use_group so each prompt forms one pair.")


@contextmanager
def disable_lora(transformer):
    module = transformer.module if isinstance(transformer, FSDP) else transformer
    if hasattr(module, "disable_adapter"):
        with module.disable_adapter():
            yield
    else:
        yield


def apply_lora(transformer, args):
    if get_peft_model is None or LoraConfig is None:
        raise ImportError(
            "LoRA training requires `peft`. Please install peft before running this script."
        )

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must not be empty for LoRA training")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    return transformer


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        prompt_attention_mask,
        img_shapes,
        txt_seq_lens,
        image_latents,
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            latent_model_input = torch.cat([z, image_latents], dim=1)
            with torch.autocast("cuda", torch.bfloat16):
                pred= transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            pred = pred[:, :z.shape[1]]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
        args,
        latents,
        pre_latents,
        encoder_hidden_states, 
        prompt_attention_masks, 
        txt_seq_lens,
        img_shapes,
        image_latents,
        transformer,
        timesteps,
        i,
        sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    if isinstance(txt_seq_lens, torch.Tensor):
        txt_seq_lens = [int(value) for value in txt_seq_lens.flatten().tolist()]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        pred= transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_attention_masks,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        pred = pred[:, :latents.shape[1]]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob



def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    prompt_attention_masks, 
    original_length,
    calculated_height, 
    calculated_width,
    image_latents,
    arr_judge,
    caption,
    source_image,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []  
    all_txt_seq_lens = []
    all_image_latents = []
    pair_item = None
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        sample_idx = int(batch_idx[0].item())
        sample_length = int(original_length[sample_idx].item())
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx][:, :sample_length]
        batch_prompt_attention_mask = prompt_attention_masks[batch_idx][:, :sample_length]
        batch_image_latents = image_latents[batch_idx]
        batch_calculated_height = calculated_height[batch_idx]
        batch_calculated_width = calculated_width[batch_idx]

        batch_caption = [caption[int(i.item())] for i in batch_idx]
        batch_source_image = [source_image[int(i.item())] for i in batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        packed_height = 2 * (int(h) // (8 * 2))
        packed_width = 2 * (int(w) // (8 * 2))
        input_latents_new = pack_latents(input_latents, len(batch_idx), 16, packed_height, packed_width)
        txt_seq_lens =  batch_prompt_attention_mask.sum(dim=1).tolist() 
        img_shapes = [[
            (1, h // 8 // 2, w // 8// 2),
            (1, batch_calculated_height.item() // 8 // 2, batch_calculated_width.item() // 8 // 2),
            ]] 
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        image_latent_height, image_latent_width = batch_image_latents.shape[3:]
        image_latents_new = pack_latents(
                batch_image_latents, len(batch_idx), 16, image_latent_height, image_latent_width
            )

        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents_new.clone(),
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_prompt_attention_mask,
                img_shapes,
                txt_seq_lens,
                image_latents_new,
                grpo_sample,
            )
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_txt_seq_lens.append(torch.tensor(txt_seq_lens))
        all_image_latents.append(image_latents_new)
        vae.enable_tiling()
        
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = latents.to(vae.dtype)
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                image = vae.decode(latents, return_dict=False)[0][:, :, 0]
                decoded_image = image_processor.postprocess(image)
        decoded_image[0].save(f"./images/qwen_image_edit_{rank}_{index}.png")
        image_path = f"./images/qwen_image_edit_{rank}_{index}.png"
        if index % 2 == 0:
            pair_item = {
                "query": batch_caption[0],
                "source_image": batch_source_image[0],
                "response": [image_path],
            }
        else:
            if pair_item is None:
                raise RuntimeError("Internal RPO pairing error: missing first sample.")
            pair_item["response"].append(image_path)
            rewards = get_arr_reward_sync(arr_judge, pair_item, device)
            if isinstance(rewards, torch.Tensor) and rewards.ndim == 2:
                rewards = rewards[:, 0]
            if rewards.numel() != 2:
                raise ValueError(f"ARR pairwise reward must return 2 advantages, got {rewards}.")
            all_rewards.append(rewards)
            pair_item = None

    if pair_item is not None:
        raise ValueError("RPO requires an even number of samples per prompt pair.")

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    all_txt_seq_lens = torch.cat(all_txt_seq_lens, dim=0)
    all_image_latents = torch.cat(all_image_latents, dim=0)
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, all_image_latents


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def compute_rpo_advantages(args, rewards, gathered_rewards):
    rewards = rewards.to(torch.float32)
    if args.use_group:
        return rewards
    return (rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-8)


def get_arr_reward_sync(arr_judge, qa_data, device):
    return asyncio.run(arr_judge.get_reward_single(qa_data, device))

def train_one_step(
    args,
    device,
    transformer,
    vae,
    arr_judge,
    optimizer,
    lr_scheduler,
    prompt_embeds, 
    prompt_attention_masks, 
    caption, 
    original_length,
    calculated_height, 
    calculated_width,
    source_image,
    image_latents,
    noise_scheduler,
    max_grad_norm,
    ema_handler
):
    total_loss = 0.0

    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(prompt_embeds)
        prompt_attention_masks = repeat_tensor(prompt_attention_masks)
        image_latents = repeat_tensor(image_latents)
        original_length = repeat_tensor(original_length)
        calculated_height = repeat_tensor(calculated_height)
        calculated_width = repeat_tensor(calculated_width)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
            source_image = [source_image] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
            source_image = [item for item in source_image for _ in range(args.num_generations)]
        elif isinstance(caption, tuple):
            caption = [item for item in caption for _ in range(args.num_generations)]
            source_image = [item for item in source_image for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    reward, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, all_image_latents = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            encoder_hidden_states, 
            prompt_attention_masks, 
            original_length,
            calculated_height, 
            calculated_width,
            image_latents,
            arr_judge,
            caption,
            source_image,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "txt_seq_lens": all_txt_seq_lens,
        "encoder_hidden_states": encoder_hidden_states,
        "prompt_attention_masks": prompt_attention_masks,
        "image_latents": all_image_latents,
        "original_length": original_length,
        "calculated_height": calculated_height,
        "calculated_width": calculated_width,
    }
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        with open('./reward.txt', 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    samples["advantages"] = compute_rpo_advantages(args, samples["rewards"], gathered_reward)

    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            sample_length = int(sample["original_length"].flatten()[0].item())
            sample_height = int(sample["calculated_height"].flatten()[0].item())
            sample_width = int(sample["calculated_width"].flatten()[0].item())
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"][:, :sample_length],
                sample["prompt_attention_masks"][:, :sample_length],
                sample["txt_seq_lens"],
                [[
                    (1, args.h // 8 // 2, args.w // 8// 2),
                    (1, sample_height // 8 // 2, sample_width // 8 // 2),
                ]],
                sample["image_latents"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
            )
            if args.kl_beta > 0:
                with torch.no_grad(), disable_lora(transformer):
                    ref_log_probs = grpo_one_step(
                        args,
                        sample["latents"][:,_],
                        sample["next_latents"][:,_],
                        sample["encoder_hidden_states"][:, :sample_length],
                        sample["prompt_attention_masks"][:, :sample_length],
                        sample["txt_seq_lens"],
                        [[
                            (1, args.h // 8 // 2, args.w // 8 // 2),
                            (1, sample_height // 8 // 2, sample_width // 8 // 2),
                        ]],
                        sample["image_latents"],
                        transformer,
                        sample["timesteps"][:,_],
                        perms[i][_],
                        sigma_schedule,
                    )
                kl_term = new_log_probs - ref_log_probs
            else:
                kl_term = torch.zeros_like(new_log_probs)

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss) + args.kl_beta * kl_term) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("reward", sample["rewards"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            print("kl", kl_term.mean().item())
            print("final loss", loss.item())
        dist.barrier()
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    validate_rpo_args(args)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    arr_judge = asyncio.run(
        Judger(config_path=args.arr_config_path).initialize()
    )

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
    transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )

    transformer = apply_lora(transformer, args)

    # Setup FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False, 
    )
    transformer = fsdp_wrapper(transformer, fsdp_config,)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    apply_fsdp_checkpointing(
            transformer, (QwenImageTransformerBlock), args.selective_checkpointing
        )

    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    #vae.enable_tiling()

    if rank <= 0:
        project = "qwenimage_edit"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running RPO training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )



    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(3):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        for step, (prompt_embeds, prompt_attention_masks, image_latents, caption, original_length, calculated_height, calculated_width, source_image) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_masks = prompt_attention_masks.to(device)
            start_time = time.time()
            image_latents = image_latents.to(device)
            original_length = original_length.to(device)
            calculated_height = calculated_height.to(device)
            calculated_width = calculated_width.to(device)
            if (step-1) % args.checkpointing_steps == 0 and step!=1:
                save_hf_lora_checkpoint(transformer, optimizer, rank, args.output_dir, step, epoch)
                if args.use_ema:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))


                dist.barrier()
            if step > (args.max_train_steps+1):
                break
            loss, grad_norm = train_one_step(
                args,
                device, 
                transformer,
                vae,
                arr_judge,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                prompt_attention_masks, 
                caption, 
                original_length,
                calculated_height, 
                calculated_width,
                source_image,
                image_latents,
                noise_scheduler,
                args.max_grad_norm,
                ema_handler,
            )

            if args.use_ema and ema_handler:
                ema_handler.update(transformer)
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0",
        help="Comma-separated module names to inject LoRA into.",
    )

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #RPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="image height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="image width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="latent temporal length; keep 1 for image-edit training",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=10,
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_arr",
        action="store_true",
        default=False,
        help="whether to use the ARR VLM judge for pairwise rewards",
    )
    parser.add_argument(
        "--arr_config_path",
        type=str,
        default=None,
        help="config path for the ARR VLM judge",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=0.2,
        help="clip range for RPO",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.02,
        help="KL penalty coefficient against the frozen base LoRA-disabled policy.",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
    )
    




    args = parser.parse_args()
    main(args)
