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
import torch
from accelerate.logging import get_logger
import json
import os
import torch.distributed as dist

logger = get_logger(__name__)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPipeline
import math

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

def contains_chinese(text):
    """检查字符串是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def calculate_embeds(pipe, height, width, data, device):
    image_path = data['source_image']
    image = Image.open(f"{image_path[0]}").convert("RGB")
    image_size = image[0].size if isinstance(image, list) else image.size
    calculated_width, calculated_height, _ = calculate_dimensions(512 * 512, image_size[0] / image_size[1])
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = 8 * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    calculated_width = width
    calculated_height = height

    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == pipe.latent_channels):
        image = pipe.image_processor.resize(image, height, width)
        prompt_image = image
        image = pipe.image_processor.preprocess(image, height, width)
        image = image.unsqueeze(2)

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            image=prompt_image,
            prompt=data['instruction'][0],
            device=device
        )
    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            image_latents = pipe._encode_vae_image(image = image.to(torch.bfloat16).to(device), generator = None)

    return prompt_embeds, prompt_embeds_mask, calculated_width, calculated_height, image_latents

    



class T5dataset(Dataset):
    def __init__(self, jsonl_path, base_image_path):
        self.jsonl_path = jsonl_path
        self.base_image_path = base_image_path
        
        # 读取JSONL文件并解析每一行
        self.data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取source_image路径并添加前缀
        source_image = item.get("source_image", "")
        source_image_path = os.path.join(self.base_image_path, source_image) if source_image else ""
        
        # 获取instruction
        instruction = item.get("instruction", "")
        
        return {
            "source_image": source_image_path,
            "instruction": instruction,
            "filename": str(idx)
        }

    def __len__(self):
        return len(self.data)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "image_latents"), exist_ok=True)

    # 创建数据集实例
    train_dataset = T5dataset(
        jsonl_path=args.prompt_dir,
        base_image_path=args.base_image_dir,
    )
    
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    pipe = QwenImageEditPipeline.from_pretrained(args.model_path).to(torch.bfloat16).to(device)
    print("pipeline loaded")
    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            with torch.autocast("cuda"):
                prompt_embeds, prompt_attention_mask, calculated_width, calculated_height, image_latents = calculate_embeds(pipe, args.height, args.width, data, device)
                # ==================== 代码修改开始 ====================
                # 1. 记录原始的序列长度 (第二个维度的大小)
                original_length = prompt_embeds.shape[1]
                target_length = 5000
                
                # 2. 计算需要填充的长度
                # 假设 original_length 不会超过 target_length
                pad_len = target_length - original_length

                # 3. 填充 prompt_embeds
                # prompt_embeds 是一个3D张量 (B, L, D)，我们需要填充第二个维度 L
                # F.pad 的填充参数顺序是从最后一个维度开始的 (pad_dim_D_left, pad_dim_D_right, pad_dim_L_left, pad_dim_L_right, ...)
                # 我们在维度1（序列长度L）的右侧进行填充
                prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad_len), "constant", 0)

                # 4. 填充 prompt_attention_mask
                # prompt_attention_mask 是一个2D张量 (B, L)，我们同样填充第二个维度 L
                # 我们在维度1（序列长度L）的右侧进行填充
                prompt_attention_mask = F.pad(prompt_attention_mask, (0, pad_len), "constant", 0)

                # ==================== 代码修改结束 ====================

                if args.vae_debug:
                    latents = data["latents"]
                for idx, video_name in enumerate(data["filename"]):
                    prompt_embed_path = os.path.join(
                        args.output_dir, "prompt_embed", video_name + ".pt"
                    )
                    prompt_attention_mask_path = os.path.join(
                        args.output_dir, "prompt_attention_mask", video_name + ".pt"
                    )
                    image_latents_path = os.path.join(
                        args.output_dir, "image_latents", video_name + ".pt"
                    )
                    # 保存 latent (注意这里保存的是填充后的张量)
                    torch.save(prompt_embeds[idx], prompt_embed_path)
                    torch.save(prompt_attention_mask[idx], prompt_attention_mask_path)
                    torch.save(image_latents[idx], image_latents_path)
                    item = {}
                    item["prompt_embed_path"] = video_name + ".pt"
                    item["prompt_attention_mask"] = video_name + ".pt"
                    item["image_latents"] = video_name + ".pt"
                    item["caption"] = data["instruction"][idx]
                    item["source_image"] = data["source_image"][idx]
                    
                    # [新增] 将原始长度记录到 item 字典中
                    item["original_length"] = original_length
                    item["calculated_height"] = calculated_height
                    item["calculated_width"] = calculated_width
                    
                    json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/qwenimage_edit")
    parser.add_argument("--model_type", type=str, default="qwen-image-edit")
    # text encoder & vae & diffusion model
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    parser.add_argument("--prompt_dir", type=str, default="./empty.txt")
    parser.add_argument(
        "--base_image_dir",
        type=str,
        default="./data/edit_images",
        help="Directory containing source images referenced by the edit jsonl.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    main(args)
