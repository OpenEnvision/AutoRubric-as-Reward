# FastVideo Components Kept In This Release

This directory contains the minimal FastVideo-derived pieces needed by Vision-Auto-Rubric:

- `train_rpo_flux.py` for FLUX LoRA RPO.
- `train_rpo_qwen_edit.py` for Qwen-Image-Edit LoRA RPO.
- FLUX and Qwen-Image-Edit embedding datasets.
- FSDP, checkpoint, sequence-parallel, optimizer, and communication utilities.

Non-target model training code has been removed from the open-source release.

For Auto-Rubric usage and launch commands, see:

- `../docs/auto_rubric.md`
- `../docs/training.md`
- `../docs/data_preprocess.md`
