# Data Formats

This page describes the JSON shapes accepted by `rubric_pipeline/utils/vision.py`.
For rubric generation strategy, VLM choice, and reuse, start from the
[Auto-Rubric Guide](auto_rubric/README.md).

## Pairwise Text-To-Image

Use this format for FLUX rubric generation and evaluation:

```json
{
  "query": "A cinematic portrait of a ceramic robot chef.",
  "response": ["path/to/image_a.png", "path/to/image_b.png"],
  "label_rank": [1, 2]
}
```

`label_rank` uses smaller-is-better rank numbers. `[1, 2]` means the first image is preferred.

Accepted image keys for text-to-image include:

- `response`
- `responses`
- `images`
- `image_paths`
- `generated_images`

## Pairwise Image Editing

Use this format for Qwen-Image-Edit:

```json
{
  "query": "Replace the sky with a sunset while preserving the building.",
  "source_image": "path/to/source.png",
  "edited_images": ["path/to/edit_a.png", "path/to/edit_b.png"],
  "label_rank": [2, 1]
}
```

Accepted source image keys include:

- `source_image`
- `base_image`
- `original_image`
- `input_image`
- `reference_image`

Accepted edited-output keys include:

- `edited_images`
- `target_images`
- `output_images`
- `result_images`
- `response`

## Pointwise Mode

Pointwise grading is available, although the included training launchers use pairwise RPO.

```json
{
  "query": "A realistic photo of a glass teapot.",
  "response": "path/to/image.png",
  "label_score": 4
}
```

Set `eval_mode: "pointwise"` and configure `min_score` / `max_score` in the YAML file.

## Path Rules

Paths may be absolute, relative to the current working directory, or URLs/data URLs accepted by the OpenAI-compatible vision endpoint. Local paths are encoded as base64 image URLs before being sent to the VLM.

## Related Guides

- [Rubric Design](auto_rubric/rubric_design.md)
- [Rubric Reuse](auto_rubric/rubric_reuse.md)
- [Auto-Rubric Workflows](auto_rubric/workflows.md)
