# Troubleshooting

## `Config must provide rubrics, rubrics_file, or seed_dataset`

`Judger.initialize()` needs a rubric source. Provide one of:

```yaml
rubrics_file: "rubric_pipeline/rubrics/flux_t2i_general_v1.txt"
```

or:

```yaml
rubrics: |
  Rubric 1:
  Theme: Prompt alignment
  - Tip: Prefer outputs that satisfy all required objects and relations.
```

or pass a seed dataset when constructing the judge:

```python
arr_judge = await Judger(
    config_path="rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml",
    seed_dataset="examples/seed_t2i_pairwise.json",
).initialize()
```

## The VLM Returns Invalid JSON

Try these fixes:

- lower concurrency and retry;
- use a stronger JSON-following model;
- shorten `task_description`;
- reduce `query_specific_generate_number`;
- reduce `categories_number`;
- inspect the raw response from the VLM server logs.

## Rank Length Is Wrong

The rank array must contain one entry per candidate output, excluding the source image for image editing.

For T2I pairwise:

```json
{
  "response": ["a.png", "b.png"],
  "label_rank": [1, 2]
}
```

For image editing:

```json
{
  "source_image": "source.png",
  "edited_images": ["edit_a.png", "edit_b.png"],
  "label_rank": [2, 1]
}
```

Do not include `Image BASE` in `label_rank`.

## Image Order Seems Wrong

The code sends images in this order:

- T2I: `Image 1`, `Image 2`, ...
- Image editing: `Image BASE`, then `Edited Image 1`, `Edited Image 2`, ...

Accepted image keys are normalized in `rubric_pipeline/utils/vision.py`. Prefer explicit keys such as `source_image` and `edited_images` for image editing to avoid ambiguity.

## Rubrics Are Too Generic

Improve seed data first. Add examples where generic aesthetics would choose the wrong image. Then update `task_description` to name the domain-specific dimensions that matter.

Useful sentence:

```yaml
task_description: |
  Do not reward generic visual attractiveness when required content,
  source preservation, or edit instruction fulfillment is missing.
```

## Rubrics Overfit One Domain

Use a broader seed set or generate separate rubric files by domain:

- `flux_t2i_general_v1.txt`
- `flux_t2i_portrait_v1.txt`
- `flux_t2i_product_v1.txt`
- `qwen_edit_identity_v1.txt`

Choose the rubric file that matches the training or evaluation distribution.

## RPO Is Slow

Common bottlenecks:

- VLM judge latency during pairwise reward calls.
- Local VLM serving throughput.
- Large image decoding and saving in rollout.
- Too many sampling steps.

Practical mitigations:

- reuse a saved `rubrics_file`;
- lower `--concurrency_limit` for offline evaluation to avoid endpoint failures;
- keep pairwise `--num_generations 2`;
- run smoke tests with fewer `max_train_steps`;
- use the paper-aligned sampling defaults before increasing steps.

