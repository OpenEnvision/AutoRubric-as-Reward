# Rubric Reuse

Rubric reuse is the recommended path for long RPO runs. Generate and validate rubrics once, save the final text, then load the same file during training. This makes runs more deterministic and avoids spending VLM calls every time training starts.

## Why Reuse Rubrics

| Benefit | Explanation |
| --- | --- |
| Determinism | The same rubric text is used across restarts and ablations. |
| Cost control | Rubric generation, verification, and categorization happen once. |
| Easier debugging | Reward changes come from model outputs, not changing rubric text. |
| Better review | Humans can inspect and edit the criteria before training. |
| Versioning | Rubric files can be named and tracked by domain or experiment. |

## Recommended Folder Layout

```text
rubric_pipeline/rubrics/
  flux_t2i_general_v1.txt
  flux_t2i_product_v1.txt
  qwen_edit_general_v1.txt
  qwen_edit_identity_preservation_v1.txt
```

Use names that encode:

- task: `flux_t2i`, `qwen_edit`
- domain: `general`, `product`, `portrait`, `architecture`
- version: `v1`, `v2`, `ablation_no_aesthetic`

## Save A Rubric Set

Run generation and evaluation:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --seed_dataset examples/seed_t2i_pairwise.json \
  --test_dataset examples/test_t2i_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 4
```

Copy the printed `Rubrics generated:` block into:

```text
rubric_pipeline/rubrics/flux_t2i_general_v1.txt
```

Then load it from YAML:

```yaml
rubrics_file: "rubric_pipeline/rubrics/flux_t2i_general_v1.txt"
```

Inline rubrics also work:

```yaml
rubrics: |
  Rubric 1:
  Theme: Prompt and instruction alignment
  - Tip: Prefer images that include all requested objects, attributes, and relations.
  - Tip: Penalize images that are visually attractive but miss required content.
```

## Validate Before Training

Always test a saved rubric on held-out pairs:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --rubrics_file rubric_pipeline/rubrics/flux_t2i_general_v1.txt \
  --test_dataset examples/test_t2i_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 4
```

Look for:

- rank arrays with the correct length;
- accuracy above the direct VLM baseline on your validation set;
- balanced performance when candidate image order is swapped;
- reasons that cite the rubric rather than generic preference language.

## When To Regenerate

Regenerate or revise rubrics when:

- the image domain changes substantially;
- the base generator changes failure modes;
- you switch from T2I to image editing;
- you switch judge model families and observe degraded accuracy;
- validation examples reveal systematic blind spots;
- the current rubric over-rewards an unwanted style.

## When To Edit Manually

Manual edits are useful after generation. Keep edits conservative:

- remove duplicated tips;
- shorten very long tips;
- add source-preservation language for editing;
- add "do not reward generic beauty over prompt adherence";
- remove criteria that depend on hidden metadata or subjective taste.

After manual edits, rerun validation. Treat the edited rubric as a new version.

