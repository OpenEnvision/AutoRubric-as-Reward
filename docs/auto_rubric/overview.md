# Auto-Rubric Overview

Auto-Rubric turns preference examples into an explicit VLM judging standard. Instead of asking a VLM to "pick the better image" with only its implicit preferences, the pipeline first asks the model to write concrete criteria, verifies those criteria on labeled examples, and then uses the verified rubric text as the conditioning context for future grading.

## Why It Helps

Plain pairwise VLM judging is often unstable because the model must infer the standard from scratch every time. It may over-weight image order, style, color saturation, or generic aesthetics. Auto-Rubric makes the standard visible:

1. A seed pair says which output is preferred.
2. The VLM proposes criteria that explain the preference.
3. The same criteria are used to judge the seed pair.
4. If the judgment does not match the label, the criteria are revised.
5. Only verified criteria are retained.
6. Verified criteria are grouped into a reusable rubric set.

## Paper Concept To Code

| Paper concept | Code location | Notes |
| --- | --- | --- |
| `M_gen` rubric generation | `QuerySpecificRubricGenerator.generate()` | Produces candidate rubrics from a labeled visual item. |
| `M_verify` verification | `QuerySpecificRubricGenerator.aevaluate()` and `validate()` | Checks whether the generated criteria recover `label_score` or `label_rank`. |
| `M_refine` refinement | `QuerySpecificRubricGenerator.revise()` | Uses expected vs. actual feedback to rewrite criteria. |
| Verified set `D_R` | `IterativeRubricsGenerator._generate_query_rubrics()` | Keeps rubrics only when validation succeeds. |
| Structured rubric `R_structured` | `LLMRubricCategorizer` | Groups verified rubrics into Theme-Tips style categories. |
| Reward conversion | `Judger._rank_to_rewards()` | Pairwise rank 1 receives `1.0`; rank 2 receives `-0.1`. |

## Supported Tasks

| Task | `task_type` | Input images | Main rubric dimensions |
| --- | --- | --- | --- |
| Text-to-image | `t2i` | Generated candidates | Prompt adherence, object attributes, relations, composition, artifacts, aesthetics. |
| Image editing | `image_edit` | Optional source image plus edited candidates | Instruction fulfillment, source preservation, local edit quality, blending, artifacts, unnecessary changes. |

## Supported Evaluation Modes

| Mode | Label key | Output | Common use |
| --- | --- | --- | --- |
| Pairwise | `label_rank` with two candidates | `{"rank": [1, 2]}` | RPO rewards and preference evaluation. |
| Listwise | `label_rank` with N candidates | `{"rank": [1, 3, 2]}` | Offline ranking benchmarks. |
| Pointwise | `label_score` | `{"score": 4}` | Ablations or scalar-style grading. |

Pairwise is implemented as the two-candidate case of listwise ranking. The rank array is aligned to the original candidate order, so `[2, 1]` means the second image is preferred.

## What This Repo Optimizes For

This release is intentionally narrow:

- FLUX text-to-image RPO.
- Qwen-Image-Edit RPO.
- Auto-Rubric generation, grading, reward plumbing, and reusable rubric files.

It does not ship large datasets, generated training images, or VLM/model checkpoints. Those are configured externally through the scripts and YAML files.

