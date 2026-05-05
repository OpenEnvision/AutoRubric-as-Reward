# Auto-Rubric Guide

This folder explains the Auto-Rubric side of Vision-Auto-Rubric: how rubrics are generated, how to choose a VLM judge, how to customize criteria for a domain, and how to reuse a verified rubric set during training.

## Start Here

| Page | When to read it |
| --- | --- |
| [Overview](overview.md) | You want the method in plain language and a map from paper concepts to code. |
| [VLM Selection](vlm_selection.md) | You need to choose between local Qwen3-VL, OpenAI-compatible GPT, Gemini-compatible endpoints, or another judge. |
| [Rubric Design](rubric_design.md) | You want to control what the rubric rewards, avoid vague criteria, or specialize to a new visual domain. |
| [Rubric Reuse](rubric_reuse.md) | You want deterministic training runs, saved rubric files, versioning, and evaluation before reuse. |
| [Workflows](workflows.md) | You want end-to-end commands for generating, testing, saving, and loading rubrics. |
| [Troubleshooting](troubleshooting.md) | The judge returns invalid JSON, ranks are unstable, images are misordered, or API calls fail. |

## Mental Model

Auto-Rubric has three objects:

- **Seed examples**: a small set of labeled visual preferences, usually pairs.
- **Rubrics**: natural-language criteria extracted from those examples and verified against the labels.
- **Judge**: a frozen VLM that receives the rubric text plus new images and returns a rank or score.

For RPO in this repo, the common path is:

```text
seed preference pairs
  -> query-specific rubric generation
  -> verification and revision
  -> categorized reusable rubric text
  -> frozen VLM judge
  -> pairwise reward tensor: [1.0, -0.1] or [-0.1, 1.0]
```

## Repository Entry Points

| Component | Path |
| --- | --- |
| CLI and reward interface | `judger.py` |
| Generation loop | `rubric_pipeline/generator/iterative_rubric/` |
| VLM grading prompts | `rubric_pipeline/generator/iterative_rubric/query_rubric_generator.py` |
| OpenAI-compatible client | `rubric_pipeline/models/openai_chat_model.py` |
| VLM configs | `rubric_pipeline/config/` |
| Saved rubric files | `rubric_pipeline/rubrics/` |

