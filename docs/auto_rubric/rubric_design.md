# Rubric Design

Auto-Rubric is data-driven, but it still needs clear task framing. The seed pairs teach the model what "better" means; the YAML `task_description` tells it which dimensions are allowed to matter.

## What A Good Rubric Does

A useful visual rubric is:

- **Observable**: the judge can check it from the images and prompt.
- **Comparative**: it helps decide which candidate is better, not just whether both are acceptable.
- **Task-specific**: it rewards the behavior you actually want.
- **Non-redundant**: each axis adds a different signal.
- **Hard to exploit**: it does not over-reward a cheap proxy such as saturation, sharpness, or decorative detail.

## T2I Rubric Dimensions

For text-to-image, start from these dimensions:

| Dimension | What to reward | What to penalize |
| --- | --- | --- |
| Prompt adherence | Required objects, counts, attributes, actions, and scene type. | Missing objects, wrong counts, ignored constraints. |
| Spatial and relational fidelity | Correct positions, interactions, containment, and relative scale. | Swapped relations, impossible layout, unrelated composition. |
| Visual coherence | Consistent anatomy, lighting, perspective, and object structure. | Warping, duplicate limbs, unstable geometry. |
| Artifact control | Clean edges, readable text when requested, natural textures. | Smears, broken text, extra objects, obvious generation artifacts. |
| Aesthetic fit | Composition, mood, style, color harmony when requested. | Generic beauty that conflicts with the prompt. |

Example `task_description`:

```yaml
task_description: |
  Compare two text-to-image outputs. Prefer the image that follows the prompt
  more faithfully, preserves requested object counts and relations, and avoids
  visual artifacts. Do not reward generic beauty when required content is missing.
```

## Image-Edit Rubric Dimensions

For image editing, the source image changes the problem. The best edit is not simply the most attractive output; it should perform the requested change while preserving unrelated content.

| Dimension | What to reward | What to penalize |
| --- | --- | --- |
| Instruction fulfillment | The requested edit is visible and complete. | Partial edit, wrong edit, over-editing. |
| Source preservation | Identity, pose, background, layout, and untouched objects stay stable. | Unrequested changes, identity drift, background rewriting. |
| Local edit quality | Edited region blends with lighting, perspective, texture, and boundaries. | Cut-and-paste edges, mismatched shadows, unnatural material. |
| Global coherence | The final image remains plausible as a whole. | Edit conflicts with scene geometry or camera viewpoint. |
| Artifact control | No distortion, halos, blurring, or broken details near the edit. | Artifacts introduced by the editing operation. |

Example `task_description`:

```yaml
task_description: |
  Compare two edited outputs against the original image and edit instruction.
  Prefer the result that completes the requested edit while preserving identity,
  layout, lighting, and all unrelated regions of the source image.
```

## Seed Pair Selection

Rubric quality depends heavily on seed examples. A small but diverse set is better than many near-duplicates.

| Goal | Include examples where |
| --- | --- |
| Teach prompt adherence | One image is prettier but misses required content. |
| Teach artifact sensitivity | One image follows the prompt but has visible defects. |
| Teach edit preservation | One edit completes the instruction but changes unrelated regions. |
| Teach fine-grained preference | Both images are plausible, but one is subtly better on a key axis. |
| Reduce position bias | Preferred images appear sometimes first and sometimes second. |

For most experiments, use 50 to 100 seed pairs. If you use fewer than 20, make them very deliberate and inspect the generated rubrics manually.

## Customizing Rubrics

You can guide the generation in three ways:

1. **Change seed data**: this is the strongest signal. Add examples from the exact domain you care about.
2. **Change `task_description`**: steer the model toward the evaluation dimensions that matter.
3. **Edit saved rubrics**: after generation, remove vague or harmful tips before reuse.

## Common Bad Rubrics

| Bad pattern | Why it hurts | Better version |
| --- | --- | --- |
| "Prefer high quality images." | Too vague; every judge already knows this. | "Prefer coherent object structure with stable anatomy, clean boundaries, and no duplicated parts." |
| "Prefer the most beautiful image." | Can override instruction fidelity. | "Use aesthetics only after required prompt content and relations are satisfied." |
| "Prefer brighter and more colorful images." | Causes saturation reward hacking. | "Reward color harmony only when it supports the requested style or scene." |
| "The edit should look good." | Does not protect the source image. | "Prefer edits that complete the requested change while preserving unrelated source-image regions." |

## Manual Review Checklist

Before saving a rubric set, read it once and ask:

- Does every rubric mention something visible or inferable?
- Does it distinguish the preferred image from the rejected one?
- Does it avoid rewarding generic prettiness over task success?
- For editing, does it explicitly protect the source image?
- Are there duplicate criteria that could be merged?
- Is the wording short enough to fit comfortably in judge prompts?

