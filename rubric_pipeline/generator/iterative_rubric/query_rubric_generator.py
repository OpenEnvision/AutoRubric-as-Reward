# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""Query-specific rubric generation for vision Auto-Rubric.

This module keeps an iterative Auto-Rubric loop and specializes the
prompts and data handling for visual generation tasks:

- Text-to-image: pointwise scoring and listwise ranking. Pairwise is the
  two-image special case of listwise.
- Image editing: pointwise scoring and listwise ranking, with an optional
  original/base image followed by one or more edited outputs.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from rubric_pipeline.graders.schema import (
    GraderMode,
    GraderRankCallback,
    GraderScoreCallback,
)
from rubric_pipeline.models.base_chat_model import BaseChatModel
from rubric_pipeline.models.schema.oai.message import ChatMessage
from rubric_pipeline.models.schema.prompt_template import LanguageEnum, PromptTemplate
from rubric_pipeline.utils.vision import (
    TASK_IMAGE_EDIT,
    TASK_T2I,
    add_images_to_messages,
    count_outputs,
    extract_image_paths,
    has_base_image,
    is_image_edit_task,
    normalize_task_type,
)

# pylint: disable=line-too-long


# ========== Text-to-Image Prompts ==========

T2I_POINTWISE_GENERATION_PROMPT_ZH = """
请基于输入的文本提示、生成图片和标注分数，生成**恰好 {generate_number} 个**用于单图评分的视觉评估 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: Pointwise，对一张生成图片独立评分，分数范围为 {min_score}-{max_score}，输出必须是整数
- 标准应覆盖文本提示遵循度、主体/属性/关系完整性、视觉质量、伪影控制和整体美感
- 标准必须具体、可操作，并能区分不同分数档位
- 只输出评分标准，不要直接给当前图片打分
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个评分标准的详细描述",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些评分标准的原因"
}}
"""

T2I_POINTWISE_GENERATION_PROMPT_EN = """
Based on the text prompt, the generated image, and the labeled score, generate **exactly {generate_number}** visual evaluation rubrics for pointwise scoring.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: Pointwise. Score one generated image independently on an integer scale from {min_score} to {max_score}
- Cover prompt adherence, subject/attribute/relation completeness, visual quality, artifact control, and overall aesthetics
- Each rubric must be concrete, actionable, and able to distinguish score levels
- Do not score the current image; only output reusable scoring criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "Detailed description of the first scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

T2I_LISTWISE_GENERATION_PROMPT_ZH = """
请基于输入的文本提示、{num_outputs} 张生成图片及标注排序，生成**恰好 {generate_number} 个**用于图片排序的视觉评估 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: {ranking_mode}，对多张文本生成图片进行相对质量排序；rank=1 表示最好
- 分析高质量与低质量图片在提示遵循度、细节完整性、真实感、构图、伪影和美学上的差异
- 标准应能对所有候选图片建立清晰质量梯度，避免位置偏见
- 只输出排序标准，不要直接给当前图片排序
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个排序标准的详细描述",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些排序标准的原因"
}}
"""

T2I_LISTWISE_GENERATION_PROMPT_EN = """
Based on the text prompt, {num_outputs} generated images, and the labeled ranking, generate **exactly {generate_number}** visual rubrics for ranking image quality.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: {ranking_mode}. Rank multiple text-to-image outputs by relative quality; rank=1 means best
- Analyze differences between strong and weak images in prompt adherence, detail completeness, realism, composition, artifacts, and aesthetics
- The criteria must establish a clear quality gradient across all candidate images and avoid position bias
- Do not rank the current images; only output reusable ranking criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "Detailed description of the first ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

T2I_POINTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的生成图片进行 pointwise 评分。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

评分范围: {min_score} 到 {max_score}，必须输出整数。

## 输出格式
{{
    "score": 分数值,
    "reason": "评分依据，说明图片如何符合或不符合各项标准"
}}
"""

T2I_POINTWISE_EVALUATION_PROMPT_EN = """
Score the generated image using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

Score range: {min_score} to {max_score}. The score must be an integer.

## Output Format
{{
    "score": score_value,
    "reason": "Reasoning explaining how the image satisfies or violates the rubrics"
}}
"""

T2I_LISTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的 {num_outputs} 张生成图片进行 {ranking_mode} 排序。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

## 任务要求
- 只对候选生成图片排序，rank=1 表示质量最好，rank={num_outputs} 表示最差
- rank 数组必须恰好包含 {num_outputs} 个值，并按图片原始顺序对应 Image 1、Image 2 ...
- rank 值必须是从 1 到 {num_outputs} 的连续整数，不能并列
- 判断时重点关注文本提示遵循度、细节完整性、真实感、伪影控制、构图和美学
- 不要受图片排列顺序影响

## 输出格式
{{
    "rank": [每张候选图片对应的rank值],
    "reason": "详细说明每张图片的质量判断和排序依据"
}}
"""

T2I_LISTWISE_EVALUATION_PROMPT_EN = """
Rank the {num_outputs} generated images using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

## Requirements
- Rank only the candidate generated images. rank=1 is best and rank={num_outputs} is worst
- The rank array must contain exactly {num_outputs} values, corresponding to Image 1, Image 2, ... in the original order
- Rank values must be consecutive integers from 1 to {num_outputs}; ties are not allowed
- Focus on prompt adherence, detail completeness, realism, artifact control, composition, and aesthetics
- Do not let image order influence your judgment

## Output Format
{{
    "rank": [rank_value_for_each_candidate_image],
    "reason": "Detailed explanation of the quality assessment and ranking"
}}
"""

T2I_POINTWISE_REVISION_PROMPT_ZH = """
之前生成的 pointwise 评分标准验证失败。请基于输入图片、标注分数和反馈，生成**恰好 {generate_number} 个**改进后的评分标准。
{task_description_section}
{sample_content}

## 之前的评分标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 明确为什么旧标准无法给出正确分数
- 强化提示遵循、主体/属性/关系、视觉自然度、伪影和美学的可判别性
- 标准要能稳定区分 {min_score}-{max_score} 的整数分数档位

## 输出格式
{{
    "rubrics": [
        "改进后的第一个评分标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

T2I_POINTWISE_REVISION_PROMPT_EN = """
The previous pointwise scoring rubrics failed validation. Generate **exactly {generate_number}** improved scoring rubrics based on the image, labeled score, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Identify why the previous rubrics did not lead to the correct score
- Improve discriminative power for prompt adherence, subjects/attributes/relations, visual naturalness, artifacts, and aesthetics
- The rubrics must reliably distinguish integer score levels from {min_score} to {max_score}

## Output Format
{{
    "rubrics": [
        "First improved scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""

T2I_LISTWISE_REVISION_PROMPT_ZH = """
之前生成的排序标准验证失败。请基于输入的 {num_outputs} 张图片、标注排序和反馈，生成**恰好 {generate_number} 个**改进后的排序标准。
{task_description_section}
{sample_content}

## 之前的排序标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 分析期望排序与实际排序的差异
- 强化能够区分所有候选图片质量层次的标准
- 标准应减少位置偏见和主观歧义，并覆盖提示遵循、真实感、伪影、构图、美学等维度

## 输出格式
{{
    "rubrics": [
        "改进后的第一个排序标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

T2I_LISTWISE_REVISION_PROMPT_EN = """
The previous ranking rubrics failed validation. Generate **exactly {generate_number}** improved ranking rubrics based on the {num_outputs} images, labeled ranking, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Analyze the gap between the expected and actual ranking
- Strengthen criteria that distinguish all candidate image quality levels
- Reduce position bias and ambiguity while covering prompt adherence, realism, artifacts, composition, and aesthetics

## Output Format
{{
    "rubrics": [
        "First improved ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""


# ========== Image Editing Prompts ==========

EDIT_POINTWISE_GENERATION_PROMPT_ZH = """
请基于编辑指令、原图（如已提供）、编辑结果和标注分数，生成**恰好 {generate_number} 个**用于图像编辑结果 pointwise 评分的 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: Pointwise，对一个编辑结果独立评分，分数范围为 {min_score}-{max_score}，输出必须是整数
- 若提供 Image BASE，必须以原图为参照；若未提供，则根据编辑指令和编辑结果评估
- 标准应覆盖编辑指令执行、原图内容保真、局部/全局一致性、自然融合、伪影控制和美学质量
- 只输出评分标准，不要直接给当前编辑结果打分
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个图像编辑评分标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些标准的原因"
}}
"""

EDIT_POINTWISE_GENERATION_PROMPT_EN = """
Based on the edit instruction, the original image if provided, the edited result, and the labeled score, generate **exactly {generate_number}** rubrics for pointwise image-edit scoring.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: Pointwise. Score one edited image independently on an integer scale from {min_score} to {max_score}
- If Image BASE is provided, use it as the reference; otherwise judge from the instruction and edited image
- Cover instruction execution, fidelity to the source image, local/global consistency, natural blending, artifact control, and aesthetic quality
- Do not score the current edit; only output reusable scoring criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "First image-edit scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

EDIT_LISTWISE_GENERATION_PROMPT_ZH = """
请基于编辑指令、原图（如已提供）、{num_outputs} 个编辑结果及标注排序，生成**恰好 {generate_number} 个**用于图像编辑结果排序的 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: {ranking_mode}，只对编辑结果排序；rank=1 表示最佳编辑结果
- 若提供 Image BASE，必须以原图为基准比较每个编辑结果；若未提供，则根据编辑指令和结果本身比较
- 标准应区分编辑指令执行、原图保真、未编辑区域保持、边界融合、光照/透视一致、伪影控制和美学质量
- 只输出排序标准，不要直接给当前编辑结果排序
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个图像编辑排序标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些排序标准的原因"
}}
"""

EDIT_LISTWISE_GENERATION_PROMPT_EN = """
Based on the edit instruction, the original image if provided, {num_outputs} edited results, and the labeled ranking, generate **exactly {generate_number}** rubrics for ranking image edits.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: {ranking_mode}. Rank only the edited results; rank=1 means the best edit
- If Image BASE is provided, use it as the reference for every edited result; otherwise compare using the edit instruction and outputs
- Distinguish instruction execution, source fidelity, preservation of untouched regions, boundary blending, lighting/perspective consistency, artifacts, and aesthetic quality
- Do not rank the current edits; only output reusable ranking criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "First image-edit ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

EDIT_POINTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的图像编辑结果进行 pointwise 评分。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

评分范围: {min_score} 到 {max_score}，必须输出整数。

## 输出格式
{{
    "score": 分数值,
    "reason": "评分依据，说明编辑结果如何满足或违反各项标准"
}}
"""

EDIT_POINTWISE_EVALUATION_PROMPT_EN = """
Score the edited image using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

Score range: {min_score} to {max_score}. The score must be an integer.

## Output Format
{{
    "score": score_value,
    "reason": "Reasoning explaining how the edit satisfies or violates the rubrics"
}}
"""

EDIT_LISTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的 {num_outputs} 个编辑结果进行 {ranking_mode} 排序。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

## 任务要求
- 只对编辑结果排序，不要把 Image BASE 纳入 rank
- rank=1 表示最佳编辑结果，rank={num_outputs} 表示最差
- rank 数组必须恰好包含 {num_outputs} 个值，并按编辑结果原始顺序对应 Edited Image 1、Edited Image 2 ...
- rank 值必须是从 1 到 {num_outputs} 的连续整数，不能并列
- 若提供 Image BASE，所有判断都必须参考原图；同时关注指令执行、原图保真、自然融合和伪影控制
- 不要受图片排列顺序影响

## 输出格式
{{
    "rank": [每个编辑结果对应的rank值],
    "reason": "详细说明每个编辑结果与原图/指令的匹配程度和排序依据"
}}
"""

EDIT_LISTWISE_EVALUATION_PROMPT_EN = """
Rank the {num_outputs} edited results using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

## Requirements
- Rank only the edited results. Do not include Image BASE in the rank array
- rank=1 is the best edit and rank={num_outputs} is the worst
- The rank array must contain exactly {num_outputs} values, corresponding to Edited Image 1, Edited Image 2, ... in the original order
- Rank values must be consecutive integers from 1 to {num_outputs}; ties are not allowed
- If Image BASE is provided, all judgments must reference it; also consider instruction execution, source fidelity, natural blending, and artifact control
- Do not let image order influence your judgment

## Output Format
{{
    "rank": [rank_value_for_each_edited_result],
    "reason": "Detailed explanation of each edit's match to the source/instruction and the final ranking"
}}
"""

EDIT_POINTWISE_REVISION_PROMPT_ZH = """
之前生成的图像编辑 pointwise 评分标准验证失败。请基于输入、标注分数和反馈，生成**恰好 {generate_number} 个**改进后的评分标准。
{task_description_section}
{sample_content}

## 之前的评分标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 分析旧标准为什么无法给出正确分数
- 强化指令执行、原图保真、未编辑区域保持、自然融合、伪影控制和美学质量的分数区分能力
- 标准要能稳定区分 {min_score}-{max_score} 的整数分数档位

## 输出格式
{{
    "rubrics": [
        "改进后的第一个图像编辑评分标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

EDIT_POINTWISE_REVISION_PROMPT_EN = """
The previous pointwise image-edit scoring rubrics failed validation. Generate **exactly {generate_number}** improved scoring rubrics based on the inputs, labeled score, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Analyze why the previous rubrics did not lead to the correct score
- Improve score-level discrimination for instruction execution, source fidelity, preservation of untouched regions, natural blending, artifact control, and aesthetics
- The rubrics must reliably distinguish integer score levels from {min_score} to {max_score}

## Output Format
{{
    "rubrics": [
        "First improved image-edit scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""

EDIT_LISTWISE_REVISION_PROMPT_ZH = """
之前生成的图像编辑排序标准验证失败。请基于输入的 {num_outputs} 个编辑结果、标注排序和反馈，生成**恰好 {generate_number} 个**改进后的排序标准。
{task_description_section}
{sample_content}

## 之前的排序标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 分析期望排序与实际排序的差异
- 强化能够区分编辑结果质量层次的标准，尤其是指令执行、原图保真、局部融合、伪影和美学
- 如果提供原图，标准必须明确要求对 Image BASE 进行比较

## 输出格式
{{
    "rubrics": [
        "改进后的第一个图像编辑排序标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

EDIT_LISTWISE_REVISION_PROMPT_EN = """
The previous image-edit ranking rubrics failed validation. Generate **exactly {generate_number}** improved ranking rubrics based on the {num_outputs} edited results, labeled ranking, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Analyze the gap between the expected and actual ranking
- Strengthen criteria that distinguish edit quality levels, especially instruction execution, source fidelity, local blending, artifacts, and aesthetics
- If the original image is provided, the criteria must explicitly require comparison with Image BASE

## Output Format
{{
    "rubrics": [
        "First improved image-edit ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""


def _prompt_pair(
    task_type: str,
    mode: GraderMode,
    prompt_kind: str,
) -> tuple[str, str]:
    task = normalize_task_type(task_type)
    is_pointwise = mode == GraderMode.POINTWISE

    if task == TASK_T2I and prompt_kind == "generation" and is_pointwise:
        return T2I_POINTWISE_GENERATION_PROMPT_ZH, T2I_POINTWISE_GENERATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "generation":
        return T2I_LISTWISE_GENERATION_PROMPT_ZH, T2I_LISTWISE_GENERATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "evaluation" and is_pointwise:
        return T2I_POINTWISE_EVALUATION_PROMPT_ZH, T2I_POINTWISE_EVALUATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "evaluation":
        return T2I_LISTWISE_EVALUATION_PROMPT_ZH, T2I_LISTWISE_EVALUATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "revision" and is_pointwise:
        return T2I_POINTWISE_REVISION_PROMPT_ZH, T2I_POINTWISE_REVISION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "revision":
        return T2I_LISTWISE_REVISION_PROMPT_ZH, T2I_LISTWISE_REVISION_PROMPT_EN

    if task == TASK_IMAGE_EDIT and prompt_kind == "generation" and is_pointwise:
        return EDIT_POINTWISE_GENERATION_PROMPT_ZH, EDIT_POINTWISE_GENERATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "generation":
        return EDIT_LISTWISE_GENERATION_PROMPT_ZH, EDIT_LISTWISE_GENERATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "evaluation" and is_pointwise:
        return EDIT_POINTWISE_EVALUATION_PROMPT_ZH, EDIT_POINTWISE_EVALUATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "evaluation":
        return EDIT_LISTWISE_EVALUATION_PROMPT_ZH, EDIT_LISTWISE_EVALUATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "revision" and is_pointwise:
        return EDIT_POINTWISE_REVISION_PROMPT_ZH, EDIT_POINTWISE_REVISION_PROMPT_EN
    return EDIT_LISTWISE_REVISION_PROMPT_ZH, EDIT_LISTWISE_REVISION_PROMPT_EN


def _build_template(task_type: str, mode: GraderMode, prompt_kind: str) -> PromptTemplate:
    zh_prompt, en_prompt = _prompt_pair(task_type, mode, prompt_kind)
    include_system = prompt_kind in {"generation", "revision"}
    zh_messages = []
    en_messages = []
    if include_system:
        zh_messages.append(ChatMessage(role="system", content="你是一个专业的视觉评估标准制定专家。"))
        en_messages.append(ChatMessage(role="system", content="You are a professional visual evaluation criteria expert."))
    zh_messages.append(ChatMessage(role="user", content=textwrap.dedent(zh_prompt).strip()))
    en_messages.append(ChatMessage(role="user", content=textwrap.dedent(en_prompt).strip()))
    return PromptTemplate(messages={LanguageEnum.ZH: zh_messages, LanguageEnum.EN: en_messages})


def get_generation_template(
    grader_mode: GraderMode | str,
    task_type: str | None = TASK_T2I,
) -> PromptTemplate:
    mode = GraderMode(grader_mode)
    return _build_template(normalize_task_type(task_type), mode, "generation")


def get_evaluation_template(
    grader_mode: GraderMode | str,
    task_type: str | None = TASK_T2I,
) -> PromptTemplate:
    mode = GraderMode(grader_mode)
    return _build_template(normalize_task_type(task_type), mode, "evaluation")


def get_revision_template(
    grader_mode: GraderMode | str,
    task_type: str | None = TASK_T2I,
) -> PromptTemplate:
    mode = GraderMode(grader_mode)
    return _build_template(normalize_task_type(task_type), mode, "revision")


# Backward-compatible defaults.  New code should call get_*_template with an
# explicit task_type.
POINTWISE_GENERATION_TEMPLATE = get_generation_template(GraderMode.POINTWISE, TASK_T2I)
LISTWISE_GENERATION_TEMPLATE = get_generation_template(GraderMode.LISTWISE, TASK_T2I)
POINTWISE_EVALUATION_TEMPLATE = get_evaluation_template(GraderMode.POINTWISE, TASK_T2I)
LISTWISE_EVALUATION_TEMPLATE = get_evaluation_template(GraderMode.LISTWISE, TASK_T2I)
POINTWISE_REVISION_TEMPLATE = get_revision_template(GraderMode.POINTWISE, TASK_T2I)
LISTWISE_REVISION_TEMPLATE = get_revision_template(GraderMode.LISTWISE, TASK_T2I)


class RubricGenerationOutput(BaseModel):
    """Structured output expected from rubric generation/revision."""

    rubrics: List[str] = Field(description="List of generated rubrics")
    reason: str = Field(description="Reasoning for the generated rubrics")


class QuerySpecificRubricGenerator:
    """Generate, validate, and revise query-specific vision rubrics."""

    def __init__(
        self,
        model: BaseChatModel,
        grader_mode: GraderMode | str = GraderMode.POINTWISE,
        generate_number: int = 3,
        max_retries: int = 5,
        max_epochs: int = 3,
        min_score: int = 0,
        max_score: int = 4,
        language: LanguageEnum | str = LanguageEnum.ZH,
        task_description: str | None = None,
        task_type: str | None = TASK_T2I,
    ):
        self.model = model
        self.grader_mode = GraderMode(grader_mode)
        self.language = LanguageEnum(language) if isinstance(language, str) else language
        self.task_type = normalize_task_type(task_type)
        self.generate_number = generate_number
        self.max_retries = max_retries
        self.max_epochs = max_epochs
        self.min_score = min_score
        self.max_score = max_score
        self.task_description = task_description

        self.generation_template = get_generation_template(self.grader_mode, self.task_type)
        self.evaluation_template = get_evaluation_template(self.grader_mode, self.task_type)
        self.revision_template = get_revision_template(self.grader_mode, self.task_type)

        logger.info(
            "QuerySpecificRubricGenerator initialized: mode={}, task_type={}, language={}",
            self.grader_mode.value,
            self.task_type,
            self.language.value,
        )

    async def generate_iterative(self, data: dict) -> Dict[str, Any]:
        rubrics = await self.generate(data)
        if not rubrics:
            return {
                "rubrics": [],
                "rubric_valid": False,
                "rubric_epoch": "0",
                "evaluation_result": {},
            }

        evaluation_result = {}
        for epoch in range(self.max_epochs):
            evaluation_result = await self.aevaluate(data, rubrics)
            is_correct = self.validate(data, evaluation_result)
            logger.debug(f"Epoch {epoch}: correctness = {is_correct}")

            if is_correct:
                return {
                    "rubrics": rubrics,
                    "rubric_valid": True,
                    "rubric_epoch": str(epoch),
                    "evaluation_result": evaluation_result,
                }

            feedback = self.generate_feedback(data, evaluation_result)
            revised_rubrics = await self.revise(data, rubrics, feedback)
            if not revised_rubrics:
                break
            rubrics = revised_rubrics

        return {
            "rubrics": rubrics,
            "rubric_valid": False,
            "rubric_epoch": str(self.max_epochs),
            "evaluation_result": evaluation_result,
        }

    async def generate(self, data: dict) -> List[str]:
        sample_content = self._format_data_context(data)
        image_paths = extract_image_paths(data, self.task_type)
        num_outputs = max(1, count_outputs(data, self.task_type))
        task_description_section = self._format_task_description_section()

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def generate_rubrics():
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
            )
            messages = self.generation_template.format(language=self.language, **params)
            messages = add_images_to_messages(
                messages,
                image_paths,
                self.task_type,
                has_base=has_base_image(data, self.task_type),
            )
            chat_response = await self.model.achat(
                messages=messages,
                structured_model=RubricGenerationOutput,
            )

            if not chat_response.parsed or "rubrics" not in chat_response.parsed:
                raise ValueError(f"Rubric generation returned invalid JSON: {chat_response.parsed}")

            rubrics = chat_response.parsed["rubrics"]
            if not rubrics:
                raise ValueError("No rubrics generated")
            return rubrics

        try:
            rubrics = await generate_rubrics()
            logger.debug(f"Generated {len(rubrics)} rubrics")
            return rubrics
        except Exception as e:
            logger.error(f"Failed to generate rubrics after {self.max_retries} attempts: {e}")
            return []

    async def aevaluate(self, data: dict, rubrics: List[str]) -> Dict[str, Any]:
        if self.grader_mode == GraderMode.POINTWISE:
            return await self._evaluate_pointwise(data, rubrics)
        return await self._evaluate_listwise(data, rubrics)

    def validate(self, data: dict, evaluation_result: Dict[str, Any]) -> bool:
        if self.grader_mode == GraderMode.POINTWISE:
            return self._validate_pointwise(data, evaluation_result)
        return self._validate_listwise(data, evaluation_result)

    async def revise(self, data: dict, rubrics: List[str], feedback: str) -> List[str]:
        sample_content = self._format_data_context(data)
        rubrics_text = self._format_rubrics_text(rubrics)
        image_paths = extract_image_paths(data, self.task_type)
        num_outputs = max(1, count_outputs(data, self.task_type))
        task_description_section = self._format_task_description_section()

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def revise_rubrics():
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
            )
            params.update({"rubrics": rubrics_text, "feedback": feedback})
            messages = self.revision_template.format(language=self.language, **params)
            messages = add_images_to_messages(
                messages,
                image_paths,
                self.task_type,
                has_base=has_base_image(data, self.task_type),
            )
            chat_response = await self.model.achat(
                messages=messages,
                structured_model=RubricGenerationOutput,
            )

            if not chat_response.parsed or "rubrics" not in chat_response.parsed:
                raise ValueError(f"Rubric revision returned invalid JSON: {chat_response.parsed}")

            revised_rubrics = chat_response.parsed["rubrics"]
            if not revised_rubrics:
                raise ValueError("No revised rubrics generated")
            return revised_rubrics

        try:
            revised_rubrics = await revise_rubrics()
            logger.debug(f"Revised {len(revised_rubrics)} rubrics")
            return revised_rubrics
        except Exception as e:
            logger.error(f"Failed to revise rubrics after {self.max_retries} attempts: {e}")
            return []

    def generate_feedback(self, data: dict, evaluation_result: Dict[str, Any]) -> str:
        if self.grader_mode == GraderMode.POINTWISE:
            return self._generate_pointwise_feedback(data, evaluation_result)
        return self._generate_listwise_feedback(data, evaluation_result)

    async def _evaluate_pointwise(self, data: dict, rubrics: List[str]) -> Dict[str, Any]:
        rubrics_text = self._format_rubrics_text(rubrics)
        sample_content = self._format_data_context(data)
        image_paths = extract_image_paths(data, self.task_type)
        num_outputs = max(1, count_outputs(data, self.task_type))
        task_description_section = self._format_task_description_section()

        try:
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
            )
            params["rubrics"] = rubrics_text
            messages = self.evaluation_template.format(language=self.language, **params)
            messages = add_images_to_messages(
                messages,
                image_paths,
                self.task_type,
                has_base=has_base_image(data, self.task_type),
            )
            response_obj = await self.model.achat(
                messages=messages,
                structured_model=GraderScoreCallback,
            )

            score = self.min_score
            if response_obj.parsed and "score" in response_obj.parsed:
                score = int(round(float(response_obj.parsed["score"])))
                score = max(self.min_score, min(self.max_score, score))

            return {"scores": [score]}

        except Exception as e:
            logger.error(f"Pointwise evaluation failed: {e}")
            return {"scores": [self.min_score]}

    async def _evaluate_listwise(self, data: dict, rubrics: List[str]) -> Dict[str, Any]:
        rubrics_text = self._format_rubrics_text(rubrics)
        sample_content = self._format_data_context(data)
        image_paths = extract_image_paths(data, self.task_type)
        num_outputs = count_outputs(data, self.task_type)
        task_description_section = self._format_task_description_section()

        try:
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
            )
            params["rubrics"] = rubrics_text
            messages = self.evaluation_template.format(language=self.language, **params)
            messages = add_images_to_messages(
                messages,
                image_paths,
                self.task_type,
                has_base=has_base_image(data, self.task_type),
            )
            response_obj = await self.model.achat(
                messages=messages,
                structured_model=GraderRankCallback,
            )

            if response_obj.parsed and "rank" in response_obj.parsed:
                rank_values = [int(v) for v in response_obj.parsed["rank"]]
                if len(rank_values) == num_outputs:
                    if len(set(rank_values)) != len(rank_values):
                        logger.warning(f"Duplicate rank values detected: {rank_values}")
                    return {"rank_values": rank_values}
                logger.warning(
                    "Invalid rank length: got {} ranks {} for {} outputs",
                    len(rank_values),
                    rank_values,
                    num_outputs,
                )

            return {"rank_values": []}

        except Exception as e:
            logger.error(f"Listwise evaluation failed: {e}")
            return {"rank_values": []}

    def _validate_pointwise(self, data: dict, evaluation_result: Dict[str, Any]) -> bool:
        scores = evaluation_result.get("scores", [])
        if not scores or len(scores) != 1:
            return False

        expected_score = data.get("label_score")
        if expected_score is not None:
            return int(round(float(scores[0]))) == int(round(float(expected_score)))

        return False

    def _validate_listwise(self, data: dict, evaluation_result: Dict[str, Any]) -> bool:
        rank_values = evaluation_result.get("rank_values", [])
        expected_ranks = data.get("label_rank", [])

        if not rank_values or not expected_ranks:
            return False

        if len(rank_values) != len(expected_ranks):
            return False

        expected_order = self._get_relative_order(expected_ranks)
        predicted_order = self._get_relative_order(rank_values)
        logger.debug(f"Expected ranks: {expected_ranks}, order: {expected_order}")
        logger.debug(f"Predicted ranks: {rank_values}, order: {predicted_order}")
        return expected_order == predicted_order

    @staticmethod
    def _get_relative_order(values: List[float]) -> List[int]:
        indexed_values = list(enumerate(values))
        indexed_values.sort(key=lambda x: x[1])
        return [idx for idx, _ in indexed_values]

    def _generate_pointwise_feedback(self, data: dict, evaluation_result: Dict[str, Any]) -> str:
        expected_score = data.get("label_score")
        actual_scores = evaluation_result.get("scores", [])
        actual_score = actual_scores[0] if actual_scores else None
        return f"Expected score: {expected_score}\nActual score: {actual_score}"

    def _generate_listwise_feedback(self, data: dict, evaluation_result: Dict[str, Any]) -> str:
        expected_ranks = data.get("label_rank", [])
        actual_ranks = evaluation_result.get("rank_values", [])
        return f"Expected ranks: {expected_ranks}\nActual ranks: {actual_ranks}"

    def _format_data_context(self, data: dict) -> str:
        query = data.get("query") or data.get("prompt") or data.get("instruction") or ""
        image_paths = extract_image_paths(data, self.task_type)
        num_outputs = count_outputs(data, self.task_type)
        ranking_mode = "Pairwise" if num_outputs == 2 else "Listwise"
        lines: list[str] = []

        if is_image_edit_task(self.task_type):
            lines.append(f"Editing instruction: {query}")
            if has_base_image(data, self.task_type):
                lines.append("Image BASE is provided as the original/reference image.")
                lines.append(f"Number of edited candidate images: {num_outputs}")
            elif image_paths:
                lines.append("Only edited image(s) are provided; no Image BASE is available.")
                lines.append(f"Number of edited candidate images: {num_outputs}")
        else:
            lines.append(f"Caption: {query}")
            if image_paths:
                lines.append(f"Number of generated candidate images: {num_outputs}")

        if self.grader_mode == GraderMode.LISTWISE:
            lines.append(f"Evaluation mode for this sample: {ranking_mode}")

        if "label_score" in data:
            lines.append(f"Expected score: {data.get('label_score')}")
        if "label_rank" in data:
            lines.append(f"Expected rank values: {data.get('label_rank')}")
        if data.get("task_type") and is_image_edit_task(self.task_type):
            lines.append(f"Edit operation tags: {data.get('task_type')}")

        return "\n".join(lines)

    def _base_template_params(
        self,
        sample_content: str,
        num_outputs: int,
        task_description_section: str,
    ) -> Dict[str, Any]:
        ranking_mode = "Pairwise" if num_outputs == 2 else "Listwise"
        return {
            "language": self.language,
            "sample_content": sample_content,
            "generate_number": self.generate_number,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "num_outputs": num_outputs,
            "ranking_mode": ranking_mode,
            "task_description_section": task_description_section,
        }

    @staticmethod
    def _format_rubrics_text(rubrics: List[str]) -> str:
        return "\n".join([f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics)])

    def _format_task_description_section(self) -> str:
        if not self.task_description:
            return ""
        if self.language == LanguageEnum.ZH:
            return f"\n## 任务场景描述\n{self.task_description}\n"
        return f"\n## Task Description\n{self.task_description}\n"
