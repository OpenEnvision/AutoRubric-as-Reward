# -*- coding: utf-8 -*-
"""
Model integrations for OpenAI-compatible judge clients.
"""

from rubric_pipeline.models.base_chat_model import BaseChatModel
from rubric_pipeline.models.openai_chat_model import OpenAIChatModel
# from rubric_pipeline.models.qwen_vl_model import QwenVLModel

__all__ = [
    "BaseChatModel",
    "OpenAIChatModel",
    # "QwenVLModel",
]
