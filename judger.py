import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.asyncio import tqdm as async_tqdm

from rubric_pipeline.generator.iterative_rubric.generator import (
    IterativeListwiseRubricsGeneratorConfig,
    IterativePointwiseRubricsGeneratorConfig,
    IterativeRubricsGenerator,
)
from rubric_pipeline.generator.iterative_rubric.query_rubric_generator import (
    get_evaluation_template,
)
from rubric_pipeline.graders.llm_grader import LLMGrader
from rubric_pipeline.graders.schema import GraderMode, GraderRank, GraderScore
from rubric_pipeline.models import OpenAIChatModel
from rubric_pipeline.models.schema.prompt_template import LanguageEnum
from rubric_pipeline.utils.vision import (
    TASK_T2I,
    count_outputs,
    extract_image_paths,
    has_base_image,
    is_image_edit_task,
    normalize_task_type,
)


NONE_VALUES = {None, "", "none", "null", "None"}


def _load_structured_file(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


class Judger:
    """Vision Auto-Rubric judger for T2I and image-editing tasks."""

    def __init__(
        self,
        config_path: str,
        seed_dataset: str | None = None,
        rubrics_file: str | None = None,
        base_url: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.config = _load_structured_file(self.config_path)
        self.config_dir = self.config_path.parent

        self.task_type = self._resolve_task_type()
        self.eval_mode = str(
            self.config.get("eval_mode")
            or self.config.get("grader_mode")
            or self.config.get("mode")
            or "pairwise",
        ).lower()
        self.grader_mode = GraderMode.POINTWISE if self.eval_mode == "pointwise" else GraderMode.LISTWISE
        self.language = LanguageEnum(self.config.get("language", "en"))
        self.min_score = int(self.config.get("min_score", 1))
        self.max_score = int(self.config.get("max_score", 5))

        self.seed_dataset = self._first_non_empty(seed_dataset, self.config.get("seed_dataset"))
        self.rubrics_file = self._first_non_empty(rubrics_file, self.config.get("rubrics_file"))
        self.base_url = self._expand_env(self._first_non_empty(base_url, self.config.get("base_url")))
        self.judger: LLMGrader | None = None

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value not in NONE_VALUES:
                return value
        return None

    @staticmethod
    def _expand_env(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    def _resolve_path(self, value: str | Path | None) -> Path | None:
        if value in NONE_VALUES:
            return None
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        project_relative = Path.cwd() / path
        if project_relative.exists():
            return project_relative
        return self.config_dir / path

    def _resolve_task_type(self) -> str:
        configured = self.config.get("task_type") or self.config.get("task")
        if configured not in NONE_VALUES:
            return normalize_task_type(str(configured))
        return normalize_task_type("edit" if "edit" in self.config_path.stem.lower() else TASK_T2I)

    def _task_description_section(self) -> str:
        task_description = self.config.get("task_description", "")
        if not task_description:
            return ""
        if self.language == LanguageEnum.ZH:
            return f"\n## 任务场景描述\n{task_description}\n"
        return f"\n## Task Description\n{task_description}\n"

    def _build_model(self) -> OpenAIChatModel:
        api_key = self._expand_env(self.config.get("api_key")) or os.getenv("OPENAI_API_KEY")
        return OpenAIChatModel(
            model=self._expand_env(self.config["model_name"]),
            api_key=api_key,
            base_url=self.base_url,
            temperature=self.config.get("temperature", 0),
        )

    async def initialize(self):
        """Build the final grader from a rubrics file or generate rubrics from data."""
        model = self._build_model()
        rubrics = self.config.get("rubrics")

        rubrics_path = self._resolve_path(self.rubrics_file)
        if rubrics_path:
            rubrics = rubrics_path.read_text(encoding="utf-8")
        elif not rubrics:
            seed_path = self._resolve_path(self.seed_dataset)
            if not seed_path:
                raise ValueError("Config must provide `rubrics`, `rubrics_file`, or `seed_dataset`.")
            rubrics_dataset = _load_structured_file(seed_path)
            generator_config_kwargs = {
                "grader_name": self.config["grader_name"],
                "model": model,
                "language": self.language,
                "query_specific_generate_number": int(self.config.get("query_specific_generate_number", 1)),
                "enable_categorization": bool(self.config.get("enable_categorization", False)),
                "categories_number": int(self.config.get("categories_number", 5)),
                "task_description": self.config.get("task_description"),
                "task_type": self.task_type,
                "max_retries": int(self.config.get("max_retries", 5)),
                "max_epochs": int(self.config.get("max_epochs", 5)),
            }
            if self.grader_mode == GraderMode.POINTWISE:
                generator_config = IterativePointwiseRubricsGeneratorConfig(
                    min_score=self.min_score,
                    max_score=self.max_score,
                    **generator_config_kwargs,
                )
            else:
                generator_config = IterativeListwiseRubricsGeneratorConfig(**generator_config_kwargs)
            generator = IterativeRubricsGenerator(generator_config)
            generated_grader = await generator.generate(rubrics_dataset)
            rubrics = generated_grader.kwargs.get("rubrics", "")
            print("Rubrics generated:")
            print(rubrics)

        self.judger = LLMGrader(
            name=self.config["grader_name"],
            model=model,
            mode=self.grader_mode,
            rubrics=rubrics,
            language=self.language,
            template=get_evaluation_template(self.grader_mode, self.task_type),
            task_type=self.task_type,
            min_score=self.min_score,
            max_score=self.max_score,
            task_description_section=self._task_description_section(),
        )
        return self

    def _format_sample_content(self, item: dict) -> str:
        query = item.get("query") or item.get("prompt") or item.get("instruction") or ""
        image_paths = extract_image_paths(item, self.task_type)
        num_outputs = count_outputs(item, self.task_type)
        lines: list[str] = []

        if is_image_edit_task(self.task_type):
            lines.append(f"Editing instruction: {query}")
            lines.append(f"Number of edited candidate images: {num_outputs}")
            if has_base_image(item, self.task_type):
                lines.append("Image BASE is provided as the original/reference image.")
        else:
            lines.append(f"Caption: {query}")
            lines.append(f"Number of generated candidate images: {num_outputs}")

        return "\n".join(lines)

    async def _evaluate_item(self, item: dict) -> GraderScore | GraderRank:
        if self.judger is None:
            raise RuntimeError("Judger not initialized. Call `await judger.initialize()` first.")

        image_paths = extract_image_paths(item, self.task_type)
        return await self.judger.aevaluate(
            image_paths=image_paths,
            has_base_image=has_base_image(item, self.task_type),
            sample_content=self._format_sample_content(item),
        )

    async def _evaluate_single(self, item: dict, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                return await self._evaluate_item(item)
            except Exception as e:
                print(f"Error evaluating item: {e}")
                return None

    @staticmethod
    def _rank_to_rewards(rank: list[int]) -> list[float]:
        if not rank:
            return []
        if len(rank) == 1:
            return [1.0]
        if len(rank) == 2:
            return [1.0 if value == 1 else -0.1 for value in rank]

        worst = max(rank)
        if worst <= 1:
            return [1.0 for _ in rank]
        return [1.0 - ((value - 1) / (worst - 1)) * 1.1 for value in rank]

    async def get_reward_single(self, item: dict, device) -> torch.Tensor:
        result = await self._evaluate_item(item)
        if isinstance(result, GraderScore):
            return torch.tensor([float(result.score)], device=device)
        return torch.tensor(self._rank_to_rewards(result.rank), device=device)

    async def get_reward(self, data: list[dict], concurrency_limit: int = 50) -> list[Any]:
        if self.judger is None:
            raise RuntimeError("Judger not initialized. Call `await judger.initialize()` first.")

        semaphore = asyncio.Semaphore(concurrency_limit)
        tasks = [self._evaluate_single(item, semaphore) for item in data]
        results = await async_tqdm.gather(*tasks, desc="Getting rewards")
        outputs = []
        for result in results:
            if isinstance(result, GraderScore):
                outputs.append(result.score)
            elif isinstance(result, GraderRank):
                outputs.append(result.rank)
            else:
                outputs.append(None)
        return outputs

    async def evaluate(self, dataset_path: str, concurrency_limit: int = 50) -> float:
        if self.judger is None:
            raise RuntimeError("Judger not initialized. Call `await judger.initialize()` first.")

        dataset = _load_structured_file(dataset_path)
        predictions = await self.get_reward(dataset, concurrency_limit=concurrency_limit)

        correct = 0
        for item, prediction in zip(dataset, predictions):
            if prediction is None:
                continue
            if self.grader_mode == GraderMode.POINTWISE:
                expected = item.get("label_score")
                if expected is not None and int(round(float(prediction))) == int(round(float(expected))):
                    correct += 1
            else:
                if prediction == item.get("label_rank"):
                    correct += 1

        total = len(dataset)
        accuracy = correct / total if total > 0 else 0.0
        print("-" * 30)
        print(f"Task:        {self.task_type}")
        print(f"Mode:        {self.eval_mode}")
        print(f"Total Tests: {total}")
        print(f"Correct:     {correct}")
        print(f"Accuracy:    {accuracy:.2%}")
        print("-" * 30)
        return accuracy


async def main(args):
    judger = await Judger(
        config_path=args.config_path,
        seed_dataset=args.seed_dataset,
        rubrics_file=args.rubrics_file,
        base_url=args.base_url,
    ).initialize()
    await judger.evaluate(
        dataset_path=args.test_dataset,
        concurrency_limit=args.concurrency_limit,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config yaml/json")
    parser.add_argument("--rubrics_file", type=str, default=None, help="Optional rubrics text file")
    parser.add_argument("--seed_dataset", type=str, default=None, help="Optional seed dataset for Auto-Rubric generation")
    parser.add_argument("--test_dataset", type=str, required=True, help="Dataset path for evaluation")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--concurrency_limit", type=int, default=50, help="Concurrent evaluation calls")
    asyncio.run(main(parser.parse_args()))
