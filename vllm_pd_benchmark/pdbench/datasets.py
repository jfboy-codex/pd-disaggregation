from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Sample


LONGBENCH_TASKS = (
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
    "lsht", "passage_count", "passage_retrieval_en",
    "passage_retrieval_zh", "lcc", "repobench-p",
)

# These task-aware templates preserve both LongBench `input` and `context`.
# They follow the intent of the official LongBench v1 prompt configuration while
# keeping the benchmark independent of the retired Hugging Face dataset script.
LONGBENCH_PROMPTS: dict[str, str] = {
    "narrativeqa": "You are given a story. Answer the question as concisely as possible.\n\nStory:\n{context}\n\nQuestion: {input}\nAnswer:",
    "qasper": "You are given a scientific article. Answer the question using the article. If it cannot be answered, say so.\n\nArticle:\n{context}\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en": "Read the following text and answer the question briefly.\n\nText:\n{context}\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "请阅读以下材料并简洁回答问题。\n\n材料：\n{context}\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the provided passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the provided passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the multi-hop question based on the provided passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:",
    "dureader": "请根据给定材料回答问题。\n\n材料：\n{context}\n\n问题：{input}\n回答：",
    "gov_report": "Write a concise summary of the following government report.\n\nReport:\n{context}\n\nSummary:",
    "qmsum": "You are given a meeting transcript. Answer the query by summarizing the relevant information.\n\nTranscript:\n{context}\n\nQuery: {input}\nAnswer:",
    "multi_news": "Write a concise summary of the following news articles.\n\nArticles:\n{context}\n\nSummary:",
    "vcsum": "请对下面的会议记录进行简洁总结。\n\n会议记录：\n{context}\n\n总结：",
    "trec": "Classify the question into the most appropriate category. Use the examples/context below and output only the category.\n\nExamples/context:\n{context}\n\nQuestion: {input}\nCategory:",
    "triviaqa": "Answer the trivia question based on the provided evidence.\n\nEvidence:\n{context}\n\nQuestion: {input}\nAnswer:",
    "samsum": "Summarize the following dialogue.\n\nDialogue:\n{context}\n\nSummary:",
    "lsht": "请根据示例或上下文判断新闻类别，只输出类别。\n\n示例或上下文：\n{context}\n\n待分类新闻：{input}\n类别：",
    "passage_count": "Count how many distinct passages are in the following text. Output only the number.\n\n{context}\n\nAnswer:",
    "passage_retrieval_en": "Find the passage that answers the question and output its passage identifier.\n\nPassages:\n{context}\n\nQuestion: {input}\nPassage identifier:",
    "passage_retrieval_zh": "请找出能够回答问题的段落，并只输出段落编号。\n\n段落：\n{context}\n\n问题：{input}\n段落编号：",
    "lcc": "Complete the code below. Output only the code continuation.\n\n{context}{input}",
    "repobench-p": "Complete the code below using the repository context. Output only the code continuation.\n\n{context}{input}",
}


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{index + 1}")
            yield index, value


def _sharegpt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("conversations") or record.get("conversation") or []
    role_map = {
        "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant",
        "system": "system",
    }
    messages: list[dict[str, str]] = []
    for turn in raw:
        role = role_map.get(str(turn.get("from", turn.get("role", ""))).lower())
        content = turn.get("value", turn.get("content", ""))
        if role and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    return messages


def load_sharegpt(spec: dict[str, Any]) -> list[Sample]:
    path = Path(spec["path"]).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records.get("data", records.get("conversations", []))
    if not isinstance(records, list):
        raise ValueError(f"ShareGPT file must contain a JSON array: {path}")
    limit = int(spec.get("max_conversation_messages", 0))
    samples: list[Sample] = []
    for index, record in enumerate(records):
        messages = _sharegpt_messages(record)
        if limit > 0:
            messages = messages[:limit]
            while messages and messages[-1]["role"] != "user":
                messages.pop()
        if not messages:
            continue
        sample_id = str(record.get("id", record.get("conversation_id", index)))
        samples.append(Sample(sample_id, "sharegpt", "chat", messages, index))
    return samples


def load_gsm8k(spec: dict[str, Any]) -> list[Sample]:
    path = Path(spec["path"]).expanduser()
    samples = []
    for index, record in _read_jsonl(path):
        question = record.get("question")
        if isinstance(question, str) and question.strip():
            prompt = f"Solve the following math problem. Show your reasoning and give the final answer.\n\n{question}"
            samples.append(Sample(str(index), "gsm8k", path.stem, [{"role": "user", "content": prompt}], index))
    return samples


def load_humaneval(spec: dict[str, Any]) -> list[Sample]:
    path = Path(spec["path"]).expanduser()
    samples = []
    for index, record in _read_jsonl(path):
        prompt = record.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            task_id = str(record.get("task_id", index))
            content = "Complete the following Python function. Return only the code continuation.\n\n" + prompt
            samples.append(Sample(task_id, "humaneval", "code", [{"role": "user", "content": content}], index))
    return samples


def load_longbench(spec: dict[str, Any]) -> list[Sample]:
    root = Path(spec["path"]).expanduser()
    tasks = spec.get("tasks") or list(LONGBENCH_TASKS)
    invalid = sorted(set(tasks) - set(LONGBENCH_TASKS))
    if invalid:
        raise ValueError(f"Unknown LongBench tasks: {invalid}")
    samples = []
    for task in tasks:
        path = root / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"LongBench task file not found: {path}")
        template = LONGBENCH_PROMPTS[task]
        for index, record in _read_jsonl(path):
            prompt = template.format(
                input=str(record.get("input", "")),
                context=str(record.get("context", "")),
            )
            sample_id = str(record.get("id", f"{task}:{index}"))
            samples.append(Sample(sample_id, "longbench", task, [{"role": "user", "content": prompt}], index))
    return samples


LOADERS = {
    "sharegpt": load_sharegpt,
    "gsm8k": load_gsm8k,
    "longbench": load_longbench,
    "humaneval": load_humaneval,
}


def load_and_sample(config: dict[str, Any]) -> tuple[list[Sample], dict[str, int]]:
    seed = int(config["benchmark"].get("seed", 0))
    rng = random.Random(seed)
    pools: list[list[Sample]] = []
    counts: dict[str, int] = {}
    for spec in config["datasets"]:
        if not spec.get("enabled", True):
            continue
        pool = LOADERS[spec["name"]](spec)
        if not pool:
            raise ValueError(f"No valid samples loaded for dataset {spec['name']}")
        rng.shuffle(pool)
        pools.append(pool)
        counts[spec["name"]] = len(pool)

    target_per_dataset = int(config["benchmark"]["request_count"])
    strategy = config.get("sampling", {}).get("strategy", "round_robin")
    repeat = bool(config.get("sampling", {}).get("repeat", False))
    if strategy not in {"round_robin", "shuffled"}:
        raise ValueError("sampling.strategy must be round_robin or shuffled")

    selected_pools: list[list[Sample]] = []
    for pool in pools:
        dataset_name = pool[0].dataset
        if not repeat and len(pool) < target_per_dataset:
            raise ValueError(
                f"Dataset {dataset_name} requested {target_per_dataset} samples "
                f"but only {len(pool)} are available; set sampling.repeat=true to cycle"
            )

        selected: list[Sample] = []
        while len(selected) < target_per_dataset:
            cycle = list(pool)
            if selected:
                rng.shuffle(cycle)
            selected.extend(cycle[:target_per_dataset - len(selected)])
        selected_pools.append(selected)

    combined: list[Sample] = []
    if strategy == "round_robin":
        for index in range(target_per_dataset):
            combined.extend(pool[index] for pool in selected_pools)
    else:
        combined = [sample for pool in selected_pools for sample in pool]
        rng.shuffle(combined)
    return combined, counts


