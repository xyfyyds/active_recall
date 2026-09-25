import os
import re
import json
import argparse
from typing import Any, Dict, List
from tqdm import tqdm


OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 180):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_existing_ids(path: str) -> set:
    existing_ids = set()
    if not os.path.exists(path):
        return existing_ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    existing_ids.add(str(obj["id"]))
            except Exception:
                continue

    return existing_ids


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"segments": []}

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {"segments": []}


def build_prompt(item: Dict[str, Any]) -> str:
    task = item.get("task", "")
    required_info = item.get("required_info", []) or []
    conversation = item.get("conversation", []) or []

    required_info_names = [x.get("name", "") for x in required_info if x.get("name", "")]
    clean_conversation = [
        {
            "speaker": str(turn.get("speaker", "")).strip().lower(),
            "text": normalize_text(turn.get("text", ""))
        }
        for turn in conversation
    ]

    prompt = f"""
You are given:
1. a task
2. a list of required information fields
3. a conversation

Your job is to segment the conversation into a small number of coherent stages.

Each turn must be assigned to exactly one stage.
Stages must be contiguous:
- if a stage starts, it must continue for consecutive turns
- stage numbers must be 1, 2, 3, ... in order
- do not go back to an earlier stage

The segmentation goal is to reflect the progress of the task:
- task introduction
- partial information collection
- further information completion
- final stage where enough information is available or the task is resolved

Use the required_info only as reference.
Do NOT create one stage per required_info field.
Instead, group nearby turns into a small number of meaningful phases.

Return a JSON object with exactly this format:
{{
  "segments": [
    {{"turn_index": 0, "segment_id": 1}},
    {{"turn_index": 1, "segment_id": 1}},
    {{"turn_index": 2, "segment_id": 2}}
  ]
}}

Rules:
1. turn_index is the 0-based index in the conversation.
2. Each turn must have exactly one segment_id.
3. segment_id must be integers starting from 1.
4. segment_id must be non-decreasing across turns.
5. Prefer 3 to 6 stages unless the conversation is extremely short or extremely long.
6. Do not output explanations.
7. Do not output anything outside the JSON object.

Task:
{task}

Required info:
{json.dumps(required_info_names, ensure_ascii=False)}

Conversation:
{json.dumps(clean_conversation, ensure_ascii=False, indent=2)}
""".strip()

    return prompt


def clean_segments(parsed: Dict[str, Any], n_turns: int) -> List[int]:
    """
    输出一个长度为 n_turns 的 segment_id 列表。
    如果模型输出有问题，就尽量修正：
    - 缺失 turn 用上一段补
    - 非法编号修正为不回退的连续正整数
    """
    raw_segments = parsed.get("segments", [])
    turn_to_seg: Dict[int, int] = {}

    if isinstance(raw_segments, list):
        for x in raw_segments:
            if not isinstance(x, dict):
                continue
            ti = x.get("turn_index", None)
            sid = x.get("segment_id", None)
            if isinstance(ti, int) and isinstance(sid, int) and 0 <= ti < n_turns and sid >= 1:
                turn_to_seg[ti] = sid

    segs: List[int] = []
    last_seg = 1
    mapping: Dict[int, int] = {}
    next_new = 1

    for i in range(n_turns):
        sid = turn_to_seg.get(i, last_seg)

        # 保证非递减
        if sid < last_seg:
            sid = last_seg

        # 重新压缩成 1,2,3,...
        if sid not in mapping:
            mapping[sid] = next_new
            next_new += 1

        sid = mapping[sid]
        if sid < last_seg:
            sid = last_seg

        segs.append(sid)
        last_seg = sid

    return segs


def apply_segments(item: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    conversation = item.get("conversation", []) or []
    segs = clean_segments(parsed, len(conversation))

    new_conversation = []
    for idx, turn in enumerate(conversation):
        new_turn = dict(turn)
        new_turn["segment_id"] = segs[idx]
        new_conversation.append(new_turn)

    new_item = dict(item)
    new_item["conversation"] = new_conversation
    return new_item


def process_item(
    item: Dict[str, Any],
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    prompt = build_prompt(item)
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)
    return apply_segments(item, parsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="multiwoz_top200_sgd_style.jsonl")
    parser.add_argument("--output", default="multiwoz_top200_segment.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data = load_jsonl(args.input)
    total = len(data) if args.max_items <= 0 else min(len(data), args.max_items)
    items_to_process = data[:total]

    if args.resume:
        existing_ids = load_existing_ids(args.output)
        print(f"[continue] found {len(existing_ids)} existing ids in {args.output}")
        items_to_process = [
            item for item in items_to_process
            if str(item.get("id", "")) not in existing_ids
        ]
    else:
        open(args.output, "w", encoding="utf-8").close()

    client = OpenAICompatibleClient(
        base_url=OPENAI_BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_MODEL,
    )

    print(f"will process {len(items_to_process)} items")

    for idx, item in enumerate(tqdm(items_to_process, desc="LLM stage segmentation"), start=1):
        try:
            new_item = process_item(item, client, args.temperature, args.max_tokens)
            append_jsonl(args.output, new_item)
            print(f"[{idx}/{len(items_to_process)}] done: {item.get('id', 'unknown')}")
        except Exception as e:
            print(f"[{idx}/{len(items_to_process)}] failed: {item.get('id', 'unknown')} | {e}")
            bad = dict(item)
            bad["error"] = str(e)
            append_jsonl(args.output, bad)

    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
