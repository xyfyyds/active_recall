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


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"conversation": []}

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

    return {"conversation": []}


def build_prompt(item: Dict[str, Any], prompt_template: str) -> str:
    task = item.get("task", "")
    required_info = item.get("required_info", []) or []
    conversation = item.get("conversation", []) or []

    clean_conversation = []
    for turn in conversation:
        clean_conversation.append({
            "speaker": str(turn.get("speaker", "")).strip().lower(),
            "text": normalize_text(turn.get("text", "")),
            "segment_id": int(turn.get("segment_id", 1)),
        })

    prompt = prompt_template
    prompt = prompt.replace("{task}", str(task))
    prompt = prompt.replace(
        "{required_info_json}",
        json.dumps(required_info, ensure_ascii=False, indent=2)
    )
    prompt = prompt.replace(
        "{conversation_json}",
        json.dumps(clean_conversation, ensure_ascii=False, indent=2)
    )
    return prompt


def clean_rewritten_conversation(
    original_conversation: List[Dict[str, Any]],
    rewritten_conversation: Any
) -> List[Dict[str, Any]]:
    """
    最小清洗：
    - 只保留 speaker/text/segment_id
    - speaker 只能是 user/assistant
    - text 非空
    - segment_id 为正整数
    - segment_id 非递减
    - 如果模型输出不可用，则退回原始 conversation
    """
    if not isinstance(rewritten_conversation, list):
        return original_conversation

    cleaned = []
    last_seg = 1

    for turn in rewritten_conversation:
        if not isinstance(turn, dict):
            continue

        speaker = str(turn.get("speaker", "")).strip().lower()
        text = normalize_text(turn.get("text", ""))
        seg = turn.get("segment_id", last_seg)

        if speaker not in {"user", "assistant"}:
            continue
        if not text:
            continue

        try:
            seg = int(seg)
        except Exception:
            seg = last_seg

        if seg < 1:
            seg = 1
        if seg < last_seg:
            seg = last_seg

        cleaned.append({
            "speaker": speaker,
            "text": text,
            "segment_id": seg,
        })
        last_seg = seg

    if not cleaned:
        return original_conversation

    return cleaned


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


def process_item(
    item: Dict[str, Any],
    prompt_template: str,
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    prompt = build_prompt(item, prompt_template)
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)

    new_item = dict(item)
    new_item["conversation"] = clean_rewritten_conversation(
        item.get("conversation", []) or [],
        parsed.get("conversation", [])
    )
    return new_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="multiwoz_top200_segment.jsonl")
    parser.add_argument("--output", default="multiwoz_top200_refined.jsonl")
    parser.add_argument("--prompt-file", default="refine_sgd.txt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    prompt_template = load_prompt_template(args.prompt_file)

    total = len(data) if args.max_items <= 0 else min(len(data), args.max_items)

    client = OpenAICompatibleClient(
        base_url=OPENAI_BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_MODEL,
    )

    existing_ids = load_existing_ids(args.output)
    print(f"[continue] found {len(existing_ids)} existing ids in {args.output}")

    items_to_process = data[:total]
    if existing_ids:
        items_to_process = [
            item for item in items_to_process
            if str(item.get("id", "")) not in existing_ids
        ]

    print(f"will process {len(items_to_process)} items")

    for idx, item in enumerate(tqdm(items_to_process, desc="Rewriting whole conversations"), start=1):
        try:
            new_item = process_item(
                item=item,
                prompt_template=prompt_template,
                client=client,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
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
