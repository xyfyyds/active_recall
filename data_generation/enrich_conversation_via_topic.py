import os
import re
import json
import math
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


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError(f"Unsupported JSON format in {path}: expected a list")


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
        return {}

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

    return {}


def clean_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "speaker": str(turn.get("speaker", "")).strip().lower(),
        "text": normalize_text(turn.get("text", "")),
        "segment_id": int(turn.get("segment_id", 1))
    }


def group_by_segment(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not conversation:
        return []

    cleaned = [clean_turn(t) for t in conversation]
    groups = []
    current_seg = cleaned[0]["segment_id"]
    current_turns = []

    for turn in cleaned:
        seg = turn["segment_id"]
        if seg != current_seg:
            groups.append({
                "segment_id": current_seg,
                "turns": current_turns
            })
            current_seg = seg
            current_turns = [turn]
        else:
            current_turns.append(turn)

    if current_turns:
        groups.append({
            "segment_id": current_seg,
            "turns": current_turns
        })

    return groups


def average_segment_turns(segment_groups: List[Dict[str, Any]]) -> float:
    if not segment_groups:
        return 0.0
    return sum(len(g["turns"]) for g in segment_groups) / len(segment_groups)


def build_topic_prompt(
    prompt_template: str,
    task: str,
    prev_segment: List[Dict[str, Any]],
    next_segment: List[Dict[str, Any]],
) -> str:
    clean_prev = [
        {"speaker": str(t.get("speaker", "")).strip().lower(), "text": normalize_text(t.get("text", ""))}
        for t in prev_segment
    ]
    clean_next = [
        {"speaker": str(t.get("speaker", "")).strip().lower(), "text": normalize_text(t.get("text", ""))}
        for t in next_segment
    ]

    prompt = prompt_template
    prompt = prompt.replace("{task}", str(task))
    prompt = prompt.replace("{prev_segment_json}", json.dumps(clean_prev, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{next_segment_json}", json.dumps(clean_next, ensure_ascii=False, indent=2))
    return prompt


def build_filler_prompt(
    prompt_template: str,
    task: str,
    bridge_topic: str,
    bridge_topic_desc: str,
    prev_segment: List[Dict[str, Any]],
    next_segment: List[Dict[str, Any]],
    target_turns: int,
) -> str:
    clean_prev = [
        {"speaker": str(t.get("speaker", "")).strip().lower(), "text": normalize_text(t.get("text", ""))}
        for t in prev_segment
    ]
    clean_next = [
        {"speaker": str(t.get("speaker", "")).strip().lower(), "text": normalize_text(t.get("text", ""))}
        for t in next_segment
    ]

    prompt = prompt_template
    prompt = prompt.replace("{task}", str(task))
    prompt = prompt.replace("{bridge_topic}", str(bridge_topic))
    prompt = prompt.replace("{bridge_topic_desc}", str(bridge_topic_desc))
    prompt = prompt.replace("{prev_segment_json}", json.dumps(clean_prev, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{next_segment_json}", json.dumps(clean_next, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{target_turns}", str(target_turns))
    return prompt


def clean_generated_filler(conversation: Any) -> List[Dict[str, str]]:
    if not isinstance(conversation, list):
        return []

    cleaned = []
    expected_speaker = "user"

    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip().lower()
        text = normalize_text(turn.get("text", ""))

        if speaker not in {"user", "assistant"}:
            continue
        if not text:
            continue

        # 严格交替，不符合就跳过
        if speaker != expected_speaker:
            continue

        cleaned.append({
            "speaker": speaker,
            "text": text
        })
        expected_speaker = "assistant" if expected_speaker == "user" else "user"

    return cleaned


def safe_get_topic_fields(parsed: Dict[str, Any]) -> Dict[str, str]:
    topic = normalize_text(parsed.get("bridge_topic", ""))
    desc = normalize_text(parsed.get("short_description", ""))

    if not topic:
        topic = "a brief side discussion related to the situation but not directly advancing the task"
    if not desc:
        desc = "A short, natural digression that can connect the adjacent segments without directly pushing the main task forward."

    return {
        "bridge_topic": topic,
        "short_description": desc
    }


def generate_bridge_topic(
    client: OpenAICompatibleClient,
    topic_prompt_template: str,
    task: str,
    prev_segment: List[Dict[str, Any]],
    next_segment: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, str]:
    prompt = build_topic_prompt(
        prompt_template=topic_prompt_template,
        task=task,
        prev_segment=prev_segment,
        next_segment=next_segment,
    )
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)
    return safe_get_topic_fields(parsed)


def generate_filler_segment_two_stage(
    client: OpenAICompatibleClient,
    filler_prompt_template: str,
    task: str,
    bridge_topic: str,
    bridge_topic_desc: str,
    prev_segment: List[Dict[str, Any]],
    next_segment: List[Dict[str, Any]],
    target_turns: int,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, str]]:
    prompt = build_filler_prompt(
        prompt_template=filler_prompt_template,
        task=task,
        bridge_topic=bridge_topic,
        bridge_topic_desc=bridge_topic_desc,
        prev_segment=prev_segment,
        next_segment=next_segment,
        target_turns=target_turns,
    )
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)
    return clean_generated_filler(parsed.get("conversation", []))


def renumber_with_fillers(
    original_segment_groups: List[Dict[str, Any]],
    fillers: List[List[Dict[str, str]]]
) -> List[Dict[str, Any]]:
    """
    保留原始 segment_id 不变。
    新插入的 filler 用新的、不冲突的 segment_id。
    """
    new_conversation = []
    next_new_seg_id = 1000

    for i, group in enumerate(original_segment_groups):
        for turn in group["turns"]:
            new_conversation.append({
                "speaker": turn["speaker"],
                "text": turn["text"],
                "segment_id": group["segment_id"]
            })

        if i < len(fillers):
            filler_turns = fillers[i]
            filler_seg_id = next_new_seg_id
            next_new_seg_id += 1

            for turn in filler_turns:
                new_conversation.append({
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "segment_id": filler_seg_id
                })

    return new_conversation


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
    topic_prompt_template: str,
    filler_prompt_template: str,
    client: OpenAICompatibleClient,
    temperature: float,
    topic_max_tokens: int,
    filler_max_tokens: int,
    min_turns: int,
    multiplier: float,
) -> Dict[str, Any]:
    task = item.get("task", "")
    conversation = item.get("conversation", []) or []

    segment_groups = group_by_segment(conversation)
    if len(segment_groups) <= 1:
        return item

    avg_turns = average_segment_turns(segment_groups)
    target_turns = max(min_turns, math.ceil(avg_turns * multiplier))

    fillers = []
    bridge_topics = []

    for i in range(len(segment_groups) - 1):
        prev_seg = segment_groups[i]["turns"]
        next_seg = segment_groups[i + 1]["turns"]

        topic_info = generate_bridge_topic(
            client=client,
            topic_prompt_template=topic_prompt_template,
            task=task,
            prev_segment=prev_seg,
            next_segment=next_seg,
            temperature=temperature,
            max_tokens=topic_max_tokens,
        )

        filler_turns = generate_filler_segment_two_stage(
            client=client,
            filler_prompt_template=filler_prompt_template,
            task=task,
            bridge_topic=topic_info["bridge_topic"],
            bridge_topic_desc=topic_info["short_description"],
            prev_segment=prev_seg,
            next_segment=next_seg,
            target_turns=target_turns,
            temperature=temperature,
            max_tokens=filler_max_tokens,
        )

        fillers.append(filler_turns)
        bridge_topics.append({
            "insert_after_segment_id": segment_groups[i]["segment_id"],
            "insert_before_segment_id": segment_groups[i + 1]["segment_id"],
            "bridge_topic": topic_info["bridge_topic"],
            "short_description": topic_info["short_description"]
        })

    new_item = dict(item)
    new_item["conversation"] = renumber_with_fillers(segment_groups, fillers)
    new_item["bridge_topics"] = bridge_topics
    return new_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="multiwoz_top200_refined_checked.jsonl")
    parser.add_argument("--output", default="multiwoz_top200_enriched_via_topic.jsonl")

    parser.add_argument("--topic-prompt-file", default="topic_prompt.txt")
    parser.add_argument("--filler-prompt-file", default="enrich_via_topic.txt")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topic-max-tokens", type=int, default=800)
    parser.add_argument("--filler-max-tokens", type=int, default=5000)
    parser.add_argument("--max-items", type=int, default=0)

    parser.add_argument("--min-turns", type=int, default=20)
    parser.add_argument("--multiplier", type=float, default=5.0)

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    data = load_json_or_jsonl(args.input)
    topic_prompt_template = load_prompt_template(args.topic_prompt_file)
    filler_prompt_template = load_prompt_template(args.filler_prompt_file)

    total = min(len(data), args.max_items) if args.max_items > 0 else len(data)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Please set OPENAI_API_KEY in environment variables.")

    client = OpenAICompatibleClient(
        base_url=OPENAI_BASE_URL,
        api_key=api_key,
        model=OPENAI_MODEL,
    )

    existing_ids = set()

    if args.overwrite:
        open(args.output, "w", encoding="utf-8").close()
        print(f"[overwrite] cleared {args.output}")
    else:
        existing_ids = load_existing_ids(args.output)
        print(f"[continue] found {len(existing_ids)} existing ids in {args.output}")

    items_to_process = data[:total]
    if existing_ids:
        items_to_process = [
            item for item in items_to_process
            if str(item.get("id", "")) not in existing_ids
        ]

    print(f"will process {len(items_to_process)} items")

    for idx, item in enumerate(tqdm(items_to_process, desc="Generating off-task bridge fillers"), start=1):
        try:
            new_item = process_item(
                item=item,
                topic_prompt_template=topic_prompt_template,
                filler_prompt_template=filler_prompt_template,
                client=client,
                temperature=args.temperature,
                topic_max_tokens=args.topic_max_tokens,
                filler_max_tokens=args.filler_max_tokens,
                min_turns=args.min_turns,
                multiplier=args.multiplier,
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
