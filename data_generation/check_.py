import os
import re
import json
import argparse
from typing import Any, Dict, List, Set
from tqdm import tqdm


OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


ANALYZE_PROMPT = """
You are given a task-oriented dialogue with segment annotations.

Your job is to analyze the dialogue at the segment level.

## Input
The input consists of:
- **Task**: the task description
- **Required Information**: the required information fields for completing the task, each with its gold answer
- **Conversation**: a dialogue where each turn contains `speaker`, `text`, and `segment_id`

## Your goals
You must determine:

1. **Final segment-to-required-info mapping**
   - For each required information field, identify the segment in which its final correct value (i.e., the gold answer, or an equivalent final expression) is established in the conversation.
   - A field should be mapped to the segment where its final usable value becomes available.
   - If an earlier segment mentions a tentative value, partial value, incorrect value, or a value that is later revised, that earlier segment must NOT be treated as the final segment for that field.

2. **Earliest resolvable segment**
   - Determine the earliest segment after which the task can be completed using the information available up to and including that segment.
   - This must be based on the final correct values, not on intermediate or later-corrected mentions.

## Important instructions
- Use the required information field names exactly as given.
- Use the gold answers as the reference for deciding when each field is truly established.
- A segment may correspond to multiple required information fields.
- Some segments may correspond to no required information fields.
- Be conservative and precise.

## Output format
Return a JSON object with exactly this format:

{
  "segment_required_info_mapping": {
    "1": ["..."],
    "2": ["..."]
  },
  "earliest_resolvable_segment": 3
}

## Output constraints
- Output only JSON.
- Do not include explanations.
- The keys of `segment_required_info_mapping` must be string versions of segment ids.
- `earliest_resolvable_segment` must be an integer.

## Task
{task}

## Required Information
{required_info_json}

## Conversation
{conversation_json}
""".strip()


REWRITE_PROMPT = """
You are given a task-oriented dialogue with segment annotations.

Your job is to rewrite the dialogue into a more natural and realistic conversational form while preserving the original task intent and overall stage structure.

## Input
The input consists of:
- **Task**
- **Required Information**
- **Missing Required Information**: the fields that must be newly established in the final segment
- **Conversation**: the original conversation with `speaker`, `text`, and `segment_id`

## Rewriting goals
1. Keep the conversation natural and human-like.
2. Important task-related information should mainly come from the user.
3. The missing required information must be established in the **final segment**.
4. The missing required information should be introduced naturally, preferably by the user.
5. Preserve the overall conversation flow and segment structure.
6. You may add or revise turns if necessary.

## Output format
Return a JSON object with exactly this format:

{
  "conversation": [
    {
      "speaker": "user",
      "text": "...",
      "segment_id": 1
    },
    {
      "speaker": "assistant",
      "text": "...",
      "segment_id": 1
    }
  ]
}

## Output constraints
- Only output the rewritten conversation.
- Each turn must contain exactly:
  - `speaker`
  - `text`
  - `segment_id`
- `speaker` must be either "user" or "assistant".
- `segment_id` must be a positive integer and must be non-decreasing.
- The missing required information must appear by the final segment.
- Do not output explanations or extra metadata.

## Task
{task}

## Required Information
{required_info_json}

## Missing Required Information
{missing_required_info_json}

## Original Conversation
{conversation_json}
""".strip()


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
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sort_key_by_id(item: Dict[str, Any]):
    item_id = str(item.get("id", ""))
    m = re.search(r"(\d+)$", item_id)
    if m:
        return (0, int(m.group(1)), item_id)
    return (1, item_id)


def write_jsonl_sorted(path: str, items: List[Dict[str, Any]]) -> None:
    sorted_items = sorted(items, key=sort_key_by_id)
    with open(path, "w", encoding="utf-8") as f:
        for item in sorted_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def build_required_info(required_info: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for x in required_info:
        name = str(x.get("name", "")).strip()
        gold_answer = str(x.get("gold_answer", "")).strip()
        if name:
            out.append({
                "name": name,
                "gold_answer": gold_answer
            })
    return out


def build_clean_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for turn in conversation:
        out.append({
            "speaker": str(turn.get("speaker", "")).strip().lower(),
            "text": normalize_text(turn.get("text", "")),
            "segment_id": int(turn.get("segment_id", 1)),
        })
    return out


def build_analyze_prompt(item: Dict[str, Any]) -> str:
    task = item.get("task", "")
    required_info = build_required_info(item.get("required_info", []) or [])
    conversation = build_clean_conversation(item.get("conversation", []) or [])

    prompt = ANALYZE_PROMPT
    prompt = prompt.replace("{task}", str(task))
    prompt = prompt.replace("{required_info_json}", json.dumps(required_info, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{conversation_json}", json.dumps(conversation, ensure_ascii=False, indent=2))
    return prompt


def build_rewrite_prompt(item: Dict[str, Any], missing_required_info: List[Dict[str, str]]) -> str:
    task = item.get("task", "")
    required_info = build_required_info(item.get("required_info", []) or [])
    conversation = build_clean_conversation(item.get("conversation", []) or [])

    prompt = REWRITE_PROMPT
    prompt = prompt.replace("{task}", str(task))
    prompt = prompt.replace("{required_info_json}", json.dumps(required_info, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{missing_required_info_json}", json.dumps(missing_required_info, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{conversation_json}", json.dumps(conversation, ensure_ascii=False, indent=2))
    return prompt


def get_segment_ids(conversation: List[Dict[str, Any]]) -> List[int]:
    segs = []
    for turn in conversation:
        seg = int(turn.get("segment_id", 1))
        if seg not in segs:
            segs.append(seg)
    return segs


def validate_output(item: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    required_info = build_required_info(item.get("required_info", []) or [])
    allowed_names = {x["name"] for x in required_info}
    conversation = item.get("conversation", []) or []
    segment_ids = get_segment_ids(conversation)
    last_segment = max(segment_ids) if segment_ids else 1

    mapping_raw = parsed.get("segment_required_info_mapping", {})
    earliest = parsed.get("earliest_resolvable_segment", None)

    if not isinstance(mapping_raw, dict):
        return {
            "ok": False,
            "reason": "invalid_mapping_format",
        }

    cleaned_mapping: Dict[str, List[str]] = {}
    covered_required_info = set()

    for seg in segment_ids:
        key = str(seg)
        vals = mapping_raw.get(key, [])
        if not isinstance(vals, list):
            vals = []

        clean_vals = []
        for v in vals:
            v = str(v).strip()
            if v in allowed_names and v not in clean_vals:
                clean_vals.append(v)

        cleaned_mapping[key] = clean_vals
        covered_required_info.update(clean_vals)

    missing_required_info = sorted(list(allowed_names - covered_required_info))
    if missing_required_info:
        return {
            "ok": False,
            "reason": "missing_required_info_coverage",
            "missing_required_info": missing_required_info,
            "earliest_resolvable_segment": earliest,
            "segment_required_info_mapping": cleaned_mapping,
        }

    if not isinstance(earliest, int):
        return {
            "ok": False,
            "reason": "invalid_earliest_resolvable_segment",
            "segment_required_info_mapping": cleaned_mapping,
        }

    if earliest != last_segment:
        return {
            "ok": False,
            "reason": "resolvable_before_last_segment",
            "earliest_resolvable_segment": earliest,
            "last_segment": last_segment,
            "segment_required_info_mapping": cleaned_mapping,
        }

    return {
        "ok": True,
        "earliest_resolvable_segment": earliest,
        "segment_required_info_mapping": cleaned_mapping,
    }


def merge_last_segment_to_previous(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    把最后一个 segment 并到前一个 segment：
    last -> last-1
    """
    new_item = dict(item)
    conversation = []
    for turn in item.get("conversation", []) or []:
        conversation.append(dict(turn))

    seg_ids = get_segment_ids(conversation)
    if len(seg_ids) < 2:
        new_item["conversation"] = conversation
        return new_item

    last_seg = max(seg_ids)
    prev_seg = sorted(seg_ids)[-2]

    for turn in conversation:
        if int(turn.get("segment_id", 1)) == last_seg:
            turn["segment_id"] = prev_seg

    new_item["conversation"] = conversation
    return new_item


def clean_rewritten_conversation(
    original_conversation: List[Dict[str, Any]],
    rewritten_conversation: Any
) -> List[Dict[str, Any]]:
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


def rewrite_with_missing_required_info(
    item: Dict[str, Any],
    missing_required_info_names: List[str],
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    required_info = build_required_info(item.get("required_info", []) or [])
    missing_required_info = [x for x in required_info if x["name"] in set(missing_required_info_names)]

    prompt = build_rewrite_prompt(item, missing_required_info)
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)

    new_item = dict(item)
    new_item["conversation"] = clean_rewritten_conversation(
        item.get("conversation", []) or [],
        parsed.get("conversation", [])
    )
    return new_item


def analyze_item(
    item: Dict[str, Any],
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    prompt = build_analyze_prompt(item)
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)
    return validate_output(item, parsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", default="multiwoz_top200_refined_checked.jsonl")
    parser.add_argument("--input", default="multiwoz_top200_refined.jsonl")
    parser.add_argument("--output", default="multiwoz_top200_refined_checked_2.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--analyze-max-tokens", type=int, default=1500)
    parser.add_argument("--rewrite-max-tokens", type=int, default=3000)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    original_data = load_jsonl(args.input)
    saved_data = load_jsonl(args.saved)
    saved_ids: Set[str] = {str(x.get("id", "")) for x in saved_data if x.get("id", "")}

    total_candidates = [x for x in original_data if str(x.get("id", "")) not in saved_ids]
    if args.max_items > 0:
        total_candidates = total_candidates[:args.max_items]

    client = OpenAICompatibleClient(
        base_url=OPENAI_BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_MODEL,
    )

    # 用 id 合并，最后统一按 id 排序写出
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for item in saved_data:
        item_id = str(item.get("id", ""))
        if item_id:
            merged_by_id[item_id] = item

    fixed_count = 0
    skipped_bad_count = 0

    for idx, item in enumerate(tqdm(total_candidates, desc="Repairing unsaved bad cases"), start=1):
        item_id = str(item.get("id", "unknown"))

        try:
            result = analyze_item(
                item=item,
                client=client,
                temperature=args.temperature,
                max_tokens=args.analyze_max_tokens
            )

            # 如果本来就 OK，放进 merged_by_id
            if result["ok"]:
                good_item = dict(item)
                good_item["segment_required_info_mapping"] = result["segment_required_info_mapping"]
                good_item["earliest_resolvable_segment"] = result["earliest_resolvable_segment"]
                merged_by_id[item_id] = good_item
                fixed_count += 1
                continue

            repaired_item = None

            if result.get("reason") == "resolvable_before_last_segment":
                repaired_item = merge_last_segment_to_previous(item)

            elif result.get("reason") == "missing_required_info_coverage":
                missing_names = result.get("missing_required_info", [])
                repaired_item = rewrite_with_missing_required_info(
                    item=item,
                    missing_required_info_names=missing_names,
                    client=client,
                    temperature=0.7,
                    max_tokens=args.rewrite_max_tokens
                )

            else:
                print("=" * 80)
                print(f"[UNHANDLED BAD CASE] id={item_id}")
                print(f"reason: {result.get('reason')}")
                print("=" * 80)
                skipped_bad_count += 1
                continue

            repaired_result = analyze_item(
                item=repaired_item,
                client=client,
                temperature=args.temperature,
                max_tokens=args.analyze_max_tokens
            )

            if repaired_result["ok"]:
                good_item = dict(repaired_item)
                good_item["segment_required_info_mapping"] = repaired_result["segment_required_info_mapping"]
                good_item["earliest_resolvable_segment"] = repaired_result["earliest_resolvable_segment"]
                merged_by_id[item_id] = good_item
                fixed_count += 1
            else:
                print("=" * 80)
                print(f"[STILL BAD AFTER REPAIR] id={item_id}")
                print(f"reason: {repaired_result.get('reason')}")
                if "missing_required_info" in repaired_result:
                    print(f"missing_required_info: {repaired_result['missing_required_info']}")
                if "earliest_resolvable_segment" in repaired_result:
                    print(f"earliest_resolvable_segment: {repaired_result['earliest_resolvable_segment']}")
                if "last_segment" in repaired_result:
                    print(f"last_segment: {repaired_result['last_segment']}")
                if "segment_required_info_mapping" in repaired_result:
                    print("segment_required_info_mapping:")
                    print(json.dumps(repaired_result["segment_required_info_mapping"], ensure_ascii=False, indent=2))
                print("=" * 80)
                skipped_bad_count += 1

        except Exception as e:
            print("=" * 80)
            print(f"[EXCEPTION] id={item_id}")
            print(str(e))
            print("=" * 80)
            skipped_bad_count += 1

    # 最后统一按 id 排序写出
    write_jsonl_sorted(args.output, list(merged_by_id.values()))

    print(f"saved merged file to: {args.output}")
    print(f"repaired_and_saved_count: {fixed_count}")
    print(f"still_bad_or_skipped_count: {skipped_bad_count}")


if __name__ == "__main__":
    main()
