import os
import re
import json
import argparse
from typing import Any, Dict, List
from tqdm import tqdm


OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


ANNOTATION_PROMPT = """
You are given:
1. a task description,
2. one local dialogue segment,
3. the required information items that are known to be established in this local segment.

Your job is to annotate evidence at the local-segment level.

## Objective

For each required information item assigned to this local segment, identify the minimal turn or turns within the local segment that establish its final usable value.

The goal is to locate explicit conversational evidence for each required information item within this local segment only.

## Definitions

- **Local segment**: the current segment only, not the full dialogue.
- **Local turn id**: the 0-based index of a turn within this local segment.
- **Evidence context**: the exact turn text or minimal set of turn texts that provide the evidence for the corresponding required information item.

## Annotation Principles

### 1. Local-only annotation
You must work strictly within the provided local segment.
Do not assume access to earlier or later segments.
Do not infer evidence from dialogue context outside the current local segment.

### 2. Restricted required-information scope
Only annotate the required information items explicitly provided in the input for this local segment.
Do not introduce additional required information items.

### 3. Minimal evidence principle
For each required information item, select the minimal turn or turns necessary to support the annotation.
Avoid selecting redundant turns when a smaller evidence set is sufficient.

### 4. Final usable value
The selected evidence should correspond to the final usable value established in this local segment.
If a turn contains a tentative, partial, or superseded mention, and a later turn in the same local segment provides the final usable value, prefer the later evidence.

### 5. Conservative annotation
Only annotate a required information item if there is clear textual support in the local segment.
Do not annotate based on weak implication alone.

## Output Format

Return a JSON object with exactly the following structure:

{
  "annotations": [
    {
      "required_info": "...",
      "local_turn_ids": [0],
      "evidence_context": ["..."]
    }
  ]
}

## Output Constraints

- Output JSON only.
- `required_info` must exactly match one of the required information items provided in the input.
- `local_turn_ids` must be a list of 0-based integers within the local segment.
- `evidence_context` must be a list of strings corresponding to the selected evidence turns.
- Include only required information items that are actually evidenced in the local segment.
- Do not output explanations, comments, or extra metadata.

## Task
{task}

## Required Information Assigned to This Local Segment
{segment_required_info_json}

## Local Segment
{local_segment_json}
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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"annotations": []}

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

    return {"annotations": []}


def build_clean_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for turn in conversation:
        out.append({
            "speaker": str(turn.get("speaker", "")).strip().lower(),
            "text": normalize_text(turn.get("text", "")),
            "segment_id": int(turn.get("segment_id", 1)),
        })
    return out


def get_original_segment_ids_from_mapping(mapping: Dict[str, Any]) -> List[int]:
    seg_ids = []
    if not isinstance(mapping, dict):
        return seg_ids

    for k in mapping.keys():
        try:
            seg_ids.append(int(k))
        except Exception:
            continue
    return sorted(seg_ids)


def get_segment_turns(conversation: List[Dict[str, Any]], segment_id: int) -> List[Dict[str, Any]]:
    seg_turns = []
    for turn in conversation:
        try:
            sid = int(turn.get("segment_id", -1))
        except Exception:
            sid = -1
        if sid == segment_id:
            seg_turns.append({
                "speaker": str(turn.get("speaker", "")).strip().lower(),
                "text": normalize_text(turn.get("text", "")),
            })
    return seg_turns


def build_local_segment_json(seg_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "local_turn_id": i,
            "speaker": t["speaker"],
            "text": t["text"]
        }
        for i, t in enumerate(seg_turns)
    ]


def build_prompt(
    item: Dict[str, Any],
    seg_turns: List[Dict[str, Any]],
    assigned_required_info: List[str]
) -> str:
    task = item.get("task", "")
    local_segment_json = build_local_segment_json(seg_turns)

    prompt = ANNOTATION_PROMPT
    prompt = prompt.replace("{task}", str(task))
    prompt = prompt.replace(
        "{segment_required_info_json}",
        json.dumps(assigned_required_info, ensure_ascii=False, indent=2)
    )
    prompt = prompt.replace(
        "{local_segment_json}",
        json.dumps(local_segment_json, ensure_ascii=False, indent=2)
    )
    return prompt


def clean_annotation_output(
    parsed: Dict[str, Any],
    seg_turns: List[Dict[str, Any]],
    allowed_required_info: List[str]
) -> List[Dict[str, Any]]:
    raw = parsed.get("annotations", [])
    if not isinstance(raw, list):
        return []

    allowed_set = set(allowed_required_info)
    cleaned = []

    for ann in raw:
        if not isinstance(ann, dict):
            continue

        req = str(ann.get("required_info", "")).strip()
        if req not in allowed_set:
            continue

        local_turn_ids = ann.get("local_turn_ids", [])
        if not isinstance(local_turn_ids, list):
            continue

        good_ids = []
        for x in local_turn_ids:
            if isinstance(x, int) and 0 <= x < len(seg_turns):
                if x not in good_ids:
                    good_ids.append(x)

        if not good_ids:
            continue

        normalized_evidence_context = [seg_turns[i]["text"] for i in good_ids]

        cleaned.append({
            "required_info": req,
            "local_turn_ids": good_ids,
            "evidence_context": normalized_evidence_context
        })

    return cleaned


def annotate_item(
    item: Dict[str, Any],
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    mapping = item.get("segment_required_info_mapping", {})
    conversation = build_clean_conversation(item.get("conversation", []) or [])

    new_item = dict(item)
    evidence_by_segment: Dict[str, List[Dict[str, Any]]] = {}

    original_segment_ids = get_original_segment_ids_from_mapping(mapping)

    for seg_id in original_segment_ids:
        assigned_required_info = mapping.get(str(seg_id), [])
        if not isinstance(assigned_required_info, list) or not assigned_required_info:
            continue

        seg_turns = get_segment_turns(conversation, seg_id)
        if not seg_turns:
            continue

        prompt = build_prompt(
            item=item,
            seg_turns=seg_turns,
            assigned_required_info=assigned_required_info
        )

        raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        parsed = safe_parse_json(raw)

        cleaned = clean_annotation_output(
            parsed=parsed,
            seg_turns=seg_turns,
            allowed_required_info=assigned_required_info
        )

        evidence_by_segment[str(seg_id)] = cleaned

    new_item["segment_required_info_evidence"] = evidence_by_segment
    return new_item


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="multiwoz_top200_enriched_via_topic.jsonl")
    parser.add_argument("--output", default="multiwoz_top200_enriched_via_topic_evidence.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if args.max_items > 0:
        data = data[:args.max_items]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in environment variables.")

    client = OpenAICompatibleClient(
        base_url=OPENAI_BASE_URL,
        api_key=api_key,
        model=OPENAI_MODEL,
    )

    existing_ids = load_existing_ids(args.output)
    print(f"[continue] found {len(existing_ids)} existing ids in {args.output}")

    items_to_process = data
    if existing_ids:
        items_to_process = [
            item for item in data
            if str(item.get("id", "")) not in existing_ids
        ]

    if args.max_items > 0:
        items_to_process = items_to_process[:args.max_items]

    for idx, item in enumerate(tqdm(items_to_process, desc="Annotating local required-info evidence"), start=1):
        try:
            new_item = annotate_item(
                item=item,
                client=client,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            append_jsonl(args.output, new_item)
            print(f"[{idx}/{len(data)}] done: {item.get('id', 'unknown')}")
        except Exception as e:
            bad = dict(item)
            bad["segment_required_info_evidence"] = {}
            bad["evidence_annotation_error"] = str(e)
            append_jsonl(args.output, bad)
            print(f"[{idx}/{len(data)}] failed: {item.get('id', 'unknown')} | {e}")


if __name__ == "__main__":
    main()
