#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MultiWOZ top-one-task data into SGD-style seed JSONL.

Input:
  ./multiwoz_top200_one_task.json

Output:
  ./multiwoz_top200_sgd_style.jsonl

Usage:
python convert_multiwoz_to_sgd_style.py \
  --input_file ./multiwoz_top200_one_task.json \
  --output_file ./multiwoz_top200_sgd_style.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(items: List[Dict[str, Any]], path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_speaker(speaker: str) -> str:
    speaker = str(speaker).upper()
    if speaker == "USER":
        return "User"
    if speaker in {"SYSTEM", "ASSISTANT", "CHATBOT"}:
        return "Assistant"
    return speaker.title()


def convert_dialogue_turns(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert current MultiWOZ dialogue format:

    {
      "turn_index": 0,
      "turn_id": "0",
      "speaker": "USER",
      "text": "..."
    }

    into SGD-style User/Assistant turns.
    """
    turns = item.get("dialogue", [])
    converted = []

    for i, t in enumerate(turns):
        turn_index = int(t.get("turn_index", i))
        turn_id = str(t.get("turn_id", turn_index))
        speaker = normalize_speaker(t.get("speaker", ""))
        text = t.get("text", "")

        converted.append({
            "turn_id": turn_id,
            "turn_index": turn_index,
            "speaker": speaker,
            "role": "user" if speaker == "User" else "assistant",
            "text": text,
        })

    return converted


def convert_required_info(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert MultiWOZ required_info into SGD-style required info.

    The important occurrence position is:
      first_state_turn_index / first_state_turn_id

    This marks the first turn where the slot-value appears in
    the same service's state.slot_values.
    """
    required_info = []

    for idx, r in enumerate(item.get("required_info", []), start=1):
        slot = r.get("slot")
        value = r.get("value")

        required_info.append({
            # Generic names for previous SGD-style pipeline
            "name": slot,
            "value": value,
            "rid": f"r{idx}",

            # Keep original MultiWOZ naming
            "slot": slot,
            "pair_key": r.get("pair_key", f"{slot}={value}"),

            # Where this required information first becomes available
            "turn_index": int(r.get("first_state_turn_index")),
            "turn_id": str(r.get("first_state_turn_id")),
            "speaker": normalize_speaker(r.get("first_state_speaker", "USER")),
            "evidence": r.get("first_state_utterance", ""),

            # Debug / trace fields
            "source_service": r.get("first_state_service"),
            "source_active_intent": r.get("first_state_active_intent"),
        })

    required_info.sort(key=lambda x: (x["turn_index"], x["name"], str(x["value"])))
    return required_info


def build_task_text(item: Dict[str, Any]) -> str:
    task = item.get("task", {})
    service = task.get("service", "")
    intent = task.get("active_intent", "")

    if intent and service:
        return f"{intent} for service {service}"
    return task.get("description", item.get("dialogue_id", ""))


def convert_one_item(item: Dict[str, Any]) -> Dict[str, Any]:
    dialogue_id = item.get("dialogue_id")
    task = item.get("task", {})
    selected_task = item.get("selected_task", {})

    dialogue = convert_dialogue_turns(item)
    required_info = convert_required_info(item)

    required_info_names = [r["name"] for r in required_info]
    required_info_values = {r["name"]: r["value"] for r in required_info}

    checkpoint_turn_indices = sorted({
        int(r["turn_index"]) for r in required_info
    })

    return {
        # Main SGD-style fields
        "id": f"multiwoz_{dialogue_id}",
        "task": build_task_text(item),
        "required_info": required_info,
        "dialogue": dialogue,

        # Convenient fields for your later pipeline
        "required_info_names": required_info_names,
        "required_info_values": required_info_values,
        "num_required_info": len(required_info),
        "checkpoint_turn_indices": checkpoint_turn_indices,

        # Extra structured task info
        "task_schema": {
            "task_key": task.get("task_key", selected_task.get("task_key")),
            "service": task.get("service", selected_task.get("service")),
            "active_intent": task.get("active_intent", selected_task.get("active_intent")),
            "description": task.get("description", selected_task.get("task_key")),
        },

        # Source trace
        "source_dataset": "multiwoz",
        "source_dialogue_id": dialogue_id,
        "source_file": item.get("source_file"),
        "services": item.get("services", []),
        "num_utterances": item.get("num_utterances"),
        "num_exchanges": item.get("num_exchanges"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="./multiwoz_top200_one_task.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./multiwoz_top200_sgd_style.jsonl",
    )
    parser.add_argument(
        "--min_required_info",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    data = load_json(args.input_file)

    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON list.")

    outputs = []
    dropped = 0

    for item in data:
        converted = convert_one_item(item)

        if converted["num_required_info"] < args.min_required_info:
            dropped += 1
            continue

        outputs.append(converted)

    write_jsonl(outputs, args.output_file)

    print(f"Loaded: {len(data)}")
    print(f"Saved: {len(outputs)}")
    print(f"Dropped: {dropped}")
    print(f"Output: {args.output_file}")

    print("\nPreview:")
    for x in outputs[:5]:
        print(
            f"- {x['id']} | "
            f"task={x['task']} | "
            f"required_info={x['num_required_info']} | "
            f"checkpoints={x['checkpoint_turn_indices']}"
        )


if __name__ == "__main__":
    main()
