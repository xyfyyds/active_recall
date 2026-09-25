#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Top-N longest MultiWOZ dialogues with ONE selected task per dialogue.

Correct logic:
1. Candidate task = service::active_intent, e.g. hotel::book_hotel.
2. For each candidate task, get its required slot-values from the task state.
3. For each required slot-value, find the FIRST turn where this slot-value appears
   in the same service's state.slot_values, regardless of active_intent.
4. Do NOT match utterance text.
5. If any required slot-value cannot be found in state history, drop this dialogue.
6. For each dialogue, keep the task with the most required information.
7. Sort by dialogue length and save Top-N.

Example:
python build_multiwoz_top200_state_grounded.py \
  --input_dir ./MultiWOZ_2.2/train \
  --top_n 200 \
  --output_file ./multiwoz_top200_state_grounded.json \
  --value_mode final \
  --sort_by num_utterances
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    obj = load_json(path)
    return obj if isinstance(obj, list) else []


def collect_excluded_dialogue_ids(exclude_file: str) -> set:
    if not exclude_file:
        return set()

    rows = load_json_or_jsonl(Path(exclude_file))
    excluded = set()

    for row in rows:
        if not isinstance(row, dict):
            continue

        candidates = [
            row.get("dialogue_id"),
            row.get("source_dialogue_id"),
            row.get("id"),
        ]

        for value in candidates:
            value = str(value or "").strip()
            if not value:
                continue
            excluded.add(value)
            if value.startswith("multiwoz_"):
                excluded.add(value[len("multiwoz_"):])
            else:
                excluded.add(f"multiwoz_{value}")

    return excluded


def norm(x: Any) -> str:
    return str(x).strip().lower()


def get_turns(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = dialogue.get("turns", [])
    return turns if isinstance(turns, list) else []


def make_task_key(service: str, active_intent: str) -> str:
    return f"{service}::{active_intent}"


def values_to_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [norm(v) for v in values if norm(v)]
    return [norm(values)] if norm(values) else []


def extract_service_state_history(dialogue: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Build state history:
    service_state_history[service][pair_key] = list of occurrences

    Occurrence means this slot-value appears in state.slot_values at this turn.
    We use USER turns because MultiWOZ dialogue states are annotated on USER turns.
    """
    service_state_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    turns = get_turns(dialogue)

    for turn_index, turn in enumerate(turns):
        if str(turn.get("speaker", "")).upper() != "USER":
            continue

        turn_id = str(turn.get("turn_id", turn_index))
        utterance = turn.get("utterance", "")

        for frame in turn.get("frames", []):
            service = frame.get("service")
            state = frame.get("state", {})

            if not service or not isinstance(state, dict):
                continue

            slot_values = state.get("slot_values", {})
            if not isinstance(slot_values, dict):
                continue

            active_intent = state.get("active_intent", "NONE")

            service_state_history.setdefault(service, {})

            for slot, values in slot_values.items():
                for value in values_to_list(values):
                    pair_key = f"{slot}={value}"

                    service_state_history[service].setdefault(pair_key, [])
                    service_state_history[service][pair_key].append({
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "speaker": "USER",
                        "utterance": utterance,
                        "service": service,
                        "active_intent": active_intent,
                        "slot": slot,
                        "value": value,
                        "pair_key": pair_key,
                    })

    return service_state_history


def extract_candidate_task_states(dialogue: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract candidate tasks from USER turns.

    Candidate task = service::active_intent where active_intent != NONE.

    For each task, we collect state snapshots where that active_intent appears.
    """
    tasks: Dict[str, Dict[str, Any]] = {}
    turns = get_turns(dialogue)

    for turn_index, turn in enumerate(turns):
        if str(turn.get("speaker", "")).upper() != "USER":
            continue

        turn_id = str(turn.get("turn_id", turn_index))
        utterance = turn.get("utterance", "")

        for frame in turn.get("frames", []):
            service = frame.get("service")
            state = frame.get("state", {})

            if not service or not isinstance(state, dict):
                continue

            active_intent = state.get("active_intent", "NONE")
            if not active_intent or active_intent == "NONE":
                continue

            slot_values = state.get("slot_values", {})
            if not isinstance(slot_values, dict) or not slot_values:
                continue

            task_key = make_task_key(service, active_intent)

            if task_key not in tasks:
                tasks[task_key] = {
                    "task_key": task_key,
                    "service": service,
                    "active_intent": active_intent,
                    "state_snapshots": [],
                }

            snapshot_pairs = []

            for slot, values in slot_values.items():
                for value in values_to_list(values):
                    snapshot_pairs.append({
                        "slot": slot,
                        "value": value,
                        "pair_key": f"{slot}={value}",
                    })

            tasks[task_key]["state_snapshots"].append({
                "turn_index": turn_index,
                "turn_id": turn_id,
                "utterance": utterance,
                "slot_values": snapshot_pairs,
            })

    return tasks


def get_task_required_pairs(task: Dict[str, Any], value_mode: str) -> List[Dict[str, str]]:
    """
    Decide which slot-values belong to this task.

    value_mode:
    - final:
        For each slot, keep the value(s) from the latest task-state snapshot
        where this slot appears.
        This is recommended, because MultiWOZ users may revise constraints.
    - all:
        Keep all slot-value pairs that ever appear under this task.
    - first:
        For each slot, keep the first value(s) under this task.
    """
    snapshots = task.get("state_snapshots", [])
    if not snapshots:
        return []

    if value_mode == "all":
        seen = {}
        for snap in snapshots:
            for p in snap["slot_values"]:
                seen[p["pair_key"]] = p
        return list(seen.values())

    slot_to_occurrences: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}

    for snap in snapshots:
        turn_index = snap["turn_index"]
        for p in snap["slot_values"]:
            slot_to_occurrences.setdefault(p["slot"], [])
            slot_to_occurrences[p["slot"]].append((turn_index, p))

    selected = []

    for slot, occs in slot_to_occurrences.items():
        occs = sorted(occs, key=lambda x: x[0])

        if value_mode == "first":
            target_turn = occs[0][0]
        elif value_mode == "final":
            target_turn = occs[-1][0]
        else:
            raise ValueError(f"Unknown value_mode: {value_mode}")

        # If the same slot has multiple values at that chosen state turn, keep all.
        chosen_pairs = [p for t, p in occs if t == target_turn]

        # Deduplicate.
        seen = set()
        for p in chosen_pairs:
            if p["pair_key"] not in seen:
                seen.add(p["pair_key"])
                selected.append(p)

    selected.sort(key=lambda x: (x["slot"], x["value"]))
    return selected


def annotate_required_info_by_first_state_occurrence(
    task: Dict[str, Any],
    service_state_history: Dict[str, Dict[str, List[Dict[str, Any]]]],
    value_mode: str,
) -> Optional[Dict[str, Any]]:
    """
    For each required slot-value of this task, find its first occurrence
    in the same service's state history.

    Only keep the first occurrence. Do not keep all occurrences.
    """
    service = task["service"]
    required_pairs = get_task_required_pairs(task, value_mode=value_mode)

    if not required_pairs:
        return None

    service_history = service_state_history.get(service, {})
    required_info = []

    for p in required_pairs:
        pair_key = p["pair_key"]
        occurrences = service_history.get(pair_key, [])

        if not occurrences:
            # The required pair comes from state snapshots, so normally this should not happen.
            # If it happens, the sample is inconsistent and should be dropped.
            return None

        first_occ = sorted(occurrences, key=lambda x: x["turn_index"])[0]

        required_info.append({
            "slot": p["slot"],
            "value": p["value"],
            "pair_key": pair_key,

            # The first time this slot-value appears in the same service's state.slot_values.
            "first_state_turn_index": first_occ["turn_index"],
            "first_state_turn_id": first_occ["turn_id"],
            "first_state_speaker": first_occ["speaker"],
            "first_state_utterance": first_occ["utterance"],
            "first_state_active_intent": first_occ["active_intent"],
            "first_state_service": first_occ["service"],
        })

    required_info.sort(
        key=lambda x: (
            x["first_state_turn_index"],
            x["slot"],
            x["value"],
        )
    )

    required_slots = sorted({x["slot"] for x in required_info})
    checkpoint_turn_indices = sorted({x["first_state_turn_index"] for x in required_info})

    return {
        "task_key": task["task_key"],
        "service": task["service"],
        "active_intent": task["active_intent"],
        "num_required_slots": len(required_slots),
        "num_required_info": len(required_info),
        "required_slots": required_slots,
        "required_info": required_info,
        "checkpoint_turn_indices": checkpoint_turn_indices,
        "task_state_turns": [
            {
                "turn_index": s["turn_index"],
                "turn_id": s["turn_id"],
                "utterance": s["utterance"],
            }
            for s in task.get("state_snapshots", [])
        ],
    }

def select_best_task_for_dialogue(
    dialogue: Dict[str, Any],
    value_mode: str,
) -> Optional[Dict[str, Any]]:
    service_state_history = extract_service_state_history(dialogue)
    candidate_tasks = extract_candidate_task_states(dialogue)

    annotated_tasks = []

    for task in candidate_tasks.values():
        annotated = annotate_required_info_by_first_state_occurrence(
            task=task,
            service_state_history=service_state_history,
            value_mode=value_mode,
        )
        if annotated is not None:
            annotated_tasks.append(annotated)

    if not annotated_tasks:
        return None

    annotated_tasks.sort(
        key=lambda x: (
            x["num_required_info"],
            x["num_required_slots"],
            len(x["checkpoint_turn_indices"]),
        ),
        reverse=True,
    )

    best = annotated_tasks[0]
    best["num_candidate_tasks"] = len(candidate_tasks)
    return best


def convert_dialogue(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for idx, t in enumerate(turns):
        out.append({
            "turn_index": idx,
            "turn_id": str(t.get("turn_id", idx)),
            "speaker": str(t.get("speaker", "")).upper(),
            "text": t.get("utterance", ""),
        })
    return out


def collect_items(input_dir: str, value_mode: str, min_required_info: int) -> List[Dict[str, Any]]:
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    items = []
    dropped = 0

    for file_path in json_files:
        data = load_json(file_path)

        if not isinstance(data, list):
            print(f"[WARN] skip {file_path.name}: top-level object is not list")
            continue

        for idx, dialogue in enumerate(data):
            if not isinstance(dialogue, dict):
                dropped += 1
                continue

            turns = get_turns(dialogue)
            if not turns:
                dropped += 1
                continue

            selected_task = select_best_task_for_dialogue(
                dialogue=dialogue,
                value_mode=value_mode,
            )

            if selected_task is None:
                dropped += 1
                continue

            if selected_task["num_required_info"] < min_required_info:
                dropped += 1
                continue

            dialogue_id = dialogue.get("dialogue_id", f"{file_path.stem}_{idx}")

            item = {
                "dialogue_id": dialogue_id,
                "source_dataset": "multiwoz",
                "source_file": file_path.name,
                "services": dialogue.get("services", []),
                "num_utterances": len(turns),
                "num_exchanges": (len(turns) + 1) // 2,

                "num_candidate_tasks": selected_task.pop("num_candidate_tasks"),
                "selected_task": selected_task,

                # convenient duplicated fields for later pipeline
                "task": {
                    "task_key": selected_task["task_key"],
                    "service": selected_task["service"],
                    "active_intent": selected_task["active_intent"],
                    "description": f"{selected_task['active_intent']} for service {selected_task['service']}",
                },
                "required_info": selected_task["required_info"],
                "num_required_info": selected_task["num_required_info"],
                "checkpoint_turn_indices": selected_task["checkpoint_turn_indices"],

                "dialogue": convert_dialogue(turns),
                "raw_dialogue": dialogue,
            }

            items.append(item)

    print(f"Dropped dialogues: {dropped}")
    return items


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="./MultiWOZ_2_2",)
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="./multiwoz_top200_one_task.json")
    parser.add_argument("--exclude_file", type=str, default="")

    parser.add_argument(
        "--value_mode",
        type=str,
        default="final",
        choices=["first", "final", "all"],
        help="How to decide required slot-values for each task."
    )

    parser.add_argument(
        "--min_required_info",
        type=int,
        default=2,
        help="Drop dialogues whose selected task has fewer required info."
    )

    parser.add_argument(
        "--sort_by",
        type=str,
        default="num_utterances",
        choices=["num_utterances", "num_required_info"],
    )

    args = parser.parse_args()

    items = collect_items(
        input_dir=args.input_dir,
        value_mode=args.value_mode,
        min_required_info=args.min_required_info,
    )

    excluded_ids = collect_excluded_dialogue_ids(args.exclude_file)
    if excluded_ids:
        before = len(items)
        items = [
            item for item in items
            if item.get("dialogue_id") not in excluded_ids
            and f"multiwoz_{item.get('dialogue_id')}" not in excluded_ids
        ]
        print(f"Excluded existing dialogues: {before - len(items)}")

    if args.sort_by == "num_required_info":
        items.sort(
            key=lambda x: (
                x["num_required_info"],
                x["num_utterances"],
                x["selected_task"]["num_required_slots"],
            ),
            reverse=True,
        )
    else:
        items.sort(
            key=lambda x: (
                x["num_utterances"],
                x["num_required_info"],
                x["selected_task"]["num_required_slots"],
            ),
            reverse=True,
        )

    top_items = items[:args.top_n]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(top_items, f, ensure_ascii=False, indent=2)

    print(f"Valid dialogues: {len(items)}")
    print(f"Saved top: {len(top_items)}")
    print(f"Output: {args.output_file}")
    print(f"value_mode: {args.value_mode}")
    print(f"sort_by: {args.sort_by}")

    print("\nTop 20 preview:")
    for i, item in enumerate(top_items[:20], 1):
        t = item["selected_task"]
        print(
            f"{i:03d}. {item['dialogue_id']} | "
            f"utterances={item['num_utterances']} | "
            f"task={t['task_key']} | "
            f"required_info={t['num_required_info']} | "
            f"required_slots={t['num_required_slots']} | "
            f"checkpoints={t['checkpoint_turn_indices']} | "
            f"file={item['source_file']}"
        )


if __name__ == "__main__":
    main()
