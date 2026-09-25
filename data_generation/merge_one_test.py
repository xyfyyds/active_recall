import os
import json
import random
import copy
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional


# =========================================================
# Config
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(SCRIPT_DIR)
SGD_DATA_DIR = os.path.join(RELEASE_ROOT, "SGD_derived_data")

INPUT_FILES = [
    os.path.join(SGD_DATA_DIR, f"dialogues_{index}_conversation_enriched_via_topic_evidence.jsonl")
    for index in (
        "001", "002", "016", "039", "053", "054", "055", "056", "069",
        "070", "091", "110", "111", "112", "113", "116", "117", "118",
    )
]

OUTPUT_PATH = os.path.join(RELEASE_ROOT, "cross_session_data", "cross_session_300.jsonl")

MAX_SAMPLES = 300
RANDOM_SEED = 42

# 目标 family 配额
BASE_TARGET_QUOTAS = {
    "RestaurantReservation": 70,
    "OnewayFlightReservation": 55,
    "AppointmentBooking": 50,
    "HotelReservation": 40,
    "CalendarEvent": 30,
    "BusTravel": 30,
    "CarReservation": 25,
}

# 配额不够时补给大的 family
FALLBACK_FAMILY_ORDER = [
    "RestaurantReservation",
    "OnewayFlightReservation",
    "AppointmentBooking",
    "HotelReservation",
    "CalendarEvent",
    "BusTravel",
    "CarReservation",
]

# target family -> 推荐背景 family 顺序
BACKGROUND_TEMPLATE = {
    "RestaurantReservation": [
        "CalendarEvent", "AppointmentBooking", "CarReservation",
        "HotelReservation", "OnewayFlightReservation", "BusTravel"
    ],
    "OnewayFlightReservation": [
        "RestaurantReservation", "CalendarEvent", "AppointmentBooking",
        "HotelReservation", "CarReservation", "BusTravel"
    ],
    "AppointmentBooking": [
        "RestaurantReservation", "CalendarEvent", "CarReservation",
        "HotelReservation", "OnewayFlightReservation", "BusTravel"
    ],
    "CalendarEvent": [
        "RestaurantReservation", "AppointmentBooking", "CarReservation",
        "HotelReservation", "OnewayFlightReservation", "BusTravel"
    ],
    "HotelReservation": [
        "RestaurantReservation", "CalendarEvent", "AppointmentBooking",
        "CarReservation", "OnewayFlightReservation", "BusTravel"
    ],
    "CarReservation": [
        "RestaurantReservation", "CalendarEvent", "AppointmentBooking",
        "HotelReservation", "OnewayFlightReservation", "BusTravel"
    ],
    "BusTravel": [
        "RestaurantReservation", "CalendarEvent", "AppointmentBooking",
        "HotelReservation", "OnewayFlightReservation", "CarReservation"
    ],
}


# =========================================================
# Basic IO
# =========================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# Task family mapping
# =========================================================
def normalize_text(s: Any) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def family_of(task: str) -> str:
    t = normalize_text(task)

    if "reserverestaurant" in t or "reserve a restaurant" in t or "reserve a table" in t:
        if "ride" in t or "getride" in t or "book a ride" in t or "get a ride" in t or "schedule a ride" in t:
            return "Restaurant+Ride"
        return "RestaurantReservation"

    if "reserveonewayflight" in t or "one-way flight" in t or "one way flight" in t:
        if "hotel" in t or "searchhotel" in t:
            return "Flight+Hotel"
        if "bus" in t or "findbus" in t or "bus ticket" in t:
            return "Flight+Bus"
        return "OnewayFlightReservation"

    if "reservehotel" in t or "reserve a hotel" in t or "reserve a hotel room" in t or "searchhotel" in t:
        if "flight" in t or "one-way flight" in t or "one way flight" in t:
            return "Flight+Hotel"
        if "bus" in t:
            return "Bus+Hotel"
        return "HotelReservation"

    if "bookappointment" in t or "dentist" in t or "doctor" in t or "salon" in t or "appointment" in t:
        return "AppointmentBooking"

    if "buybusticket" in t or "bookbusticket" in t or "findbus" in t or "bus ticket" in t or "bus tickets" in t:
        if "hotel" in t or "searchhotel" in t:
            return "Bus+Hotel"
        if "flight" in t:
            return "Flight+Bus"
        return "BusTravel"

    if "addevent" in t or "getavailabletime" in t or "getevents" in t or "calendar" in t:
        return "CalendarEvent"

    if "reservecar" in t or "reserve a rental car" in t or "reserve a car" in t or "getcarsavailable" in t:
        if "apartment" in t or "findapartment" in t:
            return "Car+Apartment"
        if "bus" in t or "flight" in t or "searchroundtripflights" in t:
            return "Car+Bus+Flight"
        return "CarReservation"

    if "findapartment" in t or "apartment" in t:
        return "ApartmentFinding"

    if "playmovie" in t:
        return "Movie"

    if "findrestaurants" in t and "findmovies" in t:
        return "Restaurant+Movie"

    if "transfermoney" in t or "checkbalance" in t:
        return "Finance"

    return "Other"


# =========================================================
# Segment helpers
# =========================================================
def get_original_segment_ids(item: Dict[str, Any]) -> List[int]:
    mapping = item.get("segment_required_info_mapping", {})
    segs = []
    if isinstance(mapping, dict) and mapping:
        for k in mapping.keys():
            try:
                segs.append(int(k))
            except Exception:
                pass
    return sorted(segs)


def get_turns_by_segment(conversation: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for turn in conversation:
        seg = turn.get("segment_id", None)
        if isinstance(seg, int):
            out[seg].append(turn)
    return out


def get_target_original_segments(item: Dict[str, Any]) -> List[Tuple[int, List[Dict[str, Any]]]]:
    conversation = item.get("conversation", []) or []
    by_seg = get_turns_by_segment(conversation)
    original_seg_ids = get_original_segment_ids(item)

    segments = []
    for seg_id in original_seg_ids:
        segments.append((seg_id, copy.deepcopy(by_seg.get(seg_id, []))))
    return segments


def choose_insert_position(n_turns: int, rng: random.Random) -> int:
    if n_turns <= 1:
        return n_turns

    if n_turns % 2 == 0:
        return n_turns // 2
    else:
        a = n_turns // 2
        b = a + 1
        return rng.choice([a, b])


# =========================================================
# Candidate filtering
# =========================================================
def is_good_target_candidate(item: Dict[str, Any]) -> bool:
    original_seg_ids = get_original_segment_ids(item)
    required_info = item.get("required_info", []) or []

    if len(original_seg_ids) not in {4, 5}:
        return False
    if len(required_info) < 4:
        return False
    if not item.get("segment_required_info_mapping"):
        return False
    return True


def is_good_background_candidate(item: Dict[str, Any]) -> bool:
    conversation = item.get("conversation", []) or []
    return len(conversation) >= 4


# =========================================================
# Gold / test case helpers
# =========================================================
def build_required_info_map(item: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for x in item.get("required_info", []) or []:
        name = str(x.get("name", "")).strip()
        gold_answer = str(x.get("gold_answer", "")).strip()
        if name:
            out[name] = gold_answer
    return out


def build_gold_solution(item: Dict[str, Any]) -> str:
    req_map = build_required_info_map(item)
    parts = []
    for k, v in req_map.items():
        parts.append(f"{k}: {v}")
    return "; ".join(parts)


def build_field_to_final_segment(item: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    mapping = item.get("segment_required_info_mapping", {})
    if not isinstance(mapping, dict):
        return out

    for seg_id_str, fields in mapping.items():
        try:
            seg_id = int(seg_id_str)
        except Exception:
            continue
        if not isinstance(fields, list):
            continue
        for f in fields:
            f = str(f).strip()
            if f:
                out[f] = seg_id
    return out


def get_last_segment_completion_local_turn(item: Dict[str, Any], last_seg_id: int) -> Optional[int]:
    ev = item.get("segment_required_info_evidence", {}) or {}
    seg_ev = ev.get(str(last_seg_id), [])
    if not isinstance(seg_ev, list) or not seg_ev:
        return None

    max_local = None
    for ann in seg_ev:
        if not isinstance(ann, dict):
            continue
        local_ids = ann.get("local_turn_ids", [])
        if not isinstance(local_ids, list):
            continue
        for x in local_ids:
            if isinstance(x, int):
                if max_local is None or x > max_local:
                    max_local = x
    return max_local


# =========================================================
# Family allocation
# =========================================================
def allocate_target_counts(candidates_by_family: Dict[str, List[Dict[str, Any]]], max_samples: int) -> Dict[str, int]:
    allocated = {}
    remaining = max_samples

    for fam, quota in BASE_TARGET_QUOTAS.items():
        avail = len(candidates_by_family.get(fam, []))
        take = min(quota, avail, remaining)
        allocated[fam] = take
        remaining -= take

    for fam in FALLBACK_FAMILY_ORDER:
        if remaining <= 0:
            break
        avail = len(candidates_by_family.get(fam, []))
        already = allocated.get(fam, 0)
        spare = max(0, avail - already)
        if spare <= 0:
            continue
        take_more = min(spare, remaining)
        allocated[fam] = already + take_more
        remaining -= take_more

    return allocated


# =========================================================
# Core generation
# =========================================================
def build_one_mixed_sample(
    target_item: Dict[str, Any],
    target_family: str,
    background_items: List[Dict[str, Any]],
    background_families: List[str],
    rng: random.Random,
    sample_index: int
) -> Dict[str, Any]:
    target_id = str(target_item.get("id", "")).strip()
    target_task = target_item.get("task", "")
    target_required_info = copy.deepcopy(target_item.get("required_info", []) or [])
    target_mapping = copy.deepcopy(target_item.get("segment_required_info_mapping", {}) or {})
    target_evidence_local = copy.deepcopy(target_item.get("segment_required_info_evidence", {}) or {})
    target_earliest = target_item.get("earliest_resolvable_segment", None)

    target_segments = get_target_original_segments(target_item)
    assert len(target_segments) == len(background_items)

    conversation = []
    target_segment_insertions = []
    target_segment_global_ranges = {}
    target_segment_local_to_global = {}

    global_turn_idx = 0

    for (seg_id, seg_turns), bg_item, bg_family in zip(target_segments, background_items, background_families):
        bg_id = str(bg_item.get("id", "")).strip()
        bg_conv = copy.deepcopy(bg_item.get("conversation", []) or [])
        split_pos = choose_insert_position(len(bg_conv), rng)

        before_turns = bg_conv[:split_pos]
        after_turns = bg_conv[split_pos:]

        for t in before_turns:
            conversation.append({
                "global_turn_id": global_turn_idx,
                "speaker": t.get("speaker", ""),
                "text": t.get("text", ""),
                "role_in_mixed_sample": "background",
                "family": bg_family,
                "dialogue_id": bg_id,
                "original_segment_id": t.get("segment_id", None),
                "is_target": False,
                "target_segment_id": None,
                "target_segment_local_turn_id": None,
            })
            global_turn_idx += 1

        seg_start_global = global_turn_idx
        local_to_global = {}
        for local_idx, t in enumerate(seg_turns):
            conversation.append({
                "global_turn_id": global_turn_idx,
                "speaker": t.get("speaker", ""),
                "text": t.get("text", ""),
                "role_in_mixed_sample": "target",
                "family": target_family,
                "dialogue_id": target_id,
                "original_segment_id": seg_id,
                "is_target": True,
                "target_segment_id": seg_id,
                "target_segment_local_turn_id": local_idx,
            })
            local_to_global[local_idx] = global_turn_idx
            global_turn_idx += 1
        seg_end_global = global_turn_idx - 1

        target_segment_global_ranges[str(seg_id)] = {
            "global_start_turn": seg_start_global,
            "global_end_turn": seg_end_global,
            "num_turns": len(seg_turns),
        }
        target_segment_local_to_global[str(seg_id)] = local_to_global

        target_segment_insertions.append({
            "target_segment_id": seg_id,
            "background_family": bg_family,
            "background_dialogue_id": bg_id,
            "background_insert_position_local": split_pos,
            "target_global_start_turn": seg_start_global,
            "target_global_end_turn": seg_end_global,
        })

        for t in after_turns:
            conversation.append({
                "global_turn_id": global_turn_idx,
                "speaker": t.get("speaker", ""),
                "text": t.get("text", ""),
                "role_in_mixed_sample": "background",
                "family": bg_family,
                "dialogue_id": bg_id,
                "original_segment_id": t.get("segment_id", None),
                "is_target": False,
                "target_segment_id": None,
                "target_segment_local_turn_id": None,
            })
            global_turn_idx += 1

    # evidence local -> global
    target_evidence_global = {}
    for seg_id_str, anns in target_evidence_local.items():
        local_to_global = target_segment_local_to_global.get(seg_id_str, {})
        out_anns = []
        if isinstance(anns, list):
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                local_ids = ann.get("local_turn_ids", [])
                global_ids = []
                if isinstance(local_ids, list):
                    for x in local_ids:
                        if isinstance(x, int) and x in local_to_global:
                            global_ids.append(local_to_global[x])

                out_anns.append({
                    "required_info": ann.get("required_info", ""),
                    "local_turn_ids": local_ids,
                    "global_turn_ids": global_ids,
                    "evidence_context": ann.get("evidence_context", []),
                })
        target_evidence_global[seg_id_str] = out_anns

    # build test cases inside this same row
    required_info_map = build_required_info_map(target_item)
    field_to_final_seg = build_field_to_final_segment(target_item)
    original_seg_ids = [seg_id for seg_id, _ in target_segments]
    last_seg_id = original_seg_ids[-1]
    last_completion_local = get_last_segment_completion_local_turn(target_item, last_seg_id)

    test_cases = []
    for seg_id in original_seg_ids:
        seg_range = target_segment_global_ranges[str(seg_id)]

        if seg_id != last_seg_id:
            test_turn_global = seg_range["global_end_turn"]
            gold_can_complete = "no"
            gold_missing = sorted([f for f in required_info_map.keys() if field_to_final_seg.get(f, 10**9) > seg_id])
            gold_solution = ""
        else:
            if last_completion_local is not None and last_completion_local in target_segment_local_to_global[str(seg_id)]:
                test_turn_global = target_segment_local_to_global[str(seg_id)][last_completion_local]
            else:
                test_turn_global = seg_range["global_end_turn"]

            gold_can_complete = "yes"
            gold_missing = []
            gold_solution = build_gold_solution(target_item)

        test_cases.append({
            "case_id": f"mixed_{sample_index:04d}__target_{target_id}__seg{seg_id}",
            "test_segment_id": seg_id,
            "test_turn_global_index": test_turn_global,
            "gold_can_complete": gold_can_complete,
            "gold_missing_required_info": gold_missing,
            "gold_solution": gold_solution,
        })

    row = {
        "id": f"mixed_{sample_index:04d}",

        "target_info": {
            "dialogue_id": target_id,
            "task": target_task,
            "family": target_family,
            "required_info": target_required_info,
            "segment_required_info_mapping": target_mapping,
            "segment_required_info_evidence_local": target_evidence_local,
            "segment_required_info_evidence_global": target_evidence_global,
            "earliest_resolvable_segment": target_earliest,
            "gold_solution": build_gold_solution(target_item),
        },

        "background_info": [
            {
                "dialogue_id": str(x.get("id", "")).strip(),
                "task": x.get("task", ""),
                "family": fam,
            }
            for x, fam in zip(background_items, background_families)
        ],

        "target_segment_insertions": target_segment_insertions,
        "target_segment_global_ranges": target_segment_global_ranges,

        "conversation": conversation,
        "test_cases": test_cases,
    }

    return row


# =========================================================
# Main
# =========================================================
def main():
    rng = random.Random(RANDOM_SEED)

    all_rows = []
    for path in INPUT_FILES:
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}")
            continue
        rows = load_jsonl(path)
        all_rows.extend(rows)

    rows_by_family = defaultdict(list)
    for item in all_rows:
        fam = family_of(item.get("task", ""))
        rows_by_family[fam].append(item)

    target_candidates_by_family = defaultdict(list)
    background_candidates_by_family = defaultdict(list)

    for fam, rows in rows_by_family.items():
        for item in rows:
            if is_good_target_candidate(item):
                target_candidates_by_family[fam].append(item)
            if is_good_background_candidate(item):
                background_candidates_by_family[fam].append(item)

    allocated = allocate_target_counts(target_candidates_by_family, MAX_SAMPLES)
    print("[Target allocation]")
    for fam in FALLBACK_FAMILY_ORDER:
        if fam in allocated:
            print(f"  {fam}: {allocated[fam]}")

    for fam in target_candidates_by_family:
        rng.shuffle(target_candidates_by_family[fam])
    for fam in background_candidates_by_family:
        rng.shuffle(background_candidates_by_family[fam])

    output_rows = []
    sample_index = 0

    for target_family in FALLBACK_FAMILY_ORDER:
        wanted = allocated.get(target_family, 0)
        if wanted <= 0:
            continue

        target_pool = target_candidates_by_family.get(target_family, [])
        used_target_count = 0

        for target_item in target_pool:
            if used_target_count >= wanted:
                break

            target_segments = get_target_original_segments(target_item)
            k = len(target_segments)

            bg_family_order = BACKGROUND_TEMPLATE.get(target_family, [])
            chosen_bg_families = []
            for fam in bg_family_order:
                if fam == target_family:
                    continue
                if len(background_candidates_by_family.get(fam, [])) > 0:
                    chosen_bg_families.append(fam)
                if len(chosen_bg_families) == k:
                    break

            if len(chosen_bg_families) < k:
                continue

            background_items = []
            used_bg_ids = set()
            bad = False

            for fam in chosen_bg_families:
                candidates = background_candidates_by_family[fam]
                picked = None
                for cand in candidates:
                    cid = str(cand.get("id", "")).strip()
                    if cid == str(target_item.get("id", "")).strip():
                        continue
                    if cid in used_bg_ids:
                        continue
                    picked = cand
                    break

                if picked is None:
                    bad = True
                    break

                background_items.append(picked)
                used_bg_ids.add(str(picked.get("id", "")).strip())

            if bad:
                continue

            sample_index += 1
            row = build_one_mixed_sample(
                target_item=target_item,
                target_family=target_family,
                background_items=background_items,
                background_families=chosen_bg_families,
                rng=rng,
                sample_index=sample_index
            )
            output_rows.append(row)
            used_target_count += 1

            if sample_index >= MAX_SAMPLES:
                break

        if sample_index >= MAX_SAMPLES:
            break

    save_jsonl(OUTPUT_PATH, output_rows)

    print("\nDone.")
    print(f"Generated mixed samples: {len(output_rows)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
