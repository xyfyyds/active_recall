import argparse
import json
from pathlib import Path
from typing import List, Tuple


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "MultiWOZ_derived_data"


def read_valid_lines(path: Path) -> List[Tuple[str, dict]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("error"):
                continue
            rows.append((raw, item))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the released MultiWOZ top-800 file."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=DATA_DIR / "multiwoz_top200_enriched_via_topic_evidence.jsonl",
        help="The original 140 retained examples from the top-200 run.",
    )
    parser.add_argument(
        "--expanded",
        type=Path,
        default=DATA_DIR / "multiwoz_top1200_enriched_via_topic_evidence.jsonl",
        help="The expanded candidate file generated after excluding the top-200 seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "multiwoz_top800_enriched_via_topic_evidence.jsonl",
    )
    parser.add_argument("--new-count", type=int, default=660)
    args = parser.parse_args()

    original = read_valid_lines(args.original)
    expanded = read_valid_lines(args.expanded)
    if len(expanded) < args.new_count:
        raise ValueError(
            f"Need {args.new_count} valid expanded rows, found {len(expanded)}."
        )

    original_ids = {str(item.get("id", "")) for _, item in original}
    selected_new = expanded[: args.new_count]
    overlap = original_ids & {str(item.get("id", "")) for _, item in selected_new}
    if overlap:
        raise ValueError(f"Original and expanded samples overlap: {sorted(overlap)[:5]}")

    output_lines = [raw for raw, _ in original] + [raw for raw, _ in selected_new]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        for raw in output_lines:
            handle.write(raw + "\r\n")

    print(
        f"Wrote {len(output_lines)} rows to {args.output} "
        f"({len(original)} original + {len(selected_new)} expanded)."
    )


if __name__ == "__main__":
    main()
