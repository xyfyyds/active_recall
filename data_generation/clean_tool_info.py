import json
import argparse
from typing import Any, Dict, List
from tqdm import tqdm


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


def clean_conversation(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []

    for turn in conversation:
        speaker = str(turn.get("speaker", "")).strip().lower()
        text = normalize_text(turn.get("text", ""))

        if not text:
            continue
        if speaker == "tool info":
            continue
        if speaker not in {"user", "assistant"}:
            continue

        cleaned.append({
            "speaker": speaker,
            "text": text
        })

    return cleaned


def simplify_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "task": item.get("task", ""),
        "required_info": item.get("required_info", []),
        "conversation": clean_conversation(item.get("conversation", []))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dialogues_002_required_info.jsonl")
    parser.add_argument("--output", default="dialogues_002_required_info_conversation.jsonl")
    args = parser.parse_args()

    data = load_jsonl(args.input)

    open(args.output, "w", encoding="utf-8").close()

    for item in tqdm(data, desc="Cleaning"):
        new_item = simplify_item(item)
        append_jsonl(args.output, new_item)

    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()