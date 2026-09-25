import os
import re
import json
import argparse
from typing import Any, Dict, List
from tqdm import tqdm


OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

TARGET_SERVICES = {
    "Restaurants_1",
    "Hotels_1",
    "Hotels_3",
    "Flights_1",
    "RentalCars_1",
    "RentalCars_2",
    "Calendar_1",
}


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


def keep_dialogue_by_service(dialog: Dict[str, Any]) -> bool:
    services = dialog.get("services", []) or []
    return any(str(s).strip() in TARGET_SERVICES for s in services)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, data: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def action_to_text(action: Dict[str, Any], service: str) -> str:
    """
    只保留真正有外部返回值的工具信息。
    没有 values / canonical_values 的 action，不加入对话。
    """
    values = action.get("canonical_values") or action.get("values") or []
    values = [str(v).strip() for v in values if str(v).strip()]
    if not values:
        return ""

    act = str(action.get("act", "")).strip()
    slot = str(action.get("slot", "")).strip()
    value_text = ", ".join(values)

    parts = []
    if service:
        parts.append(f"service={service}")
    if act:
        parts.append(f"act={act}")
    if slot:
        parts.append(f"slot={slot}")
    parts.append(f"value={value_text}")
    return " | ".join(parts)


def build_conversation(dialog: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    把原始 dialogue 转成干净的 conversation：
    - user / assistant 正常保留
    - tool info 只在真的有外部结果时插入到对应 turn 后面
    """
    conversation: List[Dict[str, str]] = []

    for turn in dialog.get("turns", []):
        speaker_raw = str(turn.get("speaker", "")).strip().lower()
        utterance = normalize_text(turn.get("utterance", ""))

        if speaker_raw == "user":
            speaker = "user"
        elif speaker_raw == "system":
            speaker = "assistant"
        else:
            speaker = speaker_raw if speaker_raw else "unknown"

        if utterance:
            conversation.append({
                "speaker": speaker,
                "text": utterance
            })

        for frame in turn.get("frames", []):
            service = str(frame.get("service", "")).strip()
            for action in frame.get("actions", []):
                tool_text = action_to_text(action, service)
                if tool_text:
                    conversation.append({
                        "speaker": "tool info",
                        "text": tool_text
                    })

    return conversation


def build_task(dialog: Dict[str, Any]) -> str:
    """
    给 LLM 的 task 输入尽量简洁：
    - services
    - 出现过的 active intents
    """
    services = dialog.get("services", []) or []
    intents: List[str] = []

    for turn in dialog.get("turns", []):
        for frame in turn.get("frames", []):
            state = frame.get("state", {}) or {}
            intent = state.get("active_intent", "")
            if intent and intent != "NONE" and intent not in intents:
                intents.append(intent)

    parts = []
    if services:
        parts.append("Services: " + ", ".join(map(str, services)))
    if intents:
        parts.append("Intents: " + ", ".join(map(str, intents)))

    return " | ".join(parts).strip()


def build_prompt(task: str, conversation: List[Dict[str, str]]) -> str:
    return f"""
You are given a task and a conversation.

Your job is to extract only the minimum required information needed to complete the task.

Return a JSON object with exactly this format:
{{
  "task": "...",
  "required_info": [
    {{
      "name": "...",
      "gold_answer": "..."
    }}
  ]
}}

Requirements:
1. Only keep necessary required info.
2. Do not output redundant fields.
3. Each required info must include:
   - "name": a short field name
   - "gold_answer": the answer inferred from the conversation
4. Keep gold_answer concise.
5. Do not add explanation.
6. If multiple required fields are needed, output all of them.
7. If a field is not explicitly answered in the conversation but is still required for task completion, you may keep "gold_answer" as an empty string.

Task:
{task}

Conversation:
{json.dumps(conversation, ensure_ascii=False, indent=2)}
""".strip()


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"task": "", "required_info": []}

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

    return {"task": "", "required_info": []}


def clean_required_info(required_info: Any) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    if not isinstance(required_info, list):
        return cleaned

    for item in required_info:
        if isinstance(item, dict):
            name = normalize_text(item.get("name", ""))
            gold_answer = normalize_text(item.get("gold_answer", ""))
            if name:
                cleaned.append({
                    "name": name,
                    "gold_answer": gold_answer
                })
        elif isinstance(item, str):
            name = normalize_text(item)
            if name:
                cleaned.append({
                    "name": name,
                    "gold_answer": ""
                })

    # 去重
    seen = set()
    deduped = []
    for item in cleaned:
        key = item["name"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def process_dialog(
    dialog: Dict[str, Any],
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    dialog_id = dialog.get("dialogue_id", "")
    task = build_task(dialog)
    conversation = build_conversation(dialog)

    prompt = build_prompt(task, conversation)
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = safe_parse_json(raw)

    task_out = normalize_text(parsed.get("task", "")) or task
    required_info = clean_required_info(parsed.get("required_info", []))

    return {
        "id": dialog_id,
        "task": task_out,
        "required_info": required_info,
        "conversation": conversation
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dialogues_002.json")
    parser.add_argument("--output", default="dialogues_002_required_info.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dialogues.")

    data = [d for d in data if keep_dialogue_by_service(d)]

    if not data:
        print("No dialogues matched allowed services. Skip this file.")
        return

    client = OpenAICompatibleClient(
        base_url=OPENAI_BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_MODEL,
    )

    total = len(data) if args.max_items <= 0 else min(len(data), args.max_items)

    # 每次运行先清空输出文件
    open(args.output, "w", encoding="utf-8").close()

    for idx, dialog in enumerate(tqdm(data[:total], desc="Processing dialogues"), start=1):
        try:
            item = process_dialog(dialog, client, args.temperature, args.max_tokens)
            append_jsonl(args.output, item)
            print(f"[{idx}/{total}] done: {dialog.get('dialogue_id', 'unknown')}")
        except Exception as e:
            print(f"[{idx}/{total}] failed: {dialog.get('dialogue_id', 'unknown')} | {e}")
            bad = {
                "id": dialog.get("dialogue_id", ""),
                "task": build_task(dialog),
                "required_info": [],
                "conversation": build_conversation(dialog),
                "error": str(e),
            }
            append_jsonl(args.output, bad)

    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
