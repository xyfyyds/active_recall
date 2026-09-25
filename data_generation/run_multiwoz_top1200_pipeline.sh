#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python3}"
TOP_N="${TOP_N:-1200}"
PREFIX="${PREFIX:-multiwoz_top${TOP_N}}"
FORCE="${FORCE:-0}"
STOP_AFTER="${STOP_AFTER:-}"

RAW_DIR="${RAW_DIR:-./MultiWOZ_2_2}"
EXCLUDE_FILE="${EXCLUDE_FILE:-./multiwoz_top200_one_task.json}"

ONE_TASK="${PREFIX}_one_task.json"
SGD_STYLE="${PREFIX}_sgd_style.jsonl"
SEGMENT="${PREFIX}_segment.jsonl"
REFINED="${PREFIX}_refined.jsonl"
CHECK1="${PREFIX}_refined_checked.jsonl"
CHECK2="${PREFIX}_refined_checked_2.jsonl"
ENRICHED="${PREFIX}_enriched_via_topic.jsonl"
EVIDENCE="${PREFIX}_enriched_via_topic_evidence.jsonl"

REFINE_PROMPT="${REFINE_PROMPT:-./refine_sgd.txt}"
TOPIC_PROMPT="${TOPIC_PROMPT:-./topic_prompt.txt}"
FILLER_PROMPT="${FILLER_PROMPT:-./enrich_via_topic.txt}"

jsonl_count() {
  [[ -f "$1" ]] || { echo 0; return; }
  awk 'NF {c++} END {print c+0}' "$1"
}

stop_after() {
  if [[ "${STOP_AFTER}" == "$1" ]]; then
    echo "[STOP] STOP_AFTER=${STOP_AFTER}"
    exit 0
  fi
  return 0
}

echo "============================================================"
echo "[MultiWOZ top${TOP_N} pipeline]"
echo "RAW_DIR  : ${RAW_DIR}"
echo "PREFIX   : ${PREFIX}"
echo "EXCLUDE  : ${EXCLUDE_FILE}"
echo "FORCE    : ${FORCE}"
echo "============================================================"

if [[ "${FORCE}" == "1" || ! -s "${ONE_TASK}" ]]; then
  "${PYTHON}" select_top_MultiWoZ.py \
    --input_dir "${RAW_DIR}" \
    --top_n "${TOP_N}" \
    --output_file "${ONE_TASK}" \
    --exclude_file "${EXCLUDE_FILE}" \
    --value_mode final \
    --min_required_info 2 \
    --sort_by num_utterances
else
  echo "[SKIP] exists: ${ONE_TASK}"
fi
stop_after one_task

if [[ "${FORCE}" == "1" || ! -s "${SGD_STYLE}" ]]; then
  "${PYTHON}" process_multiwoz.py \
    --input_file "${ONE_TASK}" \
    --output_file "${SGD_STYLE}" \
    --min_required_info 2
else
  echo "[SKIP] exists: ${SGD_STYLE}"
fi
stop_after sgd

expected="$(jsonl_count "${SGD_STYLE}")"
actual="$(jsonl_count "${SEGMENT}")"
if [[ "${FORCE}" == "1" || ! -s "${SEGMENT}" || "${actual}" != "${expected}" ]]; then
  "${PYTHON}" extract_segment.py \
    --input "${SGD_STYLE}" \
    --output "${SEGMENT}"
else
  echo "[SKIP] complete: ${SEGMENT} (${actual}/${expected})"
fi
stop_after segment

"${PYTHON}" refine_sgd.py \
  --input "${SEGMENT}" \
  --output "${REFINED}" \
  --prompt-file "${REFINE_PROMPT}"
stop_after refined

"${PYTHON}" check_.py \
  --saved "${CHECK1}" \
  --input "${REFINED}" \
  --output "${CHECK1}"
stop_after check1

"${PYTHON}" check_.py \
  --saved "${CHECK1}" \
  --input "${REFINED}" \
  --output "${CHECK2}"
stop_after check2

ENRICH_ARGS=()
[[ "${FORCE}" == "1" ]] && ENRICH_ARGS+=(--overwrite)
"${PYTHON}" enrich_conversation_via_topic.py \
  --input "${CHECK2}" \
  --output "${ENRICHED}" \
  --topic-prompt-file "${TOPIC_PROMPT}" \
  --filler-prompt-file "${FILLER_PROMPT}" \
  "${ENRICH_ARGS[@]}"
stop_after enriched

"${PYTHON}" evidence_annotation.py \
  --input "${ENRICHED}" \
  --output "${EVIDENCE}"

echo "============================================================"
echo "[DONE]"
echo "Final output: ${EVIDENCE}"
echo "============================================================"
