

# =========================================================
# Paths
# =========================================================
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${RAW_DIR:-${WORKDIR}/raw_dialogues}"
OUT_DIR="${OUT_DIR:-${WORKDIR}/dialogues}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 如有需要，可在这里设置 API key
# export OPENAI_API_KEY="YOUR_API_KEY_HERE"

# =========================================================
# Script filenames
# =========================================================
SCRIPT_EXTRACT_REQ="extract_required_info_from_sgd.py"
SCRIPT_CLEAN_TOOL="clean_tool_info.py"
SCRIPT_SEGMENT="extract_segment.py"
SCRIPT_REFINE="refine_sgd.py"
SCRIPT_CHECK="check_.py"
SCRIPT_ENRICH="enrich_conversation_via_topic.py"
SCRIPT_EVIDENCE="evidence_annotation.py"

# Prompt filenames
PROMPT_REFINE="refine_sgd.txt"
PROMPT_TOPIC="topic_prompt.txt"
PROMPT_FILLER="enrich_via_topic.txt"

mkdir -p "${OUT_DIR}"

echo "========================================================="
echo "RAW_DIR  = ${RAW_DIR}"
echo "OUT_DIR  = ${OUT_DIR}"
echo "WORKDIR  = ${WORKDIR}"
echo "========================================================="

process_one() {
  local input_file="$1"
  local basename
  basename="$(basename "${input_file}" .json)"

  echo
  echo "============================================================"
  echo "Processing file: ${basename}.json"
  echo "============================================================"

  local req="${OUT_DIR}/${basename}_required_info.jsonl"
  local clean="${OUT_DIR}/${basename}_required_info_conversation.jsonl"
  local seg="${OUT_DIR}/${basename}_required_info_conversation_segment.jsonl"
  local refine="${OUT_DIR}/${basename}_conversation_refined.jsonl"
  local check1="${OUT_DIR}/${basename}_conversation_refined_checked.jsonl"
  local check2="${OUT_DIR}/${basename}_conversation_refined_checked_2.jsonl"
  local final="${OUT_DIR}/${basename}_conversation_enriched_via_topic.jsonl"
  local evidence="${OUT_DIR}/${basename}_conversation_enriched_via_topic_evidence.jsonl"

  echo "============================================================"
  echo "STEP 1: Extract required info"
  echo "INPUT : ${input_file}"
  echo "OUTPUT: ${req}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_EXTRACT_REQ}" \
    --input "${input_file}" \
    --output "${req}"

  if [[ ! -f "${req}" ]]; then
    echo "[SKIP] STEP 1 did not create output: ${req}"
    return 0
  fi

  if [[ ! -s "${req}" ]]; then
    echo "[SKIP] No matched dialogues in ${basename}.json . STEP 1 output is empty."
    return 0
  fi

  echo "============================================================"
  echo "STEP 2: Clean tool info"
  echo "INPUT : ${req}"
  echo "OUTPUT: ${clean}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_CLEAN_TOOL}" \
    --input "${req}" \
    --output "${clean}"

  if [[ ! -f "${clean}" || ! -s "${clean}" ]]; then
    echo "[SKIP] STEP 2 output is empty: ${clean}"
    return 0
  fi

  echo "============================================================"
  echo "STEP 3: Extract segment"
  echo "INPUT : ${clean}"
  echo "OUTPUT: ${seg}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_SEGMENT}" \
    --input "${clean}" \
    --output "${seg}"

  if [[ ! -f "${seg}" || ! -s "${seg}" ]]; then
    echo "[SKIP] STEP 3 output is empty: ${seg}"
    return 0
  fi

  echo "============================================================"
  echo "STEP 4: Refine conversation"
  echo "INPUT : ${seg}"
  echo "OUTPUT: ${refine}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_REFINE}" \
    --input "${seg}" \
    --output "${refine}" \
    --prompt-file "${WORKDIR}/${PROMPT_REFINE}"

  if [[ ! -f "${refine}" || ! -s "${refine}" ]]; then
    echo "[SKIP] STEP 4 output is empty: ${refine}"
    return 0
  fi

  echo "============================================================"
  echo "STEP 5: Check pass 1"
  echo "INPUT : ${refine}"
  echo "SAVED : ${check1}"
  echo "OUTPUT: ${check1}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_CHECK}" \
    --saved "${check1}" \
    --input "${refine}" \
    --output "${check1}"

  if [[ ! -f "${check1}" || ! -s "${check1}" ]]; then
    echo "[SKIP] STEP 5 output is empty: ${check1}"
    return 0
  fi

  echo "============================================================"
  echo "STEP 6: Check pass 2"
  echo "INPUT : ${refine}"
  echo "SAVED : ${check1}"
  echo "OUTPUT: ${check2}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_CHECK}" \
    --saved "${check1}" \
    --input "${refine}" \
    --output "${check2}"

  if [[ ! -f "${check2}" || ! -s "${check2}" ]]; then
    echo "[SKIP] STEP 6 output is empty: ${check2}"
    return 0
  fi

  echo "============================================================"
  echo "STEP 7: Enrich via topic"
  echo "INPUT : ${check2}"
  echo "OUTPUT: ${final}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_ENRICH}" \
    --input "${check2}" \
    --output "${final}" \
    --topic-prompt-file "${WORKDIR}/${PROMPT_TOPIC}" \
    --filler-prompt-file "${WORKDIR}/${PROMPT_FILLER}"

  if [[ ! -f "${final}" || ! -s "${final}" ]]; then
    echo "[SKIP] STEP 7 output is empty: ${final}"
    return 0
  fi

  echo "============================================================"
  echo "STEP 8: Annotate required-information evidence"
  echo "INPUT : ${final}"
  echo "OUTPUT: ${evidence}"
  "${PYTHON_BIN}" "${WORKDIR}/${SCRIPT_EVIDENCE}" \
    --input "${final}" \
    --output "${evidence}"

  echo "Done: ${basename}.json"
}

shopt -s nullglob
files=("${RAW_DIR}"/dialogues_*.json)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No files found in ${RAW_DIR}/dialogues_*.json"
  exit 0
fi

for f in "${files[@]}"; do
  process_one "${f}"
done

echo
echo "============================================================"
echo "All files processed successfully."
echo "============================================================"
