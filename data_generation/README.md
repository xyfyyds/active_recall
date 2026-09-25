# Data-generation

## Environment

```bash
pip install -r data_generation/requirements.txt
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

The original source datasets are not duplicated here. Obtain SGD and MultiWOZ 2.2 separately and follow their respective licenses.

## Pipelines

### SGD-derived data

`run_pipeline.sh` applies the following stages to each raw SGD dialogue file:

1. `extract_required_info_from_sgd.py`
2. `clean_tool_info.py`
3. `extract_segment.py`
4. `refine_sgd.py` with `refine_sgd.txt`
5. `check_.py` (two checking passes)
6. `enrich_conversation_via_topic.py` with `topic_prompt.txt` and
   `enrich_via_topic.txt`
7. `evidence_annotation.py`

Example:

```bash
RAW_DIR=/path/to/sgd/dialogues \
OUT_DIR="/SGD_derived_data" \
bash data_generation/run_pipeline.sh
```

### MultiWOZ-derived data

data presented in ./MultiWOZ-derived_data

### Cross-session data

`merge_one_test.py` deterministically combines examples from 18 released SGD
files using the fixed random seed and quotas in the script:

```bash
python data_generation/merge_one_test.py
```

### If-Then data

data presented in ./if_then
