import json
from lab_runtime import read_jsonl
r = read_jsonl('data/train.jsonl')[0]
print(json.dumps(r, ensure_ascii=False, indent=2))
