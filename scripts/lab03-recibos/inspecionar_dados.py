import json
from lab_runtime import read_jsonl
r = read_jsonl('data/train.jsonl')[0]
print('OCR:\n', r['text'])
print('Alvo:', r['answer'])
print(json.dumps(json.load(open('data/length_report.json')), indent=2))
