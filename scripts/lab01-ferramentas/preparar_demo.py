from lab_runtime import read_jsonl, write_jsonl
rows = [r for r in read_jsonl('data/validation.jsonl') if r['eligible']][:50]
assert rows, 'Nao ha validacao elegivel'
write_jsonl('data/validation_demo.jsonl', rows)
