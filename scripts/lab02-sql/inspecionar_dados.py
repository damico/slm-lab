from lab_runtime import read_jsonl
r = read_jsonl('data/train.jsonl')[0]
print(r['messages'][1]['content'])
print('\nSQL de treino:', r['answer'])
print('\nBanco:', r['db_path'])
