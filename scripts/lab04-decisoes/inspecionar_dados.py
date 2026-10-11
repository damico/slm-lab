from decision_core import read_json, read_rows
r = read_json('data/preparation_report.json')
for name, s in r['splits'].items():
    print(name, s['conversations'], 'conversas;', s['decisions'], 'decisoes;',
          s['eligible'], 'elegiveis;', s['with_history_window'], 'com janela')
row = next(r for r in read_rows('data/train.jsonl') if r['label'] != 'retrieve_utterance')
print('\nENTRADA:\n', row['text'])
print('\nALVO:', row['label'])
