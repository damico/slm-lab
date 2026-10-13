"""Reference outcomes for prepared ABCD demos; never part of model inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def history_fingerprint(history):
    data = json.dumps(history, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def event(raw, annotated, conversation_id, index):
    result = {'id': f'{conversation_id}:{index}', 'role': raw[0], 'text': raw[1]}
    if annotated['targets'][1] == 'take_action':
        result['action'] = annotated['targets'][2]
    return result


def build_outcomes(corpus, references):
    wanted = {str(row['id']).rsplit(':', 1)[0] for row in references}
    conversations = {str(c['convo_id']): c for c in corpus['test'] if str(c['convo_id']) in wanted}
    results = {}
    for reference in references:
        identifier = str(reference['id'])
        conversation_id, position = identifier.rsplit(':', 1)
        conversation = conversations[conversation_id]
        index = int(position)
        original, annotated = conversation['original'], conversation['delexed']
        if len(original) != len(annotated) or not 0 <= index < len(original):
            raise ValueError('Referencia desalinhada: ' + identifier)
        step = annotated[index]['targets'][1]
        label = annotated[index]['targets'][2] if step == 'take_action' else step
        if label != reference['expected']:
            raise ValueError('Rotulo da referencia mudou: ' + identifier)
        events = [event(raw, ann, conversation_id, i)
                  for i, (raw, ann) in enumerate(zip(original, annotated))]
        observed = [{key: value for key, value in e.items() if key != 'id'} for e in events[:index]]
        continuation = events[index:]
        replies = [e for e in continuation if e['role'] == 'agent']
        actions = [e for e in continuation if 'action' in e]
        results[identifier] = {
            'id': identifier, 'source': 'ABCD', 'expected': label,
            'history_sha256': history_fingerprint(observed),
            'target': events[index], 'continuation': continuation,
            'delivered_messages': replies,
            'last_action': actions[-1] if actions else None,
            'final_event': events[-1],
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--raw', help='Pasta ABCD contendo abcd_v1.1.json.gz')
    args = parser.parse_args()
    data = Path(args.data)
    raw = Path(args.raw) if args.raw else data.parent / 'raw' / 'abcd'
    source = raw / 'abcd_v1.1.json.gz'
    references = json.loads((data / 'demo_expected.json').read_text(encoding='utf-8'))
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        corpus = json.load(stream)
    outcomes = build_outcomes(corpus, references)
    path = data / 'demo_outcomes.json'
    path.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Desfechos ABCD preparados: {len(outcomes)} em {path}')


if __name__ == '__main__':
    main()
