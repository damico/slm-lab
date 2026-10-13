"""Contrato, recuperação sem gabarito e executor restrito do Lab 05."""
import hashlib
import json
import math
import re
from pathlib import Path

STUDENT = 'Qwen/Qwen3-1.7B'
TEACHER = 'Qwen/Qwen3-8B'
FINQA_REVISION = '0f16e2867befa6840783e58be38c9efb9229d742'
OPS = {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater',
       'table_sum', 'table_average', 'table_min', 'table_max'}
CONSTANTS = {f'const_{n}': float(n) for n in [1,2,3,4,5,6,7,8,9,10,100,1000,10000,100000,1000000,1000000000]}
CONSTANTS['const_m1'] = -1.0
SYSTEM = '''Answer a financial numerical question using only the supplied sources. Return only a JSON object with exactly "evidence" (a nonempty array of at most 8 source IDs) and "program" (a string). Example format: {"evidence":["table_1"],"program":"subtract(118, 100), divide(#0, 100), multiply(#1, const_100)"}.
Programs contain up to 12 sequential binary calls separated by commas. Allowed: add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_min, table_max. #0 is the result of the first call; references must point backwards. greater returns yes/no. Table calls use an exact row label and none: table_sum(revenue, none). Other operands must be numbers from your cited sources or question, or one of these constants: const_m1, const_1, const_2, const_3, const_4, const_5, const_6, const_7, const_8, const_9, const_10, const_100, const_1000, const_10000, const_100000, const_1000000, const_1000000000. Percent literals are fractions: 10% means 0.1. Do not assume an extra multiplication by 100 unless the question requires it. Cite the sources of the actual operands. Do not emit Python, explanations, a guessed answer, or Markdown.'''


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def number(value):
    if not isinstance(value, str) or len(value) > 60: raise ValueError('Operando numérico inválido')
    text = value.strip().replace(',', '').replace('$', '').replace(' ', '')
    percent = text.endswith('%')
    if percent: text = text[:-1]
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', text): raise ValueError('Número inválido: ' + value)
    out = float(text) / (100 if percent else 1)
    if not math.isfinite(out) or abs(out) > 1e15: raise ValueError('Número fora do limite')
    return out


def cell_number(text):
    # Tabela normalizada FinQA já explicita o sinal antes de parênteses explicativos.
    text = text.replace('$', '').strip()
    if text.startswith('(') and text.endswith(')'): return -number(text[1:-1])
    return number(text.split('(')[0].strip())


def parse_program(text):
    if not isinstance(text, str) or not 0 < len(text) <= 3000: raise ValueError('Programa vazio ou grande')
    calls = []; pos = 0
    while pos < len(text):
        match = re.match(r'\s*([a-z_]+)\(', text[pos:])
        if not match or match[1] not in OPS: raise ValueError('Operação inválida')
        start = pos + match.end(); cursor = start; depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == '(': depth += 1
            if text[cursor] == ')': depth -= 1
            cursor += 1
        if depth: raise ValueError('Parênteses incompletos')
        args = text[start:cursor-1].rsplit(',', 1)
        if len(args) != 2: raise ValueError('Cada operação requer dois argumentos')
        calls.append((match[1], args[0].strip(), args[1].strip()))
        if len(calls) > 12: raise ValueError('Mais de 12 operações')
        tail = text[cursor:].lstrip()
        if not tail: break
        if not tail.startswith(',') or not tail[1:].strip(): raise ValueError('Texto após o programa')
        pos = len(text) - len(tail) + 1
    return calls


def source_map(raw):
    sources = {f'text_{i}': text for i, text in enumerate(raw['pre_text'] + raw['post_text'])}
    table = raw['table']; header = table[0] if table else []
    for i, row in enumerate(table):
        sources[f'table_{i}'] = ('Row label: ' + str(row[0]) + ' | ' + ' | '.join(
            f'{header[j] if j < len(header) else j}: {value}' for j, value in enumerate(row[1:], 1)))
    return sources


def retrieve(raw, top_k=12):
    # Ajuste TF-IDF por documento. Usa somente contexto e pergunta, nunca qa.gold_inds/model_input.
    from collections import Counter
    sources = source_map(raw); ids = list(sources)
    if len(ids) <= top_k: chosen = ids
    else:
        def features(text):
            words = re.findall(r'\w+', text.lower())
            return Counter(words + [' '.join(pair) for pair in zip(words,words[1:])])
        docs = [features(sources[key]) for key in ids]
        frequencies = Counter(word for doc in docs for word in doc)
        idf = {word: math.log((1+len(docs))/(1+count))+1 for word,count in frequencies.items()}
        query = features(raw['qa']['question'])
        qvec = {word:count*idf.get(word,1) for word,count in query.items()}
        qnorm = math.sqrt(sum(v*v for v in qvec.values())) or 1
        scores = []
        for doc in docs:
            vec = {word:count*idf[word] for word,count in doc.items()}
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1
            scores.append(sum(value*qvec.get(word,0) for word,value in vec.items())/(norm*qnorm))
        chosen = [ids[i] for i in sorted(range(len(ids)), key=lambda i:(-scores[i], i))[:top_k]]
    if 'table_0' in sources and 'table_0' not in chosen: chosen.append('table_0')
    return {key: sources[key] for key in chosen}


def messages(context):
    return [{'role':'system', 'content':SYSTEM}, {'role':'user', 'content':
        'Sources:\n' + '\n'.join(f'[{key}] {text}' for key,text in context['sources'].items())
        + '\nQuestion:\n' + context['question']}]


def parse_output(text):
    def pairs(items):
        out = {}
        for key,value in items:
            if key in out: raise ValueError('Chave JSON duplicada')
            out[key] = value
        return out
    payload = json.loads(text.strip(), object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Constante JSON inválida')))
    if not isinstance(payload, dict) or set(payload) != {'evidence','program'}: raise ValueError('Contrato JSON inválido')
    return payload


def numeric_sources(context, evidence):
    values = set()
    text = context['question'] + '\n' + '\n'.join(context['sources'][key] for key in evidence)
    for token in re.findall(r'[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*%?', text):
        try:
            values.add(number(token)); values.add(number(token.replace('%','')))
        except ValueError: pass
    return values


def execute(payload, context, grounded=True):
    evidence = payload.get('evidence')
    if not isinstance(evidence,list) or not 1 <= len(evidence) <= 8 or any(not isinstance(k,str) or k not in context['sources'] for k in evidence):
        raise ValueError('Evidência ausente ou fora do contexto')
    if len(set(evidence)) != len(evidence): raise ValueError('Evidência duplicada')
    values = numeric_sources(context,evidence); results=[]; trace=[]
    def operand(text):
        if text in CONSTANTS: return CONSTANTS[text]
        if re.fullmatch(r'#\d+',text):
            idx=int(text[1:])
            if idx >= len(results) or isinstance(results[idx],str): raise ValueError('Referência inválida')
            return results[idx]
        value=number(text)
        if grounded and not any(math.isclose(value,v,rel_tol=1e-9,abs_tol=1e-9) for v in values):
            raise ValueError('Operando sem origem nas evidências: '+text)
        return value
    for op,a,b in parse_program(payload.get('program')):
        if op.startswith('table_'):
            if b != 'none': raise ValueError('Operando de tabela deve terminar em none')
            matches=[(i,row) for i,row in enumerate(context['table']) if row[0] == a]
            if len(matches) != 1: raise ValueError('Linha de tabela ausente ou ambígua')
            i,row=matches[0]
            if grounded and f'table_{i}' not in evidence: raise ValueError('Linha de tabela não citada')
            nums=[cell_number(v) for v in row[1:]]
            if not nums: raise ValueError('Linha vazia')
            value={'table_sum':sum,'table_min':min,'table_max':max,'table_average':lambda ns:sum(ns)/len(ns)}[op](nums)
        else:
            x,y=operand(a),operand(b)
            if op=='divide' and y==0: raise ValueError('Divisão por zero')
            if op=='exp' and (abs(y)>12 or abs(x)>1e6): raise ValueError('Expoente fora do limite')
            if op=='greater': value='yes' if x>y else 'no'
            else:
                value={'add':lambda:x+y,'subtract':lambda:x-y,'multiply':lambda:x*y,
                       'divide':lambda:x/y,'exp':lambda:x**y}[op]()
        if not isinstance(value,str) and (isinstance(value,complex) or not math.isfinite(value) or abs(value)>1e18): raise ValueError('Resultado fora do limite')
        results.append(value);trace.append({'step':len(results)-1,'op':op,'arg1':a,'arg2':b,'result':value})
    if not results: raise ValueError('Programa vazio')
    value=results[-1]
    return {'answer':value if isinstance(value,str) else round(value,5), 'trace':trace,
            'evidence':{key:context['sources'][key] for key in evidence}}


def correct(value, gold):
    if isinstance(gold,str) and gold in ('yes','no'): return value==gold
    if isinstance(value,str) or isinstance(gold,bool): return False
    try: return round(float(value),5)==round(float(gold),5)
    except (ValueError,TypeError): return False
