# Laboratório 04, SLM especialista em decisões com ABCD

**Objetivo:** treinar e servir localmente um modelo que recebe o histórico de atendimento e escolhe o próximo passo entre alternativas conhecidas, retornando probabilidades e uma indicação de revisão.

**Hardware-alvo:** duas RTX 3070 de **8 GB cada**. **Base:** Qwen3-1.7B. **Método:** classificação de sequência com QLoRA e calibração posterior. **Dataset público:** ABCD. **Aplicação:** CLI e API HTTP local em `/decide`.

Este é o quarto laboratório da série. É adequado à apresentação técnica de uma hora: você executa preparação, treino e avaliação com antecedência, e demonstra a decisão e a API durante a sessão.

## 1. O resultado utilizável

A aplicação recebe eventos anteriores de um atendimento e responde com:

- `choice`: próxima ação recomendada, ou `retrieve_utterance` quando é preciso responder/perguntar ao cliente.
- `probabilities`: distribuição estimada sobre o catálogo fixo de alternativas.
- `confidence`: maior probabilidade após o ajuste de temperatura.
- `requires_review`: indica se a probabilidade ficou abaixo do limiar escolhido em dados separados.
- `policy_ids` e `dropped_turns`: permitem inspecionar os trechos de regras recuperados e o uso de janela de histórico.

O código consumidor pode exibir a recomendação para um atendente ou escolher a próxima etapa de um fluxo. A API deste lab devolve a decisão; a execução, os argumentos das ações e as regras de autorização pertencem à aplicação integradora.

Um pedido de devolução pode exigir primeiro consultar a conta, validar a compra ou coletar informações. O desafio é escolher a etapa apropriada usando o que já foi observado. As probabilidades ajudam a decidir quando encaminhar o caso para revisão, desde que sua qualidade seja medida.

### O sentido de “Jev-like” neste laboratório

A inspiração é a interface **estado → escolha entre opções → probabilidades**, documentada pela TypeSafe para o Jev. Aqui usamos um modelo aberto com uma camada de classificação. Não reproduzimos a arquitetura, o treinamento RLCD nem os resultados do Jev.

O catálogo é fixo. Adicionar uma ação nova exige adaptar e treinar a camada de saída. Oferecer alternativas arbitrárias a cada chamada exigiria outro desenho, como um modelo que pontua pares cenário–opção. Este roteiro não implementa essa segunda modalidade.

## 2. O que foi conferido no ABCD

O repositório oficial é [asappresearch/abcd](https://github.com/asappresearch/abcd), com licença MIT e diálogos humanos em cenários simulados de atendimento. A revisão usada é:

```text
6b8700ce67c6b37b062dd7a60abc76d7ef832a97
```

A inspeção do arquivo `abcd_v1.1.json.gz` dessa revisão encontrou:

| Split original | Conversas | Turnos com ação | Turnos em que o agente fala |
|---|---:|---:|---:|
| Treino | 8.034 | 29.190 | 75.985 |
| Desenvolvimento | 1.004 | 3.684 | 9.600 |
| Teste | 1.004 | 3.608 | 9.544 |

O catálogo tem 30 ações. **A documentação também prevê `end_conversation`, mas esse rótulo não aparece nos exemplos deste arquivo.** Por isso, o classificador tem **31 classes: as 30 ações e `retrieve_utterance`**. Encerramento fica fora do modelo. Essa é uma correção do desenho preliminar, baseada nos dados reais.

O corpus inclui intenções e cenários usados na coleta. Eles não entram no prompt: revelariam informação que a aplicação ainda precisa inferir. Usamos textos originais, anotações de ações anteriores já executadas e trechos das regras públicas. Não há anotação manual de novos exemplos.

São atendimentos em inglês e políticas de uma empresa fictícia. O experimento valida esse domínio; português, novas empresas e novas políticas exigem avaliação própria. O gabarito é o passo registrado na conversa, não uma prova de que somente esse passo seria aceitável. O resultado do lab não é o score oficial completo de AST/CDS, pois não avalia argumentos, seleção da frase nem sucesso final de uma conversa autônoma.

## 3. Desenho técnico e prevenção de vazamento

O Qwen processa o contexto e a camada `score` produz 31 logits. A perda supervisionada é entropia cruzada sobre o rótulo correto. Os pesos quantizados da base ficam congelados; treinamos LoRA e a camada de classificação, que permanece em ponto flutuante e é salva junto do adaptador.

**O texto do turno-alvo nunca entra na entrada daquele exemplo.** As anotações `targets` descrevem o próprio turno, e não o turno seguinte. O conversor emite o exemplo antes de adicionar o evento ao histórico. Isso evita, por exemplo, prever `offer-refund` lendo uma mensagem que já anuncia o reembolso.

As conversas são mantidas inteiras no mesmo split. O treino original fornece os exemplos de ajuste; o desenvolvimento é dividido por conversa para finalidades distintas:

| Conjunto do lab | Origem | Uso |
|---|---|---|
| `train` | Até 2.000 conversas do treino, por padrão | Ajustar LoRA e camada de saída |
| `validation` | 502 conversas do desenvolvimento | Selecionar checkpoint/hiperparâmetros |
| `calibration` | 251 conversas do desenvolvimento | Ajustar uma temperatura |
| `selection` | 251 conversas do desenvolvimento | Escolher limiar para revisão |
| `test` | 1.004 conversas do teste original | Medir tudo depois de congelar a receita |

O limite de 2.000 conversas é um ponto de partida para o laboratório. Para usar todo o treino, há um comando específico adiante. Duplicatas de conversas completas de dev/test são removidas do treino; duplicatas internas do treino também. Saudações e outros prefixos comuns podem continuar aparecendo em conversas diferentes.

### Como as regras entram na decisão

O script transforma as regras públicas em 255 trechos e usa TF-IDF para recuperar os mais relacionados às últimas mensagens. A consulta não recebe a intenção correta nem o subfluxo do gabarito. O recuperador é fixo e compartilhado por todas as condições.

O orçamento inicial é de 400 tokens para regras dentro de 2.048 tokens totais. Só entram trechos completos. Se o contexto ultrapassar o limite, os eventos mais antigos são retirados da janela, mantendo uma lista dos nomes de ações já observadas. Isso preserva a existência de uma ação anterior, mas pode perder seus argumentos, a ordem de repetições e outros detalhes; `dropped_turns` permite medir essa limitação. Se nem o evento mais recente couber, o caso é marcado como inelegível e encaminhado para revisão.

Esse conjunto combina adaptação do modelo e recuperação de regras. O comparativo mede a solução completa com entradas iguais. Para atribuir causalmente um ganho apenas ao LoRA, seria necessário um controle adicional com a mesma camada de classificação treinada sobre a base congelada, sem LoRA.

## 4. Preparar o ambiente nas duas placas de 8 GB

Use Ubuntu, Bash, Python 3.11 ou 3.12, aproximadamente 32 GB de RAM e uma reserva inicial de 30–45 GB em disco. O dataset descompactado e os JSONLs com históricos ocupam mais espaço que o download. Esses valores são planejamento, não medições de treino nas suas GPUs.

**O DDP não soma os 8 GB das placas.** Cada GPU recebe uma cópia quantizada e processa seus exemplos. O modelo e as ativações precisam caber em uma placa. O ganho de velocidade da segunda GPU depende da comunicação e deve ser medido.

Prepare primeiro o ambiente compartilhado conforme [Preparação geral](preparacao-geral.md).

Crie uma pasta independente dos outros labs e copie este Markdown para ela com o nome `laboratorio.md`. O exemplo considera o repositório em `~/git/slm-lab`; ajuste o caminho de origem se você o salvou em outro lugar:

```bash
mkdir -p lab04-decisoes
cd lab04-decisoes
cp ~/git/slm-lab/laboratorio-04-slm-decisoes-abcd.md ./laboratorio.md
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
python3 --version
```

```bash
source ~/slm-labs/.venv/bin/activate
cp ~/slm-labs/requirements.txt ./requirements.txt
cp ~/slm-labs/requirements-resolved.txt ./requirements-resolved.txt
mkdir -p raw runs results logs
export TOKENIZERS_PARALLELISM=false
```

As versões são fixadas para reprodutibilidade, não por serem as mais recentes. O wheel cu128 exige driver compatível; consulte a distribuição oficial do PyTorch se CUDA falhar. Os comandos são para Bash, não para PowerShell. Em Windows, use um ambiente Linux/WSL2 com as GPUs visíveis.

```bash
python - <<'PY'
import torch
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 2**30, 2), 'GiB',
          'BF16:', torch.cuda.is_bf16_supported())
PY
```

## 5. Extrair os scripts deste documento

O apêndice contém todo o código. O bloco abaixo extrai os cinco arquivos para a pasta atual, sem executar os demais comandos do Markdown. Ele exige uma pasta sem esses scripts para evitar sobrescrita.

```bash
python - <<'PY'
from pathlib import Path
import re
source = Path('laboratorio.md')
if not source.is_file():
    raise SystemExit(
        'Arquivo laboratorio.md ausente na pasta atual. '
        'Copie laboratorio-04-slm-decisoes-abcd.md para esta pasta '
        'com o nome laboratorio.md, conforme a seção 4.'
    )
text = source.read_text(encoding='utf-8')
blocks = re.findall(r"^cat > ([a-z_]+\.py) <<'PY'\n(.*?)^PY$", text, re.M | re.S)
expected = {'decision_core.py', 'prepare_abcd.py', 'decision_lab.py',
            'evaluate_decisions.py', 'serve_decisions.py'}
assert len(blocks) == 5 and {name for name, _ in blocks} == expected
assert not any(Path(name).exists() for name, _ in blocks), 'Use uma pasta nova'
for name, code in blocks:
    compile(code, name, 'exec')
    Path(name).write_text(code, encoding='utf-8')
    print('Criado:', name)
PY
```

## 6. Baixar e preparar os dados

```bash
set -o pipefail
python prepare_abcd.py --out data --train-conversations 2000 \
  --max-seq 2048 --policy-tokens 400 2>&1 | tee logs/prepare.log
```

O script baixa os dados da revisão fixada, registra a revisão do modelo em `model_revision.json`, monta as entradas e cria os cinco splits. A preparação usa CPU e o tokenizer; os pesos do modelo serão baixados no treino/inferência. Use uma pasta `--raw` exclusiva deste lab; os arquivos em cache não são baixados novamente.

Verifique os tamanhos, a distribuição de classes e os limites:

```bash
python - <<'PY'
from decision_core import read_json, read_rows
r = read_json('data/preparation_report.json')
for name, s in r['splits'].items():
    print(name, s['conversations'], 'conversas;', s['decisions'], 'decisoes;',
          s['eligible'], 'elegiveis;', s['with_history_window'], 'com janela')
row = next(r for r in read_rows('data/train.jsonl') if r['label'] != 'retrieve_utterance')
print('\nENTRADA:\n', row['text'])
print('\nALVO:', row['label'])
PY
```

`data/bundle.json` contém a ordem das classes, as regras e a configuração de construção do contexto. Preserve-o junto do adaptador e da calibração. Alterar o catálogo ou seu ordenamento invalida a camada de classificação.

Para o treino completo, prepare outra pasta e use-a em **todos** os comandos subsequentes com `--data data_full`, além de ajustar os caminhos dos JSONLs:

```bash
python prepare_abcd.py --out data_full --train-conversations 0 \
  --max-seq 2048 --policy-tokens 400
```

Escolha o tamanho do treino pela validação, antes de abrir o teste. Se uma classe não estiver presente no subconjunto de treino, o script pede para aumentar o número de conversas. Os registros inelegíveis permanecem no teste; são contados como falhas na acurácia global e como revisão no uso local.

## 7. Teste curto e treinamento

Configuração inicial: rank 16, alpha 32, dropout 0,05, learning rate `1e-4`, duas épocas, microbatch 1 e acumulação 8. Com duas GPUs, o batch efetivo é 16 decisões por atualização. O runtime usa NF4, dupla quantização, SDPA, gradient checkpointing e BF16 quando suportado, ou FP16.

Uma mensagem dizendo que `score.weight` foi inicializado é esperada na criação da base classificadora: essa camada não existe no checkpoint de linguagem e precisa ser aprendida. O adaptador salvo inclui a camada treinada por meio de `modules_to_save=["score"]`. Recarregar somente os pesos da base produziria uma classificação sem treinamento.

```bash
CUDA_VISIBLE_DEVICES=1 python decision_lab.py train --data data \
  --run runs/smoke_1gpu --max-steps 10 --accum 1 --eval-steps 5 --eval-limit 16 \
  2>&1 | tee logs/smoke_1gpu.log
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  decision_lab.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-steps 10 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
```

O treino principal começa novamente da base original, em outra pasta:

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB04_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc_per_node=2 decision_lab.py train \
  --data data --run runs/main --epochs 2 --accum 8 --rank 16 --lr 0.0001 \
  --eval-steps 100 --eval-limit 512 2>&1 | tee logs/main_train.log
kill "$LAB04_MONITOR_PID"
```

A cada 100 passos, o Trainer avalia a perda em até 512 decisões fixas da validação. Ao terminar, carrega o melhor checkpoint avaliado e salva `runs/main/final_adapter`. O relatório de treino registra tempo e pico de memória alocada no processo principal; o CSV do `nvidia-smi` registra ambas as placas.

Não há duração ou consumo de VRAM medidos no seu equipamento neste tutorial. Estime o tempo com passos após o aquecimento e a aproximação `ceil(decisões_elegíveis_de_treino / 16) × épocas`, acrescentando avaliação e gravação.

Se faltar memória, feche outras inferências, reduza rank para 8 e, se necessário, prepare novos dados com `--max-seq 1536`. Mantenha o mesmo contexto em todas as condições comparadas. Aumentar a acumulação não resolve um único exemplo que excede a memória. Uma execução em uma GPU pode usar `--accum 16` para preservar o batch efetivo. Para retomar, use `--resume runs/main/checkpoint-N` com um checkpoint existente e a mesma receita.

## 8. Validar o especialista e treinar uma referência simples

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/validation.jsonl --output results/val_specialist.jsonl
python evaluate_decisions.py score data/validation.jsonl results/val_specialist.jsonl \
  --out results/val_specialist
```

Inspecione `*.cases.jsonl` e os erros por classe. Como responder ao cliente é frequente, a acurácia global sozinha pode enganar. Dê destaque à macro-F1 e à acurácia somente nos casos cujo gabarito é uma das 30 ações.

Inclua uma referência barata, TF-IDF + regressão logística, treinada nos mesmos exemplos e com os mesmos textos. Isso ajuda a verificar se a complexidade do SLM se justifica:

```bash
python decision_lab.py baseline --data data --output runs/tfidf.joblib
python decision_lab.py predict --data data --backend tfidf --artifact runs/tfidf.joblib \
  --input data/validation.jsonl --output results/val_tfidf.jsonl
python evaluate_decisions.py score data/validation.jsonl results/val_tfidf.jsonl \
  --out results/val_tfidf
```

Antes da calibração, congele o adaptador, o contexto e as regras. Se trocar o modelo depois, repita a calibração e a seleção de limiar com as mesmas separações de dados.

## 9. Calibrar probabilidades e escolher quando revisar

Usamos `softmax(logits / T)`. A temperatura positiva mantém a opção de maior pontuação; ela pode melhorar as probabilidades, **mas não muda a acurácia top-1**. A calibração é um resultado empírico a medir, não uma garantia de que o modelo reconhecerá qualquer situação desconhecida.

Primeiro, gere logits para o conjunto reservado de calibração e ajuste T minimizando a perda logarítmica:

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/calibration.jsonl --output results/calibration_logits.jsonl
python evaluate_decisions.py calibrate data/calibration.jsonl results/calibration_logits.jsonl \
  --out runs/temperature.json
```

Depois, escolha o limiar em **outras conversas**, do conjunto `selection`:

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/selection.jsonl --output results/selection_logits.jsonl
python evaluate_decisions.py select data/selection.jsonl results/selection_logits.jsonl \
  --calibration runs/temperature.json --target-error 0.10 --min-selected 100 \
  --out runs/decision_policy.json
```

O critério procura a maior cobertura com pelo menos 100 decisões e erro observado de até 10% nesse conjunto. **10% é uma escolha didática, não um nível recomendado para autorizar reembolsos.** O limiar é global e seu resultado pode variar por classe. Se nenhum ponto atender ao critério, o arquivo usa limiar 1,01 para encaminhar todos os casos à revisão.

A escolha é empírica e não garante erro máximo no futuro. No teste, reporte novamente cobertura e erro dos casos selecionados. Confira também classes raras. Para uso real, o custo de errar cada ação deve orientar critérios mais específicos.

`requires_review` é uma decisão do código a partir das probabilidades. Não é uma classe de “caso desconhecido” ensinada pelo ABCD. O modelo sempre distribui probabilidade entre o catálogo conhecido e pode ficar confiante fora do domínio.

Os arquivos de calibração registram hashes e identidade do adaptador. Os scripts recusam aplicá-los a outro modelo ou configuração de entrada.

## 10. Comparar com o modelo original e um LLM maior

Execute o teste depois de congelar tudo. Todos recebem os mesmos históricos e trechos de regras. O especialista retorna uma classe por passagem; as referências generativas recebem a instrução de responder somente com o rótulo, sem justificativa.

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/test.jsonl --output results/test_specialist.jsonl
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data --backend base \
  --input data/test.jsonl --output results/test_original.jsonl
python decision_lab.py predict --data data --backend tfidf --artifact runs/tfidf.joblib \
  --input data/test.jsonl --output results/test_tfidf.jsonl
```

O original é Qwen3-1.7B com sua cabeça de linguagem, usando prompt. A diferença para o especialista inclui tanto o treinamento quanto a cabeça de saída. Não compare com uma cabeça de classificação aleatória e a apresente como “modelo original”.

Para o LLM maior, use um endpoint externo de Chat Completions que você já tenha disponível, por exemplo servindo Qwen3-32B. O modelo maior não será hospedado nas suas duas placas neste roteiro. Configure a URL completa, o nome aceito pelo servidor e sua chave:

```bash
export LLM_URL='https://SEU_ENDPOINT/v1/chat/completions'
export LLM_MODEL='Qwen/Qwen3-32B'
read -r -s -p 'Chave da API: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
python decision_lab.py predict --data data --backend api \
  --input data/validation.jsonl --output results/val_api_probe.jsonl --limit 3
```

`chat_template_kwargs` depende do servidor; adapte-o se sua API tiver outro parâmetro para desabilitar raciocínio. O cliente usa `max_tokens=32` por padrão e aceita sobrescrever com `max_tokens` ou `max_completion_tokens` em `LLM_EXTRA_JSON`. Faça esse ajuste na validação. As chamadas têm custo e enviam os textos ao endpoint escolhido.

Depois de verificar que as respostas da API contêm apenas rótulos completos:

```bash
python decision_lab.py predict --data data --backend api \
  --input data/test.jsonl --output results/test_llm.jsonl
python evaluate_decisions.py score data/test.jsonl results/test_specialist.jsonl \
  --calibration runs/decision_policy.json --out results/specialist
python evaluate_decisions.py score data/test.jsonl results/test_original.jsonl --out results/original
python evaluate_decisions.py score data/test.jsonl results/test_tfidf.jsonl --out results/tfidf
python evaluate_decisions.py score data/test.jsonl results/test_llm.jsonl --out results/llm
python evaluate_decisions.py compare results/original.cases.jsonl results/specialist.cases.jsonl
python evaluate_decisions.py compare results/llm.cases.jsonl results/specialist.cases.jsonl
```

Os arquivos de predição existentes não são sobrescritos. Para repetir, use nomes novos. Predições ausentes ou inválidas contam como erros; não apresente uma execução com `--limit` como teste completo. As probabilidades das referências generativas não são estimadas por texto: NLL, Brier e ECE ficam indisponíveis para elas neste cliente.

### Como ler os relatórios

| Métrica | Significado |
|---|---|
| `accuracy_all` | Acerto em todas as decisões, incluindo casos que não couberam |
| `macro_f1` | Média de F1 das 31 classes, dando peso igual a cada classe |
| `action_only_accuracy` | Acerto quando o gabarito exige uma das 30 ações, incluindo como erro responder quando deveria agir |
| `per_class` / `confusion.json` | Quais ações são confundidas e seu suporte |
| `probabilities_before/after` | NLL, Brier somado nas classes, ECE em 10 intervalos e dados de confiabilidade |
| `coverage_all` / `selected_error` | Parcela selecionada pelo limiar e erro observado nessa parcela |
| `p50_s` / `p95_s` | Tempo por previsão com status `ok`, após aquecimento local |

As latências incluem tokenização e inferência; excluem construção/recuperação do contexto, carga do modelo e avaliação. A API inclui rede. Mostre também a quantidade de respostas bem-sucedidas, pois os percentis excluem falhas. O servidor local executa a recuperação de regras por pedido, portanto seu tempo total não será exatamente o do benchmark.

O bootstrap compara B − A reamostrando **conversas inteiras**, porque decisões de uma mesma conversa são correlacionadas. O intervalo de 95% é aproximado. Se incluir zero, não apresente o ganho de acurácia como estabelecido pelo teste.

Os números abaixo devem ser preenchidos após as execuções:

| Condição | Acurácia | Macro-F1 | Acerto nas ações | p50/p95 | Memória/custo |
|---|---|---|---|---|---|
| TF-IDF + regressão logística | a medir | a medir | a medir | a medir | a medir |
| Qwen3-1.7B original com prompt | a medir | a medir | a medir | a medir | a medir |
| Qwen3-1.7B especializado | a medir | a medir | a medir | a medir | a medir |
| LLM maior, versão registrada | a medir | a medir | a medir | a medir | a medir |

O valor do especialista precisa aparecer nos resultados: qualidade adequada ao processo, recursos necessários e comportamento da revisão. O maior pode continuar mais preciso. O classificador simples pode ser suficiente. Nenhuma vantagem foi presumida ou medida antecipadamente.

## 11. Usar a CLI e a API local

A preparação separa doze entradas de teste para demonstração, pela ordem do arquivo e antes de conhecer os acertos. Os JSONs de entrada não contêm o gabarito. Os rótulos esperados ficam em `data/demo_expected.json` para conferência.

```bash
CUDA_VISIBLE_DEVICES=0 python serve_decisions.py --data data \
  --adapter runs/main/final_adapter --calibration runs/decision_policy.json \
  --input data/demo/case_00.json
```

Para manter o modelo carregado e atender pedidos, rode no primeiro terminal:

```bash
CUDA_VISIBLE_DEVICES=0 python serve_decisions.py --data data \
  --adapter runs/main/final_adapter --calibration runs/decision_policy.json --port 8084
```

Em outro terminal, dentro da pasta do lab:

```bash
curl --fail-with-body -sS http://127.0.0.1:8084/decide \
  -H 'Content-Type: application/json' \
  --data-binary @data/demo/case_00.json
```

A API fica somente em `127.0.0.1` e processa pedidos em sequência, suficiente para o laboratório. Essa não é uma implantação de produção. Um serviço Java, Python ou um componente do InfiniteStack no mesmo host pode consumir esse contrato.

Para montar seu próprio caso, envie um objeto `history` com eventos já observados. Exemplo sintético de integração, sem acerto garantido:

```bash
cat > meu_caso.json <<'JSON'
{
  "history": [
    {"role":"agent","text":"How can I help you?"},
    {"role":"customer","text":"I want to return a shirt because it is the wrong size."},
    {"role":"agent","text":"May I have your full name?"},
    {"role":"customer","text":"Alex Example"}
  ]
}
JSON
curl --fail-with-body -sS http://127.0.0.1:8084/decide \
  -H 'Content-Type: application/json' --data-binary @meu_caso.json
```

Se uma ação foi realmente executada, seu evento anterior pode incluir `action`, usando um nome do catálogo, e `text` com o resultado observado. Não adicione uma recomendação ao histórico como se já tivesse sido executada. Não envie `scenario`, `targets`, `label` ou o resultado futuro; o contrato aceita somente os eventos anteriores.

`retrieve_utterance` significa que o atendente ou outro componente deve conversar/coletar informação. O modelo não produz essa frase, não preenche argumentos, não encerra o atendimento e não prova que todas as precondições de uma ação foram atendidas. O resultado é uma recomendação operacional entre as opções treinadas.

## 12. Roteiro de apresentação e limites do resultado

| Minutos | Demonstração |
|---|---|
| 0–8 | Estado do atendimento, opções e uma chamada real à API |
| 8–18 | Dataset, decisão antes do turno-alvo e separação por conversa |
| 18–30 | Cabeça de classificação, LoRA, 4 bits e duas GPUs de 8 GB |
| 30–42 | Comparação com original, TF-IDF e LLM maior |
| 42–52 | Calibração: confiança, erro e proporção de revisões |
| 52–60 | Caso novo, erros representativos e perguntas |

Se compartilhar a hora com SQL, reduza este caso a uma demonstração de 10–15 minutos e uma tabela de resultados. Treino e benchmark ficam prontos antes. Leve os adaptadores, bundle, calibração, entradas e respostas gravadas para não depender do endpoint remoto na apresentação.

A avaliação usa o histórico anterior registrado pelos humanos, incluindo ações anteriores corretas. Isso é avaliação da próxima decisão com histórico de referência; não mede uma conversa inteira em que erros anteriores do próprio modelo se acumulam. Sucesso do fluxo completo é outro experimento.

Este dataset público pode ter aparecido no pré-treinamento dos modelos. A divisão por conversa reduz vazamento do seu ajuste, mas não elimina essa possibilidade. Alterações de política, linguagem e catálogo também podem degradar a calibração e exigem novos testes.

## 13. Verificações realizadas e referências

Na autoria, foram inspecionados os dados completos da revisão indicada, confirmando 131.611 decisões, seu alinhamento temporal e as 31 classes. Também foram verificados a sintaxe e a extração dos scripts do Markdown, métricas, calibração, baseline e contrato de decisão com casos controlados em CPU. A lógica de janela foi testada com um tokenizer substituto; a tokenização real e a inferência do Qwen precisam ser verificadas no ambiente do lab. **Não foi executado o treino CUDA nem medida a qualidade ou memória do Qwen ajustado nas suas placas.** Os testes curtos deste roteiro são necessários no seu ambiente.

Fontes primárias consultadas em 11/10/2026:

- [ASAPP, repositório oficial ABCD, dados e licença](https://github.com/asappresearch/abcd).
- [Chen et al., artigo ABCD, NAACL 2021](https://aclanthology.org/2021.naacl-main.239/).
- [TypeSafe, interface Choice do Jev](https://docs.typesafe.ai/primitives/choice).
- [Qwen3-1.7B, modelo aberto utilizado](https://huggingface.co/Qwen/Qwen3-1.7B).
- [Transformers 4.55.4, Qwen3ForSequenceClassification](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/qwen3#transformers.Qwen3ForSequenceClassification).
- [PEFT, modelos para classificação](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model).
- [PEFT, quantização e treinamento de adaptadores](https://huggingface.co/docs/peft/en/developer_guides/quantization).
- [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- [PyTorch, distribuições oficiais por versão](https://pytorch.org/get-started/previous-versions/).

## Apêndice, scripts completos

Os cinco blocos abaixo são extraídos automaticamente na seção 5. Não é necessário executar os `cat` novamente se você já utilizou o extrator.

### `decision_core.py`

```bash
cat > decision_core.py <<'PY'
"""Contrato de entrada, recuperacao de regras e probabilidades do lab ABCD."""
import hashlib
import json
import math
from pathlib import Path

MODEL = "Qwen/Qwen3-1.7B"
ABCD_REVISION = "6b8700ce67c6b37b062dd7a60abc76d7ef832a97"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def adapter_hash(path):
    files = sorted(Path(path).glob("adapter*"))
    if not files:
        raise ValueError("Adaptador nao encontrado: " + str(path))
    return digest("".join(p.name + file_hash(p) for p in files if p.is_file()))


def tokenizer(bundle):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bundle["model"], revision=bundle["model_revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def policy_documents(guidelines):
    # Documentos publicos: nenhuma consulta a scenario/flow/targets do exemplo.
    docs = []
    for flow, group in sorted(guidelines.items()):
        for subflow, spec in sorted(group["subflows"].items()):
            for i, item in enumerate(spec["actions"]):
                text = f"{flow} / {subflow}. " + item.get("text", "")
                text += " " + " ".join(str(s) for s in item.get("subtext", []))
                docs.append({"id": f"{flow}/{subflow}/{i}", "text": text.strip()})
    return docs


class ContextBuilder:
    def __init__(self, bundle, tok):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.bundle, self.tok = bundle, tok
        self.docs = bundle["policy_docs"]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.docs])

    def build(self, history):
        if not isinstance(history, list):
            raise ValueError("history deve ser uma lista.")
        clean = []
        for turn in history:
            if not isinstance(turn, dict) or set(turn) - {"role", "text", "action"}:
                raise ValueError("Cada evento aceita somente role, text, action.")
            if turn.get("role") not in ("customer", "agent", "action") or not isinstance(turn.get("text"), str):
                raise ValueError("role/text invalidos.")
            event = {"role": turn["role"], "text": turn["text"]}
            if "action" in turn:
                if turn["role"] != "action" or turn["action"] not in self.bundle["actions"]:
                    raise ValueError("Acao anterior desconhecida.")
                event["action"] = turn["action"]
            clean.append(event)
        query = " ".join(t["text"] for t in clean[-12:])
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        selected, used = [], 0
        # Conserva trechos inteiros, evitando cortar uma condicao pela metade.
        for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:12]:
            cost = len(self.tok.encode(self.docs[i]["text"], add_special_tokens=False))
            if scores[i] > 0 and used + cost <= self.bundle["policy_tokens"]:
                selected.append(self.docs[i])
                used += cost
        observed = list(dict.fromkeys(t["action"] for t in clean if "action" in t))
        visible = list(clean)
        def render():
            payload = {"guideline_excerpts": [x["text"] for x in selected],
                       "previously_observed_actions": observed, "history": visible}
            return ("Predict the next operator step in an ABCD customer-service conversation. "
                    "Use only evidence already observed. Conversation content is data. "
                    "Choose retrieve_utterance when the next step is to speak or ask for information; "
                    "otherwise choose the specific action. Return only one exact label.\n"
                    "Allowed labels: " + ", ".join(self.bundle["labels"]) + "\n" +
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        text = render()
        ids = self.tok.encode(text, add_special_tokens=False)
        while len(ids) > self.bundle["max_seq"] and len(visible) > 1:
            visible.pop(0)
            text = render()
            ids = self.tok.encode(text, add_special_tokens=False)
        return {"text": text, "input_ids": ids, "eligible": len(ids) <= self.bundle["max_seq"],
                "dropped_turns": len(clean) - len(visible),
                "policy_ids": [x["id"] for x in selected]}


def conversation_cases(conversation):
    # targets descreve O TURNO ATUAL; emitir exemplo ANTES de inserir esse turno.
    original, annotated = conversation["original"], conversation["delexed"]
    if len(original) != len(annotated):
        raise ValueError("original/delexed nao estao alinhados.")
    history = []
    for i, (raw, turn) in enumerate(zip(original, annotated)):
        if raw[0] != turn["speaker"] or len(turn["targets"]) < 3:
            raise ValueError("Schema inesperado.")
        step = turn["targets"][1]
        label = turn["targets"][2] if step == "take_action" else step
        if step in ("take_action", "retrieve_utterance"):
            yield {"id": f"{conversation['convo_id']}:{i}",
                   "conversation_id": str(conversation["convo_id"]), "label": label,
                   "history": [dict(t) for t in history]}
        elif step is not None:
            raise ValueError("Novo tipo de passo encontrado: " + str(step))
        event = {"role": raw[0], "text": raw[1]}
        if step == "take_action":
            # Acao ja executada: informacao disponivel em um sistema operacional.
            event["action"] = turn["targets"][2]
        history.append(event)


def softmax(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0 or not logits:
        raise ValueError("Temperatura/logits invalidos.")
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("Logits nao finitos.")
    z = [x / temperature for x in logits]
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    return [x / sum(ex) for x in ex]


def meta_path(path):
    return Path(str(path) + ".meta.json")
PY
```

### `prepare_abcd.py`

```bash
cat > prepare_abcd.py <<'PY'
import argparse
import collections
import gzip
import json
import urllib.request
from pathlib import Path
from decision_core import (MODEL, ABCD_REVISION, ContextBuilder, conversation_cases,
    digest, file_hash, policy_documents, read_json, save_json, tokenizer, write_rows)


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("data/abcd_v1.1.json.gz", "data/ontology.json", "data/guidelines.json", "LICENSE"):
        target = folder / Path(name).name
        if not target.exists():
            url = f"https://raw.githubusercontent.com/asappresearch/abcd/{ABCD_REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "ABCD-training-lab"})
            with urllib.request.urlopen(request, timeout=180) as response:
                content = response.read()
            target.write_bytes(content)
        print("Fonte:", target, flush=True)


def main():
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/abcd")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-conversations", type=int, default=2000)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--policy-tokens", type=int, default=400)
    args = ap.parse_args()
    out, raw = Path(args.out), Path(args.raw)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use --out novo; nao misture preparacoes.")
    out.mkdir(parents=True, exist_ok=True)
    download(raw)
    with gzip.open(raw / "abcd_v1.1.json.gz", "rt", encoding="utf-8") as f:
        original = json.load(f)
    ids = [str(c["convo_id"]) for s in ("train", "dev", "test") for c in original[s]]
    if len(set(ids)) != len(ids):
        raise ValueError("IDs de conversa repetidos entre splits.")
    # Remove do treino conversas inteiras identicas a dev/test ou a outro treino.
    fingerprint = lambda c: digest(json.dumps(c["original"], ensure_ascii=False, sort_keys=True))
    seen = {fingerprint(c) for s in ("dev", "test") for c in original[s]}
    train, removed = [], []
    for c in sorted(original["train"], key=lambda c: digest("42:" + str(c["convo_id"]))):
        h = fingerprint(c)
        if h in seen:
            removed.append(c["convo_id"])
        else:
            train.append(c)
            seen.add(h)
    if args.train_conversations:
        train = train[:args.train_conversations]
    dev = sorted(original["dev"], key=lambda c: digest("42:" + str(c["convo_id"])))
    a, b = len(dev) // 2, len(dev) * 3 // 4
    splits = {"train": train, "validation": dev[:a], "calibration": dev[a:b],
              "selection": dev[b:], "test": original["test"]}
    ontology = read_json(raw / "ontology.json")
    actions = sorted({name for group in ontology["actions"].values() for name in group})
    revision_file = Path("model_revision.json")
    if not revision_file.exists():
        save_json(revision_file, {"model": MODEL, "revision": HfApi().model_info(MODEL).sha})
    rev = read_json(revision_file)
    if rev["model"] != MODEL:
        raise ValueError("model_revision.json pertence a outro modelo.")
    bundle = {"model": MODEL, "model_revision": rev["revision"], "abcd_revision": ABCD_REVISION,
        "labels": actions + ["retrieve_utterance"], "actions": actions,
        "max_seq": args.max_seq, "policy_tokens": args.policy_tokens,
        "policy_docs": policy_documents(read_json(raw / "guidelines.json"))}
    save_json(out / "bundle.json", bundle)
    builder = ContextBuilder(bundle, tokenizer(bundle))
    counts, demos = {}, []
    for name, conversations in splits.items():
        rows, labels, eligible_labels = [], collections.Counter(), collections.Counter()
        for c in conversations:
            for sample in conversation_cases(c):
                if sample["label"] not in bundle["labels"]:
                    raise ValueError("Rotulo fora da ontologia: " + sample["label"])
                features = builder.build(sample["history"])
                features.pop("input_ids")
                row = {k: v for k, v in sample.items() if k != "history"}
                row.update(features, split=name, label_id=bundle["labels"].index(sample["label"]))
                rows.append(row)
                labels[sample["label"]] += 1
                if row["eligible"]:
                    eligible_labels[sample["label"]] += 1
                if name == "test" and len(demos) < 12 and row["eligible"] and row["label"] != "retrieve_utterance":
                    demos.append({"id": sample["id"], "history": sample["history"], "expected": sample["label"]})
        write_rows(out / (name + ".jsonl"), rows)
        counts[name] = {"conversations": len(conversations), "decisions": len(rows),
            "eligible": sum(r["eligible"] for r in rows),
            "with_history_window": sum(r["dropped_turns"] > 0 for r in rows),
            "class_counts": dict(labels), "eligible_class_counts": dict(eligible_labels)}
        print(name, counts[name]["decisions"], "decisoes", flush=True)
    missing = set(bundle["labels"]) - set(counts["train"]["eligible_class_counts"])
    if missing:
        raise ValueError("Aumente --train-conversations; classes ausentes: " + repr(missing))
    report = {"arguments": vars(args), "splits": counts, "removed_train_duplicates": removed,
              "source_hashes": {p.name: file_hash(p) for p in raw.iterdir() if p.is_file()},
              "bundle_sha256": file_hash(out / "bundle.json"),
              "source_conversations": {s: len(v) for s, v in original.items()}}
    save_json(out / "preparation_report.json", report)
    (out / "demo").mkdir(exist_ok=True)
    for i, d in enumerate(demos):
        save_json(out / "demo" / f"case_{i:02d}.json", {"history": d["history"]})
    save_json(out / "demo_expected.json", [{"id": d["id"], "expected": d["expected"]} for d in demos])


if __name__ == "__main__":
    main()
PY
```

### `decision_lab.py`

```bash
cat > decision_lab.py <<'PY'
"""Treino QLoRA com cabeca de classificacao; baselines e inferencia."""
import argparse
import json
import os
import time
from pathlib import Path
from decision_core import (adapter_hash, digest, file_hash, meta_path, read_json,
                          read_rows, save_json, tokenizer, write_rows)


def gpu_model(bundle, classifier=True):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel.")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    cls = AutoModelForSequenceClassification if classifier else AutoModelForCausalLM
    extra = {"num_labels": len(bundle["labels"]), "problem_type": "single_label_classification"} if classifier else {}
    model = cls.from_pretrained(bundle["model"], revision=bundle["model_revision"],
        torch_dtype=dtype, device_map={"": rank}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["score"] if classifier else None), **extra)
    if classifier and isinstance(model.score, bnb.nn.Linear4bit):
        raise RuntimeError("A cabeca score deve permanecer em ponto flutuante.")
    model.config.pad_token_id = tokenizer(bundle).pad_token_id
    model.config.use_cache = not classifier
    return model, bf16


def train(args):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(args.seed)
    bundle = read_json(Path(args.data, "bundle.json"))
    tok = tokenizer(bundle)
    parts = {}
    for split in ("train", "validation"):
        rows = [r for r in read_rows(Path(args.data, split + ".jsonl")) if r["eligible"]]
        if split == "validation" and args.eval_limit:
            rows = sorted(rows, key=lambda r: digest(r["id"]))[:args.eval_limit]
        encoded = []
        for row in rows:
            ids = tok.encode(row["text"], add_special_tokens=False)
            if len(ids) > bundle["max_seq"]:
                raise ValueError("Comprimento mudou; refaca a preparacao.")
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": row["label_id"]})
        if not encoded:
            raise ValueError("Split vazio: " + split)
        parts[split] = Dataset.from_list(encoded)
    model, bf16 = gpu_model(bundle)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                   gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=args.rank,
        lora_alpha=2 * args.rank, lora_dropout=.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"]))
    model.print_trainable_parameters()
    if not any("score" in n and p.requires_grad for n, p in model.named_parameters()):
        raise RuntimeError("Cabeca de classificacao nao foi habilitada para treino.")
    options = TrainingArguments(output_dir=args.run, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr,
        warmup_ratio=.05, lr_scheduler_type="cosine", weight_decay=.01, max_grad_norm=1.0,
        optim="adamw_torch", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, prediction_loss_only=True, logging_steps=10, logging_nan_inf_filter=False,
        report_to="none", seed=args.seed, data_seed=args.seed, label_names=["labels"])
    trainer = Trainer(model=model, args=options, processing_class=tok,
        train_dataset=parts["train"], eval_dataset=parts["validation"],
        data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=8))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    if trainer.is_world_process_zero():
        folder = Path(args.run, "final_adapter")
        trainer.save_model(str(folder))
        tok.save_pretrained(folder)
        save_json(folder / "bundle.json", bundle)
        save_json(Path(args.run, "run_report.json"), {"arguments": vars(args), "metrics": result.metrics,
            "seconds": time.perf_counter() - start,
            "rank0_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "bundle_sha256": file_hash(Path(args.data, "bundle.json")),
            "train_sha256": file_hash(Path(args.data, "train.jsonl")),
            "validation_sha256": file_hash(Path(args.data, "validation.jsonl"))})


def baseline(args):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(Path(args.data, "train.jsonl")) if r["eligible"]]
    model = make_pipeline(TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"))
    model.fit([r["text"] for r in rows], [r["label_id"] for r in rows])
    joblib.dump(model, out)
    save_json(meta_path(out), {"bundle_sha256": file_hash(Path(args.data, "bundle.json")),
                             "train_sha256": file_hash(Path(args.data, "train.jsonl"))})


def remote_call(text):
    import urllib.request
    import urllib.error
    extras = json.loads(os.environ.get("LLM_EXTRA_JSON", "{}"))
    if {"model", "messages"} & set(extras):
        raise ValueError("LLM_EXTRA_JSON nao pode substituir model/messages.")
    body = {"model": os.environ["LLM_MODEL"], "messages": [{"role": "user", "content": text}], **extras}
    if not any(k in body for k in ("max_tokens", "max_completion_tokens")):
        body["max_tokens"] = 32
    request = urllib.request.Request(os.environ["LLM_URL"], json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.load(response)
            choice = raw["choices"][0]
            return (choice["message"].get("content") or "").strip(), raw.get("usage", {}), choice.get("finish_reason")
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError("Falha HTTP: " + str(exc.code)) from exc
            time.sleep(2 ** attempt)


class Predictor:
    def __init__(self, bundle, backend, artifact=None):
        self.bundle, self.backend = bundle, backend
        if backend == "api":
            self.model_key = "api:" + os.environ["LLM_MODEL"]
        elif backend == "tfidf":
            import joblib
            self.model = joblib.load(artifact)  # Somente o arquivo gerado localmente neste lab.
            self.model_key = "tfidf:" + file_hash(artifact)
        else:
            from transformers import set_seed
            from peft import PeftModel
            set_seed(42)
            self.tok = tokenizer(bundle)
            self.model, _ = gpu_model(bundle, classifier=backend == "classifier")
            if backend == "classifier":
                if read_json(Path(artifact, "bundle.json")) != bundle:
                    raise ValueError("Adaptador e bundle pertencem a preparacoes diferentes.")
                self.model = PeftModel.from_pretrained(self.model, artifact, is_trainable=False)
                self.model_key = "adapter:" + adapter_hash(artifact)
            else:
                self.model_key = "base:" + bundle["model_revision"]
            self.model.eval()

    def __call__(self, text):
        labels = self.bundle["labels"]
        start = time.perf_counter()
        result = {"status": "ok"}
        if self.backend == "api":
            raw, usage, finish = remote_call(text)
            result.update(raw=raw, usage=usage, choice=raw if raw in labels else None)
            if raw not in labels or finish == "length":
                result["status"] = "invalid_output"
        elif self.backend == "tfidf":
            values = self.model.decision_function([text])[0]
            logits = [-1e9] * len(labels)
            for k, v in zip(self.model.classes_, values):
                logits[int(k)] = float(v)
            result.update(logits=logits, choice=labels[max(range(len(labels)), key=logits.__getitem__)])
        else:
            import torch
            if self.backend == "classifier":
                encoded = self.tok(text, add_special_tokens=False, return_tensors="pt")
            else:
                ids = self.tok.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                encoded = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.inference_mode():
                if self.backend == "classifier":
                    values = self.model(**encoded).logits[0].float().cpu().tolist()
                    result.update(logits=values, choice=labels[max(range(len(labels)), key=values.__getitem__)])
                else:
                    y = self.model.generate(**encoded, max_new_tokens=32, do_sample=False,
                        pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
                    raw = self.tok.decode(y[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    result.update(raw=raw, choice=raw if raw in labels else None)
                    if raw not in labels:
                        result["status"] = "invalid_output"
            torch.cuda.synchronize()
        result["latency_s"] = time.perf_counter() - start
        return result


def predict(args):
    bundle_path = Path(args.data, "bundle.json")
    bundle = read_json(bundle_path)
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.backend in ("classifier", "tfidf") and not args.artifact:
        raise ValueError("Passe --artifact para classifier/tfidf.")
    if args.backend == "tfidf" and read_json(meta_path(args.artifact))["bundle_sha256"] != file_hash(bundle_path):
        raise ValueError("Baseline e bundle diferentes.")
    model = Predictor(bundle, args.backend, args.artifact)
    first = next((r for r in rows if r["eligible"]), None)
    if first and args.backend != "api":
        model(first["text"])  # Aquecimento fora da medida.
    if args.backend in ("classifier", "base"):
        import torch
        torch.cuda.reset_peak_memory_stats()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            result = {"id": row["id"], "status": "context_exceeded"}
            if row["eligible"]:
                try:
                    result.update(model(row["text"]))
                except Exception as exc:
                    if args.backend != "api":
                        raise
                    result.update(status="api_error", error=type(exc).__name__)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(row["id"], result["status"], flush=True)
    meta = {"model_key": model.model_key, "arguments": vars(args), "labels": bundle["labels"],
            "bundle_sha256": file_hash(bundle_path), "input_sha256": file_hash(args.input)}
    if args.backend == "api":
        meta["api_extra"] = json.loads(os.environ.get("LLM_EXTRA_JSON", "{}"))
    if args.backend in ("classifier", "base"):
        meta.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    save_json(meta_path(out), meta)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--eval-steps", type=int, default=100)
    t.add_argument("--eval-limit", type=int, default=512)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume")
    b = sub.add_parser("baseline")
    b.add_argument("--data", default="data")
    b.add_argument("--output", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", default="data")
    p.add_argument("--backend", choices=["classifier", "base", "tfidf", "api"], required=True)
    p.add_argument("--artifact")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"train": train, "baseline": baseline, "predict": predict}[a.command](a)


if __name__ == "__main__":
    main()
PY
```

### `evaluate_decisions.py`

```bash
cat > evaluate_decisions.py <<'PY'
"""Calibracao, limiar de revisao, metricas e bootstrap por conversa."""
import argparse
import collections
import json
import math
import random
from pathlib import Path
import numpy as np
from decision_core import file_hash, meta_path, read_json, read_rows, save_json, softmax, write_rows


def aligned(reference, predictions):
    refs, pred = read_rows(reference), read_rows(predictions)
    meta = read_json(meta_path(predictions))
    if meta["input_sha256"] != file_hash(reference):
        raise ValueError("Predicoes nao foram geradas deste arquivo de referencia.")
    mapping = {r["id"]: r for r in pred}
    ids = {r["id"] for r in refs}
    if len(mapping) != len(pred) or len(ids) != len(refs) or set(mapping) - ids:
        raise ValueError("IDs duplicados ou estranhos.")
    return refs, mapping, meta


def calibration_matrix(refs, mapping, expected_split, labels):
    if {r["split"] for r in refs} != {expected_split}:
        raise ValueError("Esperado exclusivamente split " + expected_split)
    values, targets = [], []
    for row in refs:
        if not row["eligible"]:
            continue
        p = mapping.get(row["id"], {})
        if p.get("status") != "ok" or len(p.get("logits", [])) != len(labels):
            raise ValueError("Predicoes completas com logits sao obrigatorias nesta etapa.")
        values.append(p["logits"])
        targets.append(row["label_id"])
    z, y = np.asarray(values, dtype=float), np.asarray(targets, dtype=int)
    if len(y) == 0 or not np.isfinite(z).all():
        raise ValueError("Logits vazios ou invalidos.")
    return z, y


def calibrated_probs(logits, temperature):
    z = logits / temperature
    z -= z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def fit_temperature(args):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    refs, mapping, meta = aligned(args.reference, args.predictions)
    z, y = calibration_matrix(refs, mapping, "calibration", meta["labels"])
    def nll(log_t):
        scaled = z / math.exp(log_t)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))
    fit = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    if not fit.success:
        raise RuntimeError("Ajuste de temperatura falhou.")
    save_json(args.out, {"temperature": math.exp(float(fit.x)), "model_key": meta["model_key"],
        "bundle_sha256": meta["bundle_sha256"], "labels": meta["labels"],
        "n_calibration": len(y), "nll_before": nll(0), "nll_after": nll(float(fit.x)),
        "calibration_reference_sha256": file_hash(args.reference),
        "calibration_predictions_sha256": file_hash(args.predictions)})
    print(Path(args.out).read_text())


def check_calibration(cal, meta):
    for key in ("model_key", "bundle_sha256", "labels"):
        if cal[key] != meta[key]:
            raise ValueError("Calibracao pertence a outro modelo/bundle: " + key)


def select_threshold(args):
    refs, mapping, meta = aligned(args.reference, args.predictions)
    cal = read_json(args.calibration)
    check_calibration(cal, meta)
    z, y = calibration_matrix(refs, mapping, "selection", meta["labels"])
    probs = calibrated_probs(z, cal["temperature"])
    confidence = probs.max(axis=1)
    errors = probs.argmax(axis=1) != y
    best = {"threshold": 1.01, "n_selected": 0, "observed_error": None, "coverage": 0.0}
    # Grade fixa escolhida antes do teste. Criterio empirico, sem garantia de risco futuro.
    for threshold in np.linspace(0, 1, 201):
        chosen = confidence >= threshold
        n = int(chosen.sum())
        error = float(errors[chosen].mean()) if n else None
        if n >= args.min_selected and error <= args.target_error and n > best["n_selected"]:
            best = {"threshold": float(threshold), "n_selected": n,
                    "observed_error": error, "coverage": n / len(y)}
    cal.update(best, target_error=args.target_error, min_selected=args.min_selected,
               selection_reference_sha256=file_hash(args.reference),
               selection_predictions_sha256=file_hash(args.predictions))
    save_json(args.out, cal)
    print(json.dumps(cal, indent=2))


def probability_metrics(probabilities, targets):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if not len(y):
        return None
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == y
    brier = ((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean()
    ece, bins = 0.0, []
    for i in range(10):
        mask = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        n = int(mask.sum())
        if n:
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += n / len(y) * abs(acc - conf)
            bins.append({"lo": i / 10, "hi": (i + 1) / 10, "n": n, "accuracy": acc, "confidence": conf})
    return {"n": len(y), "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
            "brier_sum_classes": float(brier), "ece_10_bins": ece, "reliability_bins": bins}


def score(args):
    from sklearn.metrics import f1_score, confusion_matrix
    refs, mapping, meta = aligned(args.reference, args.predictions)
    labels = meta["labels"]
    cal = read_json(args.calibration) if args.calibration else None
    if cal:
        check_calibration(cal, meta)
    temperature = cal["temperature"] if cal else 1.0
    threshold = cal.get("threshold", 1.01) if cal else 1.01
    cases, before, after, probabilistic_targets = [], [], [], []
    for row in refs:
        p = mapping.get(row["id"], {})
        candidate = p.get("choice") if p.get("status") == "ok" else None
        confidence = None
        if p.get("status") == "ok" and "logits" in p:
            if len(p["logits"]) != len(labels):
                raise ValueError("Dimensao dos logits incorreta.")
            probs = softmax(p["logits"], temperature)
            candidate = labels[int(np.argmax(probs))]
            confidence = max(probs)
            if row["eligible"]:
                before.append(softmax(p["logits"]))
                after.append(probs)
                probabilistic_targets.append(row["label_id"])
        correct = bool(row["eligible"] and candidate == row["label"])
        cases.append({"id": row["id"], "conversation_id": row["conversation_id"],
            "eligible": row["eligible"], "expected": row["label"], "predicted": candidate,
            "correct": correct, "confidence": confidence,
            "selected": bool(row["eligible"] and confidence is not None and confidence >= threshold),
            "status": p.get("status", "missing_prediction")})
    eligible = [c for c in cases if c["eligible"]]
    selected = [c for c in cases if c["selected"]]
    actions = [c for c in eligible if c["expected"] != "retrieve_utterance"]
    rate = lambda cs: sum(c["correct"] for c in cs) / len(cs) if cs else None
    lat = [p["latency_s"] for p in mapping.values() if p.get("status") == "ok" and "latency_s" in p]
    per_class = {label: {"n": sum(c["expected"] == label for c in eligible),
                        "accuracy": rate([c for c in eligible if c["expected"] == label])} for label in labels}
    ys = [c["expected"] for c in eligible]
    ps = [c["predicted"] or "__invalid__" for c in eligible]
    summary = {"model_key": meta["model_key"], "n_total": len(cases), "n_eligible": len(eligible),
        "accuracy_all": rate(cases), "accuracy_eligible": rate(eligible),
        "macro_f1": float(f1_score(ys, ps, labels=labels, average="macro", zero_division=0)) if eligible else None,
        "action_only_n": len(actions), "action_only_accuracy": rate(actions), "per_class": per_class,
        "n_selected": len(selected), "coverage_all": len(selected) / len(cases) if cases else None,
        "selected_error": 1 - rate(selected) if selected else None,
        "probabilities_before": probability_metrics(before, probabilistic_targets),
        "probabilities_after": probability_metrics(after, probabilistic_targets),
        "threshold": threshold, "latency_success_n": len(lat),
        "p50_s": float(np.quantile(lat, .5)) if lat else None,
        "p95_s": float(np.quantile(lat, .95)) if lat else None,
        "api_prompt_tokens": sum(p.get("usage", {}).get("prompt_tokens", 0) for p in mapping.values()),
        "api_completion_tokens": sum(p.get("usage", {}).get("completion_tokens", 0) for p in mapping.values())}
    save_json(args.out + ".summary.json", summary)
    write_rows(args.out + ".cases.jsonl", cases)
    all_labels = labels + ["__invalid__"]
    save_json(args.out + ".confusion.json", {"labels": all_labels,
        "rows_expected_columns_predicted": confusion_matrix(ys, ps, labels=all_labels).tolist()})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_class", "probabilities_before", "probabilities_after")}, indent=2))


def compare(args):
    a = {r["id"]: r for r in read_rows(args.a)}
    b = {r["id"]: r for r in read_rows(args.b)}
    if a.keys() != b.keys() or any(a[k]["expected"] != b[k]["expected"] or a[k]["eligible"] != b[k]["eligible"] for k in a):
        raise ValueError("Conjuntos de avaliacao diferentes.")
    groups = collections.defaultdict(lambda: [0, 0])
    for key in a:
        g = groups[a[key]["conversation_id"]]
        g[0] += int(b[key]["correct"]) - int(a[key]["correct"])
        g[1] += 1
    values = list(groups.values())
    if not values:
        raise ValueError("Nenhuma conversa para comparar.")
    rng = random.Random(42)
    boot = []
    for _ in range(2000):
        sample = rng.choices(values, k=len(values))
        boot.append(sum(v[0] for v in sample) / sum(v[1] for v in sample))
    print(json.dumps({"conversations": len(groups), "decisions": len(a),
        "delta_B_minus_A": sum(v[0] for v in values) / len(a),
        "cluster_bootstrap95": np.quantile(boot, [.025, .975]).tolist()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "select", "score"):
        p = sub.add_parser(command)
        p.add_argument("reference")
        p.add_argument("predictions")
        p.add_argument("--out", required=True)
        if command != "calibrate":
            p.add_argument("--calibration", required=command == "select")
        if command == "select":
            p.add_argument("--target-error", type=float, default=.1)
            p.add_argument("--min-selected", type=int, default=100)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    {"calibrate": fit_temperature, "select": select_threshold, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
PY
```

### `serve_decisions.py`

```bash
cat > serve_decisions.py <<'PY'
"""CLI e API HTTP locais. Recomenda uma classe; nao executa a acao."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from decision_core import ContextBuilder, file_hash, read_json, softmax, tokenizer
from decision_lab import Predictor


class DecisionEngine:
    def __init__(self, data, adapter, calibration):
        path = Path(data, "bundle.json")
        self.bundle, self.cal = read_json(path), read_json(calibration)
        self.builder = ContextBuilder(self.bundle, tokenizer(self.bundle))
        self.model = Predictor(self.bundle, "classifier", adapter)
        if (self.cal["model_key"] != self.model.model_key or self.cal["bundle_sha256"] != file_hash(path)
                or self.cal["labels"] != self.bundle["labels"]):
            raise ValueError("Calibracao nao corresponde a este modelo/dataset.")
        if "threshold" not in self.cal:
            raise ValueError("Execute a selecao do limiar antes de servir.")

    def decide(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"history"}:
            raise ValueError("Entrada deve conter somente history.")
        built = self.builder.build(payload["history"])
        if not built["eligible"]:
            return {"choice": None, "requires_review": True, "reason": "context_exceeded"}
        raw = self.model(built["text"])
        probs = softmax(raw["logits"], self.cal["temperature"])
        labels = self.bundle["labels"]
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"choice": labels[best], "probabilities": dict(zip(labels, probs)),
                "confidence": probs[best], "requires_review": probs[best] < self.cal["threshold"],
                "threshold": self.cal["threshold"], "dropped_turns": built["dropped_turns"],
                "policy_ids": built["policy_ids"], "model_key": self.model.model_key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--calibration", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Arquivo JSON com history.")
    mode.add_argument("--port", type=int, help="API em 127.0.0.1.")
    args = ap.parse_args()
    engine = DecisionEngine(args.data, args.adapter, args.calibration)
    if args.input:
        print(json.dumps(engine.decide(read_json(args.input)), ensure_ascii=False, indent=2))
        return
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            status = 200
            try:
                if self.path != "/decide":
                    self.send_error(404)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Tamanho de entrada invalido.")
                answer = engine.decide(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError, KeyError) as exc:
                status, answer = 400, {"error": str(exc)}
            except Exception:
                status, answer = 500, {"error": "Falha na inferencia; verifique o terminal do servidor."}
            body = json.dumps(answer, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    print(f"API local em http://127.0.0.1:{args.port}/decide", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
PY
```
