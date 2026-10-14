# Laboratório 05: cálculos financeiros verificáveis com FinQA e destilação offline

**Objetivo:** especializar um Qwen3-1.7B em perguntas numéricas sobre textos e tabelas de relatórios, usando um Qwen3-8B como professor. O modelo propõe evidências e um programa; um executor restrito calcula o resultado e mostra cada etapa. A hipótese é preservar parte da qualidade do professor em um aluno menor e verificar se a destilação acrescenta valor ao fine-tuning convencional.

Este laboratório usa o [FinQA oficial](https://github.com/czyssrs/FinQA), na revisão `0f16e2867befa6840783e58be38c9efb9229d742`. São **8.281 exemplos: 6.251 de treino, 883 de validação e 1.147 de teste público**. O corpus é em inglês e o repositório contém licença MIT. As tabelas e os textos já estão estruturados; OCR e extração de PDFs pertencem a outra etapa. Não usamos o teste privado sem gabarito nem afirmamos reproduzir o leaderboard oficial.

**Modalidade:** destilação de sequências offline. O professor gera respostas de treino, elas são verificadas e salvas, e o aluno aprende a reproduzi-las com QLoRA. Professor e aluno nunca precisam ficar carregados ao mesmo tempo. Não calculamos uma perda entre distribuições de probabilidades de ambos. [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)

## 1. O que a aplicação entrega

Exemplo didático independente do dataset: receita de 100 milhões para 118 milhões. A pergunta é o crescimento percentual. A saída tem somente `evidence` e `program`:

```json
{
  "evidence": ["table_1", "table_2"],
  "program": "subtract(118, 100), divide(#0, 100), multiply(#1, const_100)"
}
```

O executor retorna **18**, as etapas intermediárias **18 → 0,18 → 18** e as fontes citadas. Aqui a pergunta pede percentual; o modelo deve propor a conversão correspondente. O número não é calculado pelo LLM nem extraído de uma explicação textual.

São permitidas dez operações: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, `table_sum`, `table_average`, `table_min`, `table_max`. O programa contém até 12 chamadas binárias, com referências anteriores `#0`, `#1` etc. Operações de tabela usam o rótulo exato da linha e `none` no segundo argumento. Constantes permitidas estão declaradas no prompt e no executor. Não há `eval`, acesso a arquivos, execução de Python ou chamadas externas durante o cálculo.

Evidências são IDs como `text_3` e `table_2`. Literais numéricos precisam aparecer na pergunta ou nas fontes citadas; agregações de tabela precisam citar a própria linha. São recusados números sem origem, IDs inexistentes, divisão por zero, referências futuras, resultados não finitos e expoentes excessivos. Essas verificações não provam que o modelo escolheu o cálculo semanticamente correto: evidências irrelevantes podem conter o mesmo número. Por isso há revisão humana da supervisão e avaliação da resposta executada.

## 2. Ambiente único e separação das GPUs

Use a [preparação geral](preparacao-geral.md) e o único `requirements.txt` do repositório. Não crie outro venv nem instale as dependências antigas do projeto FinQA: este laboratório tem seu próprio executor e runtime, compatíveis com as versões compartilhadas.

O laço abaixo copia os scripts deste laboratório somente quando o destino ainda não existe, preservando suas cópias locais sem depender de `cp -n`. A cópia do Markdown mantém o roteiro junto ao experimento. O `requirements.txt` vem diretamente do repositório e registra a receita compartilhada; `python -m pip freeze` registra as versões efetivamente instaladas no venv ativo. Esses comandos não instalam pacotes nem criam um ambiente novo. Ajuste `~/git/slm-lab` se seu checkout estiver em outra pasta.

```bash
mkdir -p ~/slm-labs/05-finqa
cd ~/slm-labs/05-finqa
source ~/slm-labs/.venv/bin/activate
for LAB05_SCRIPT in ~/git/slm-lab/scripts/lab05-finqa/*.py; do
  if [ ! -e "./${LAB05_SCRIPT##*/}" ]; then
    cp "$LAB05_SCRIPT" ./
  fi
done
cp ~/git/slm-lab/laboratorio-05-slm-financeiro-finqa.md ./laboratorio.md
cp ~/git/slm-lab/requirements.txt ./requirements.txt
python -m pip freeze > requirements-resolved.txt
mkdir -p raw logs runs results
set -o pipefail
export TOKENIZERS_PARALLELISM=false
```

Se a versão anterior mostrou `cannot stat` para os requisitos, execute novamente este bloco corrigido. Ele não precisa dos arquivos em `~/slm-labs` e preserva scripts, dados e resultados existentes. O aviso de `cp -n` não impede a cópia; o laço evita essa opção.

Antes de usar GPUs, encerre as APIs de inferência dos outros labs e qualquer treino anterior. Modelos ociosos continuam ocupando VRAM. Confira ambas as placas e o disco:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
nvidia-smi
df -h .
```

| Fase | Configuração inicial |
|---|---|
| Professor | Qwen3-8B, NF4 de 4 bits, sem thinking, um processo Python |
| Distribuição do professor | `--shard`, mapa `balanced`, orçamento inicial de 5 GiB de pesos por GPU |
| Aluno | Qwen3-1.7B, QLoRA rank 16, LR 0,0001, microbatch 1 |
| Treino DDP | Uma cópia do aluno por GPU; acumulação 8, batch efetivo 16 |
| Janela | 2.048 tokens incluindo reserva de até 512 tokens de saída |
| Piloto | Até 200 candidatos do treino, antes de gerar o conjunto completo |

O orçamento de `--gpu-memory 5GiB` orienta a colocação de pesos; não é um limite rígido da VRAM total. Ativações, buffers e KV cache também precisam caber nos 8 GB. `--shard` é inferência por distribuição de camadas, não DDP nem uma promessa de acelerar duas vezes. O runtime recusa CPU/disco no mapa para evitar uma medição silenciosamente diferente. Ajuste o orçamento e a janela somente depois do smoke test. [Accelerate, Big Model Inference](https://huggingface.co/docs/accelerate/v1.10.0/en/usage_guides/big_modeling)

As duas placas não formam automaticamente uma GPU de 16 GB. No treino, cada placa precisa comportar seu aluno. Para o aluno usamos mapa explícito por `LOCAL_RANK`, NCCL com dispositivo vinculado e encerramento do grupo ao concluir. Para o professor use `python`, nunca `torchrun`. Reserve espaço também para o cache dos pesos originais, que são baixados antes da quantização, e para checkpoints; quantização em GPU não reduz automaticamente o download.

## 3. Preparar dados e travar revisões

`prepare_finqa.py` baixa `train.json`, `dev.json`, `test.json` e `LICENSE` da revisão fixada do FinQA. Os três JSONs fornecem perguntas, textos, tabelas e referências para a supervisão/avaliação; a licença é preservada com o material. A preparação consulta o Hugging Face para travar as revisões de **ambos os modelos** em `revisions.json` e baixa seus tokenizers para medir a janela real. Pesos ainda não são carregados nessa etapa. Os dados brutos ficam em `raw/finqa`; hashes e contagens ficam em `data/source_report.json`.

```bash
python prepare_finqa.py --raw raw/finqa --out data \
  --max-seq 2048 --max-new 512 --top-k 12 \
  2>&1 | tee logs/prepare.log
python inspect_finqa.py --data data
```

A recuperação é TF-IDF por documento, com unigrams e bigrams, usando **somente pergunta, pre_text, post_text e table**. Seleciona até 12 fontes mais o cabeçalho da tabela. Não usa `qa.gold_inds`, `qa.model_input`, `tfidftopn`, respostas, programas ou resultados para escolher o contexto. Selecione outro `top-k` na validação e preserve-o para todas as condições.

Arquivos produzidos:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: todos os exemplos dos splits nativos, com prompt e referências em campos separados.
- `sft_available/train.jsonl` e `sft_available/validation.jsonl`: alvos originais utilizáveis no contrato, com fontes recuperadas, cálculo válido e alvo dentro da janela.
- `train_candidates.jsonl`: o mesmo conjunto utilizável de treino, para o professor. O filtro facilita produzir o par SFT/destilado sem um alvo original impossível de ensinar nesse contexto.
- `config.json`, `source_report.json`: janela, revisões, hashes, contagens, cobertura de evidências e sobreposição de documentos entre splits.
- `demo_input.json`, `demo_prediction.json`: exemplo didático de crescimento, sem confundi-lo com uma predição do modelo.

Os campos de referência existem no JSONL para validação offline, mas **somente `messages` é passado à geração**. No teste, elegibilidade depende apenas do prompt dos dois tokenizers e da reserva de saída, nunca do tamanho da resposta ou da disponibilidade das evidências corretas. Todos os exemplos permanecem no teste, incluindo erros potenciais do professor e falhas de recuperação. `reference_valid` é outra flag, independente, que verifica se o programa original reproduz `exe_ans` sob as regras locais.

A filtragem de treino/validação utiliza os gabaritos para construir alvos válidos, o que é supervisão; não é a regra de elegibilidade do teste. A seleção favorece casos ensináveis nesse contexto e, depois, casos resolvidos pelo professor. Declare esse viés; não apresente o subconjunto como representação perfeita dos 6.251 exemplos.

A divisão é a nativa do FinQA. Não afirmamos que empresas/documentos sejam totalmente inéditos: o relatório registra a sobreposição, e a comparação pareada agrupa por relatório (empresa/ano extraídos do caminho), não apenas por página. Para estudar generalização para novas empresas, faça um experimento separado com nova divisão, sem alterar retrospectivamente o teste usado aqui.

Para refazer a preparação use outro `--out`; o script recusa um diretório não vazio. `--skip-download` reaproveita os quatro arquivos brutos já existentes, registra seus hashes e evita um novo download FinQA, mas tokenizers ainda dependem das revisões/cache.

## 4. Conferir o executor sem carregar modelos

```bash
python evaluate_finqa.py execute \
  --input data/demo_input.json --prediction data/demo_prediction.json \
  2>&1 | tee logs/demo_executor.log
```

O resultado esperado deste exemplo controlado é `answer: 18`, com três etapas e duas evidências. Isso valida a calculadora, não a capacidade do professor ou aluno de propor o programa correto.

**Convenção FinQA:** usamos `qa.exe_ans`, não a resposta textual `qa.answer`, que pode ter escala/apresentação diferente. Um literal `10%` vale `0.1`; o programa decide se precisa multiplicar por 100. A métrica local compara resultados arredondados a cinco casas decimais e trata `yes/no` separadamente. Não escolhe automaticamente entre uma resposta e cem vezes essa resposta para favorecer o modelo.

Acertar `exe_ans` não prova que a anotação financeira está correta. Divergências entre pergunta, resposta textual, programa e resultado podem ser apresentação/unidades ou inconsistências de anotação. A revisão deve conferir esses campos; qualquer correção de gabaritos cria uma versão separada do experimento, com novos hashes e avaliação comum das quatro condições.

A [auditoria CPU do executor](scripts/lab05-finqa/executor_audit.json), realizada em 13/10/2026, verificou os programas originais dos 8.281 exemplos: 6.246/6.251 no treino, 883/883 na validação e 1.147/1.147 no teste reproduziram `exe_ans`. Os cinco programas de treino incompatíveis têm linha de tabela ausente/ambígua ou segundo argumento de tabela fora do contrato. A preparação registra essas exclusões. Esses números não são acurácias de modelos.

## 5. Professor: smoke test e piloto offline

Na primeira inferência, o runtime baixa do Hugging Face os pesos/configuração do Qwen3-8B na revisão fixada que ainda não estiverem no cache. O professor fornece a supervisão gerada; não precisa de treinamento nem adaptador. Os pesos do Qwen3-1.7B serão obtidos quando o aluno for carregado. Ambos são quantizados em memória pelo bitsandbytes; os caches são reutilizados nos demais labs quando a revisão coincide. [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [QLoRA com PEFT](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization)

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python finqa_runtime.py predict --data data --model Qwen/Qwen3-8B \
  --shard --gpu-memory 5GiB --input data/train_candidates.jsonl \
  --output results/teacher_smoke.jsonl --limit 3 \
  2>&1 | tee logs/teacher_smoke.log
python inspect_finqa.py --data data --predictions results/teacher_smoke.jsonl
```

Se couber e concluir, gere até 200 candidatos para o piloto. São perguntas determinísticas do treino, não do teste. O professor recebe o mesmo contexto limitado que o aluno receberá; não fornecemos fontes corretas escolhidas pelo gabarito nem a resposta.

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python finqa_runtime.py predict --data data --model Qwen/Qwen3-8B \
  --shard --gpu-memory 5GiB --input data/train_candidates.jsonl \
  --output results/teacher_pilot.jsonl --limit 200 \
  2>&1 | tee logs/teacher_pilot.log
python inspect_finqa.py --data data --predictions results/teacher_pilot.jsonl
```

As gerações usam decodificação gulosa, `enable_thinking=False`, seed 42 e o mesmo contrato JSON em todas as condições. O limite de saída faz parte da configuração preparada. O programa é curto; este experimento não destila explicações longas nem raciocínio oculto do professor.

Ao terminar cada comando, o processo do professor encerra e libera as GPUs. Os JSONLs são gravados e descarregados linha a linha, com metadados de revisão, hashes, mapa de dispositivos e pico alocado/reservado de **cada GPU**. Para repetir use outro nome. Para retomar uma inferência interrompida, repita o mesmo comando com `--resume`; isso exige assinatura idêntica, mesmo limite e JSONL íntegro. Uma última linha incompleta precisa ser preservada/inspecionada antes de recuperar o arquivo; o script não a apaga silenciosamente.

## 6. Validar a supervisão e construir o par experimental

```bash
python build_distillation.py --data data \
  --predictions results/teacher_pilot.jsonl --out paired_pilot \
  2>&1 | tee logs/filter_pilot.log
python -m json.tool paired_pilot/selection_report.json
head -n 3 paired_pilot/review.jsonl
```

O validador exige: origem Qwen3-8B com revisão/hash do conjunto corretos, IDs exclusivos do treino, geração sem truncamento, JSON estrito, operações permitidas, operandos com origem nas evidências, resultado correto frente a `exe_ans` e alvo dentro da janela do aluno. Não substitui erros do professor pelo gabarito. Se nenhuma resposta passar, interrompe e deixa `audit.jsonl` para diagnóstico.

**Revise `paired_pilot/review.jsonl` antes de treinar.** A amostra é determinística e inclui pergunta, fontes, saída do professor, execução e referência. Conferir o número não basta: um cálculo incorreto pode coincidir por acaso. A aprovação automática não é uma garantia de causalidade das evidências. Se a revisão revelar erros, ajuste critérios na validação e regenere o par; não edite só o conjunto destilado para favorecer a comparação.

Dois conjuntos são produzidos com **IDs, prompts e validação idênticos**:

| Pasta | Alvo de treino |
|---|---|
| `paired_pilot/sft` | Evidências/programas originais do FinQA |
| `paired_pilot/distilled` | Evidências/programas aprovados do professor |

A validação usa os mesmos alvos originais nas duas condições. O relatório preserva os IDs aprovados, quantidade gerada/aprovada, hashes e a amostra revisável. O professor pode gerar um programa diferente do original; esse é precisamente o efeito que queremos testar. Se gerar alvos praticamente idênticos, pode haver pouco sinal adicional para a destilação.

## 7. Treino do aluno: smoke test e piloto pareado

Confira que o professor e as APIs encerraram. Treinos usam uma cópia Qwen3-1.7B por GPU, e a perda ignora os tokens de entrada. O aluno não acessa o professor durante o treino.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -- finqa_runtime.py train \
  --data paired_pilot/distilled --run runs/smoke_distilled \
  --max-steps 10 --accum 8 --eval-steps 5 --eval-limit 16 \
  2>&1 | tee logs/smoke_distilled.log
```

O smoke test é diagnóstico; não use seu adaptador como resultado final. Para o piloto pareado, dois treinos novos com as mesmas perguntas, semente, 40 atualizações, rank e avaliação:

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_pilot/sft --run runs/pilot_sft \
  --max-steps 40 --accum 8 --rank 16 --lr 0.0001 --eval-steps 10 --eval-limit 128 \
  2>&1 | tee logs/pilot_sft.log
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_pilot/distilled --run runs/pilot_distilled \
  --max-steps 40 --accum 8 --rank 16 --lr 0.0001 --eval-steps 10 --eval-limit 128 \
  2>&1 | tee logs/pilot_distilled.log
```

O orçamento é igual por exemplos/atualizações. As respostas podem ter comprimentos diferentes; `run_report.json` registra os tokens-alvo para não alegar igualdade exata de FLOPs ou tempo. Cada treino começa da base original e salva `final_adapter`. A seleção do checkpoint usa perda de validação; qualidade funcional é medida depois pelo executor, não inferida da perda. O relatório registra duração, pico alocado no rank 0, hashes e parâmetros. Para retomar treino use `--resume runs/.../checkpoint-N`, com os mesmos dados e configuração.

## 8. Conferir as quatro condições na validação

```bash
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/validation.jsonl --output results/val_base.jsonl
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/validation.jsonl --output results/val_sft.jsonl \
  --adapter runs/pilot_sft/final_adapter
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/validation.jsonl --output results/val_distilled.jsonl \
  --adapter runs/pilot_distilled/final_adapter
CUDA_VISIBLE_DEVICES=0,1 python finqa_runtime.py predict --data data \
  --model Qwen/Qwen3-8B --shard --gpu-memory 5GiB \
  --input data/validation.jsonl --output results/val_teacher.jsonl
```

```bash
python evaluate_finqa.py score data/validation.jsonl results/val_base.jsonl --out results/val_base
python evaluate_finqa.py score data/validation.jsonl results/val_sft.jsonl --out results/val_sft
python evaluate_finqa.py score data/validation.jsonl results/val_distilled.jsonl --out results/val_distilled
python evaluate_finqa.py score data/validation.jsonl results/val_teacher.jsonl --out results/val_teacher
python evaluate_finqa.py compare results/val_sft.cases.jsonl results/val_distilled.cases.jsonl
```

Use somente a validação para escolher recuperação, janela, formato e hiperparâmetros. Um piloto pequeno pode não mostrar ganho; isso justifica revisar supervisão e diversidade antes de gerar milhares de respostas. Congele as escolhas antes do teste. Se mudar janela/reserva, refaça a preparação em outra pasta e reavalie todas as condições.

## 9. Destilação completa e treinos finais

Se o piloto for satisfatório, gere os candidatos completos do treino, sem `--limit`. Este comando não gera respostas de teste para supervisão.

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python finqa_runtime.py predict --data data --model Qwen/Qwen3-8B \
  --shard --gpu-memory 5GiB --input data/train_candidates.jsonl \
  --output results/teacher_train.jsonl \
  2>&1 | tee logs/teacher_train.log
python inspect_finqa.py --data data --predictions results/teacher_train.jsonl
python build_distillation.py --data data \
  --predictions results/teacher_train.jsonl --out paired_main \
  2>&1 | tee logs/filter_main.log
python -m json.tool paired_main/selection_report.json
head -n 3 paired_main/review.jsonl
```

Faça a revisão antes de executar os dois treinos finais. Mesma seleção aprovada, mesmas épocas, batch efetivo, LR, rank e seed:

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_main/sft --run runs/main_sft \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-steps 100 --eval-limit 128 \
  2>&1 | tee logs/main_sft.log
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_main/distilled --run runs/main_distilled \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-steps 100 --eval-limit 128 \
  2>&1 | tee logs/main_distilled.log
```

Pode haver menos de 100 passos em um subconjunto pequeno; nesse caso use uma frequência de avaliação menor, escolhida previamente e aplicada a ambos. Não confunda o piloto de 40 passos com este treinamento final por épocas.

## 10. Teste completo com os mesmos 1.147 exemplos

```bash
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/test.jsonl --output results/test_base.jsonl
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/test.jsonl --output results/test_sft.jsonl \
  --adapter runs/main_sft/final_adapter
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/test.jsonl --output results/test_distilled.jsonl \
  --adapter runs/main_distilled/final_adapter
CUDA_VISIBLE_DEVICES=0,1 python finqa_runtime.py predict --data data \
  --model Qwen/Qwen3-8B --shard --gpu-memory 5GiB \
  --input data/test.jsonl --output results/test_teacher.jsonl
```

Nenhum exemplo é removido porque o professor errou. `context_exceeded`, `output_limit`, predições ausentes e programas rejeitados contam como falhas nos denominadores pertinentes. Geração recebe somente contexto/pergunta e instruções públicas; evidências anotadas, programas e `exe_ans` ficam fora do prompt.

```bash
python evaluate_finqa.py score data/test.jsonl results/test_base.jsonl --out results/base
python evaluate_finqa.py score data/test.jsonl results/test_sft.jsonl --out results/sft
python evaluate_finqa.py score data/test.jsonl results/test_distilled.jsonl --out results/distilled
python evaluate_finqa.py score data/test.jsonl results/test_teacher.jsonl --out results/teacher
python evaluate_finqa.py compare results/base.cases.jsonl results/sft.cases.jsonl
python evaluate_finqa.py compare results/sft.cases.jsonl results/distilled.cases.jsonl
python evaluate_finqa.py compare results/teacher.cases.jsonl results/distilled.cases.jsonl
```

## 11. Medidas e interpretação

| Condição | Resultado executado | Programas válidos | Cobertura | p50 / p95 | VRAM | Tempo/energia |
|---|---|---|---|---|---|---|
| Qwen3-1.7B original | medir | medir | mesma regra | medir | medir | medir |
| Aluno com gabaritos | medir | medir | mesma regra | medir | medir | medir |
| Aluno destilado | medir | medir | mesma regra | medir | medir | medir |
| Qwen3-8B professor | medir | medir | mesma regra | medir | ambas as GPUs | medir |

`accuracy_eligible` usa referências válidas que cabem no contexto; `accuracy_all_valid` inclui as que não couberam. `program_valid_rate` exige cálculo permitido e operandos fundamentados nas evidências. Mostre também recall das evidências e casos de coincidência numérica. A métrica é **local_grounded_execution_accuracy_not_official_FinQA**: não é o avaliador oficial de equivalência simbólica, nem certifica significado, unidades ou exatidão financeira em produção.

O intervalo de Wilson é por exemplo; a diferença pareada usa bootstrap por documento. Relate o número de grupos e as sobreposições dos splits. Benchmarks públicos podem ter sido vistos no pré-treinamento: nossa separação impede usar o teste na destilação, mas não certifica ausência de contaminação anterior dos Qwen.

Latência inclui geração após aquecimento, sem carregar pesos; a execução aritmética é registrada no resultado da aplicação. O professor usa duas GPUs por distribuição de camadas e o aluno uma GPU na inferência: declare essa diferença de recursos. Repita medições em arquivos distintos se precisar estimar dispersão.

O laboratório usa professor local, sem chamadas pagas de API. Isso não significa custo zero. Meça duração, potência das GPUs e custo de energia/hardware, incluindo a geração offline e os dois treinos. `inspect_finqa.py` projeta a duração do professor usando as latências observadas do piloto; não há uma duração inventada antes dele. Estime `N_candidatos × segundos_médios`, mais carregamento e filtragem. Treino tem aproximadamente `ceil(N_aprovados / 16) × épocas` atualizações; acrescente avaliações e gravações.

Para potência, execute o monitor em um **terminal separado**, durante a fase que deseja medir, e encerre com Ctrl+C ao final:

```bash
cd ~/slm-labs/05-finqa
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_phase.csv
```

Use outro nome para cada fase. Integrar watts ao longo do tempo estima energia das GPUs, não da máquina inteira. Para estimar amortização da destilação, divida o custo de preparação/treino pela economia por inferência; se a economia for desconhecida ou negativa, não afirme um ponto de equilíbrio.

Uma conclusão válida pode ser que o aluno destilado se aproxime do professor, que só melhore o aluno original, que não supere SFT convencional ou que piore. Não escolha exclusivamente casos favoráveis. A comparação SFT/destilado com os mesmos IDs é a principal evidência do efeito da supervisão do professor.

## 12. API do aluno funcionando sozinho

Depois do treino, o aluno carrega seu próprio adaptador uma vez, sem professor. Este serviço recebe tabela/textos estruturados; não lê relatórios PDF automaticamente.

```bash
CUDA_VISIBLE_DEVICES=1 python serve_finqa.py --data data \
  --adapter runs/main_distilled/final_adapter --port 8085 \
  2>&1 | tee logs/api_finqa.log
```

Em outro terminal:

```bash
cd ~/slm-labs/05-finqa
curl -fsS http://127.0.0.1:8085/predict \
  -H 'Content-Type: application/json' --data-binary @data/demo_input.json
```

A resposta inclui a geração, resultado executado, operações intermediárias e fontes. O exemplo didático da calculadora tem resposta 18; a inferência real pode errar e será exibida como tal. `requires_review=true` indica que validade mecânica não garante que a interpretação/unidade esteja correta; não inventamos confiança calibrada para este modelo generativo. A API rejeita campos de gabarito e programas arbitrários enviados junto do contexto.

O dashboard do [avaliador web](avaliador/README.md) mostra o Lab 05 e seu roteiro como **em preparação**, até o adaptador ser treinado e a integração web ser implementada. A API Python é executável após o treino; não mostramos inferência simulada como se houvesse pesos prontos.

## 13. Diagnóstico e arquivos a preservar

| Sintoma | Ação |
|---|---|
| OOM no professor | Pare outros processos, confira duas GPUs, mantenha um processo Python, ajuste orçamento de pesos e reduza janela em uma nova preparação se necessário. |
| Mapa em CPU/disco | O runtime interrompe. Confira `--gpu-memory`, GPUs visíveis e contexto antes de mudar a configuração. |
| OOM no aluno | Professor/APIs devem estar descarregados; mantenha microbatch 1. Reduza contexto ou rank igualmente nas duas condições em outro experimento. |
| Poucos candidatos/aprovados | Consulte source_report/audit; pode ser recuperação, janela, contrato ou erros do professor. Não preencha falhas copiando o gabarito para o conjunto destilado. |
| Programa numérico certo, explicação errada | Revise fontes e programa; as verificações automáticas não provam causalidade. |
| Resultado 0,18 versus 18 | Confira a pergunta, unidades e convenção `exe_ans`; não use tolerância de fator 100 para esconder erro. |
| Inferência parcial | `--resume` somente com mesmos parâmetros, limite e assinatura. Preserve arquivos interrompidos antes de qualquer reparo. |
| Treino aparenta terminar sem artefato | Espere adaptador/relatório e retorno ao terminal; não use somente a barra em 100% como confirmação. |

Preserve roteiro, requisitos resolvidos, licença, brutos/hash, revisões, prompts, supervisão do professor, auditoria, amostra revisada, pares, logs, checkpoints, adaptadores, predições e métricas. O aluno ajustado e o destilado devem ser artefatos separados.

**Validação da implementação:** executor auditado em CPU no FinQA oficial, testes de origem dos operandos, prevenção de entrada do gabarito no prompt, filtragem pareada e contrato da API. Não foram executados geração Qwen3-8B, treinamento CUDA ou inferências dos modelos para este novo laboratório. Memória, tempo e ganho de destilação precisam ser medidos no piloto da sua máquina.

## 14. Referências e scripts

- [FinQA, dataset e licença](https://github.com/czyssrs/FinQA).
- [FinQA, artigo](https://arxiv.org/abs/2109.00122).
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947).
- [QLoRA](https://arxiv.org/abs/2305.14314).
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) e [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
- [Accelerate 1.10, inferência de modelos grandes](https://huggingface.co/docs/accelerate/v1.10.0/en/usage_guides/big_modeling).
- [PEFT 0.17, quantização](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).

Scripts separados em [scripts/lab05-finqa](scripts/lab05-finqa): contrato/executor (`finqa_core.py`), preparação, runtime, construção da destilação, avaliação, inspeção e API. Todo código Python está em `.py`, fora do Markdown. Fontes verificadas em 13/10/2026.

O [script Bash](laboratorio-05-slm-financeiro-finqa.sh) reúne os 18 blocos Bash extraídos deste roteiro e grava log próprio. Use `--list` e selecione etapas com `--steps`, por exemplo `--steps 1-4`. Há servidores/monitor em primeiro plano e etapas que dependem de revisão humana: não execute todas cegamente. `LAB_WORK_DIR` permite escolher outra pasta de trabalho. O [diagrama Mermaid](diagramas/laboratorio-05-finqa.mmd) mostra a separação entre professor, validação, pares, aluno e avaliação.
