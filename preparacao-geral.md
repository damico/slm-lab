# Preparação geral dos laboratórios de SLMs

Execute este guia uma vez antes dos laboratórios 01 a 04. Todos usam um único ambiente virtual em `~/slm-labs/.venv`, com as versões já fixadas nos roteiros. Cada laboratório mantém sua própria pasta de scripts, dados, logs e checkpoints: alguns scripts têm o mesmo nome e precisam ficar separados.

## 1. Conferir a máquina

Os roteiros consideram Ubuntu, Bash, Python 3.11 ou 3.12, duas RTX 3070 de 8 GB e aproximadamente 32 GB de RAM. Confira:

```bash
python3 --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
df -h ~
```

O espaço necessário para dados e checkpoints está descrito em cada laboratório; um ambiente único reduz a duplicação de pacotes, mas os dados dos quatro experimentos continuam ocupando espaço. Execute os treinos em sequência quando usarem as duas GPUs.

Esta receita mantém o PyTorch 2.7.1 com runtime cu128 dos roteiros. A compatibilidade com o driver deve ser conferida na máquina antes do treino. A versão CUDA exibida por `nvidia-smi` descreve a capacidade do driver. Os wheels do PyTorch incluem seu runtime; estes laboratórios não exigem instalar um CUDA Toolkit separado.

## 2. Instalar ferramentas do sistema

Incluímos Tesseract e Poppler para o laboratório de recibos. `apt-get update` baixa os índices de pacotes do Ubuntu para localizar as versões disponíveis; `apt-get install` baixa as ferramentas do sistema usadas pelos roteiros, incluindo OCR, leitura de PDFs e download de arquivos.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git curl unzip time tesseract-ocr tesseract-ocr-eng poppler-utils
```

## 3. Criar o ambiente compartilhado

Os comandos `pip install` baixam as bibliotecas necessárias para carregar, quantizar e treinar os modelos, preparar datasets, processar imagens e calcular métricas. O PyTorch com CUDA 12.8 vem do índice oficial indicado no `requirements.txt`; as demais dependências também são resolvidas pelo pip. Essa instalação fornece software ao ambiente compartilhado; datasets e pesos são obtidos nas etapas dos laboratórios.

Se você já estiver em um ambiente de laboratório, desative-o antes. Os ambientes antigos podem permanecer nas suas pastas; os comandos seguintes passam a usar o novo ambiente compartilhado.

`cp ~/git/slm-lab/requirements.txt ./requirements.txt` leva a lista única de dependências do repositório para `~/slm-labs`, onde ela será usada na instalação do ambiente compartilhado.

```bash
if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate; fi
mkdir -p ~/slm-labs
cd ~/slm-labs
python3 -m venv .venv
source ~/slm-labs/.venv/bin/activate
python -m pip install --upgrade pip
cp ~/git/slm-lab/requirements.txt ./requirements.txt
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > requirements-resolved.txt
```

O arquivo [requirements.txt](requirements.txt) reúne as dependências dos quatro roteiros, incluindo o wheel CUDA 12.8 do PyTorch. O exemplo considera o repositório em `~/git/slm-lab`; ajuste a origem do comando `cp` se necessário. `requirements-resolved.txt` registra também as dependências transitivas instaladas. Execute os testes curtos de cada laboratório antes dos treinos completos.

## 4. Verificar imports e as duas GPUs

Execute o script [verificar_ambiente.py](verificar_ambiente.py). Ajuste o caminho do repositório se necessário.

```bash
python ~/git/slm-lab/verificar_ambiente.py
tesseract --version
pdftotext -v
```

Essa verificação cobre imports e uma operação simples em cada GPU. Os testes curtos dos roteiros conferem o carregamento quantizado e o treino dos modelos.

## 5. Usar o mesmo ambiente em qualquer laboratório

Em cada terminal novo, ative o ambiente pelo caminho completo. Mudar de pasta com `cd` mantém o ambiente ativo.

```bash
source ~/slm-labs/.venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_THREAD_LIMIT=1
```

Depois, entre na pasta do laboratório e siga seu roteiro:

| Laboratório | Roteiro |
|---|---|
| 01 — Ferramentas | [SLM e xLAM](laboratorio-01-slm-ferramentas-xlam.md) |
| 02 — SQL | [SLM e BIRD](laboratorio-02-slm-sql-bird.md) |
| 03 — Recibos | [Extração de recibos](laboratorio-03-slm-extracao-recibos.md) |
| 04 — Decisões | [Decisões com ABCD](laboratorio-04-slm-decisoes-abcd.md) |

Para continuar o laboratório 04 que você já iniciou:

A cópia do Markdown para `./laboratorio.md` deixa uma referência do roteiro na pasta do experimento. Ela é opcional para executar os scripts, que agora estão em arquivos `.py` separados.

As cópias de `~/slm-labs/requirements.txt` e `~/slm-labs/requirements-resolved.txt` para a pasta atual registram, junto dos resultados, a receita de dependências e as versões efetivamente instaladas. Elas não criam outro venv nem instalam pacotes.

`cp -n` copia os scripts da pasta específica deste laboratório no repositório para a pasta atual. Isso permite executar os comandos `python arquivo.py` e resolver os imports locais, mantendo dados e resultados isolados dos outros laboratórios. A opção `-n` preserva arquivos de destino que já existem.

```bash
cd ~/lab04-decisoes
source ~/slm-labs/.venv/bin/activate
cp ~/git/slm-lab/laboratorio-04-slm-decisoes-abcd.md ./laboratorio.md
cp ~/slm-labs/requirements.txt ./requirements.txt
cp ~/slm-labs/requirements-resolved.txt ./requirements-resolved.txt
cp -n ~/git/slm-lab/scripts/lab04-decisoes/*.py ./
mkdir -p raw runs results logs
export TOKENIZERS_PARALLELISM=false
```

O exemplo considera o repositório em `~/git/slm-lab`; ajuste esse caminho se necessário. Após copiar os scripts, continue na etapa de preparação dos dados do roteiro. A cópia dos scripts usa `cp -n` para preservar arquivos existentes. Para atualizar um script já copiado, compare-o com a versão do repositório antes de substituí-lo.

Os caches de modelos e datasets do Hugging Face usam seus caminhos padrão no usuário e podem ser reaproveitados entre laboratórios. Os diretórios locais `raw`, `data` e `runs` continuam específicos de cada experimento.

Para sair do ambiente ao terminar:

```bash
deactivate
```
