# Sistema DNS Personalizado

Este projeto implementa um sistema DNS utilizando árvores AVL e Rubro-Negra para armazenamento e gerenciamento de registros DNS. Inclui ferramentas para benchmarks, geração de dados realistas, análise de desempenho e uma interface interativa para operações DNS.

## Sumário

- [Instalação](#instalação)
- [Requisitos](#requisitos)
- [Visão Geral dos Modos de Uso](#visão-geral-dos-modos-de-uso)
- [Comandos Disponíveis](#comandos-disponíveis)
  - [Benchmark](#benchmark)
  - [Geração de Dados](#geração-de-dados)
  - [Modo Interativo](#modo-interativo)
  - [Análise de Resultados](#análise-de-resultados)
- [Exemplos de Uso](#exemplos-de-uso)
- [Estrutura dos Dados](#estrutura-dos-dados)
- [Resultados e Visualizações](#resultados-e-visualizações)

---

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone <url-do-repositório>
cd dns_system
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

Certifique-se de ter o Python 3.7+ instalado.

---

## Requisitos

- Python 3.7 ou superior
- Bibliotecas: `faker`, `matplotlib`, `pandas`, `numpy`, `seaborn` (e outras listadas no requirements.txt)

---

## Visão Geral dos Modos de Uso

O sistema pode ser utilizado via linha de comando pelo arquivo `dns_cli.py`, que oferece quatro modos:

1. **Benchmark:** Executa testes de desempenho comparando AVL e Rubro-Negra.
2. **Geração de Dados:** Cria arquivos de dados DNS realistas para testes.
3. **Modo Interativo:** Permite manipular registros DNS manualmente.
4. **Análise:** Gera relatórios e gráficos a partir dos resultados dos benchmarks.

---

## Comandos Disponíveis

### Benchmark

Executa benchmarks de desempenho com diferentes tamanhos de dados e distribuições de operações.

```bash
python dns_cli.py benchmark [opções]
```

**Principais opções:**

- `--sizes`: Tamanhos dos conjuntos de dados (ex: `--sizes 100,1000,10000`)
- `--operations`: Distribuição das operações (busca,inserção,add_ip,remoção) (ex: `--operations 90,5,3,2`)
- `--operation-count`: Número de operações por benchmark
- `--output`: Arquivo de saída dos resultados (padrão: `results/benchmark_results.json`)
- `--full`: Executa todos os benchmarks padrões
- `--comprehensive`: Executa benchmarks completos com tamanhos customizados
- `--analyze`: Gera análise e visualizações após os benchmarks
- `--seed`: Semente para reprodutibilidade
- `--num-runs`: Número de execuções por benchmark (padrão: 5)
- `--warmup-runs`: Execuções de aquecimento (padrão: 2)

### Geração de Dados

Gera um arquivo JSON com domínios e IPs realistas.

```bash
python dns_cli.py generate --size <N> --output <arquivo.json> [--distribution <dist>]
```

- `--size`: Quantidade de domínios a gerar (obrigatório)
- `--output`: Nome do arquivo de saída (obrigatório)
- `--distribution`: Distribuição de IPs por domínio (ex: `40,35,25` para 1, 2 e 3 IPs)

### Modo Interativo

Permite manipular registros DNS manualmente.

```bash
python dns_cli.py interactive [--tree-type avl|red-black] [--load-data <arquivo.json>]
```

- `--tree-type`: Tipo de árvore (AVL ou Rubro-Negra, padrão: AVL)
- `--load-data`: Arquivo JSON para carregar dados iniciais

**Comandos disponíveis no modo interativo:**

- `add <domínio> <ip>`: Adiciona um novo domínio
- `search <domínio>`: Busca um domínio
- `remove <domínio>`: Remove um domínio
- `addip <domínio> <ip>`: Adiciona um IP a um domínio existente
- `removeip <domínio> <ip>`: Remove um IP de um domínio
- `list`: Lista todos os domínios e IPs
- `stats`: Mostra estatísticas do sistema
- `help`: Mostra a ajuda
- `quit`: Sai do modo interativo

### Análise de Resultados

Gera relatórios e gráficos a partir de um arquivo de resultados de benchmark.

```bash
python dns_cli.py analyze --input <arquivo.json> [--output-dir <diretório>]
```

- `--input`: Arquivo JSON com resultados de benchmark (obrigatório)
- `--output-dir`: Diretório para salvar as análises (padrão: `results/analysis`)

---

## Exemplos de Uso

- Executar benchmarks padrões:
  ```bash
  python dns_cli.py benchmark --full
  ```

- Executar benchmarks customizados:
  ```bash
  python dns_cli.py benchmark --sizes 500,2000 --operations 80,10,5,5
  ```

- Gerar 1000 domínios de teste:
  ```bash
  python dns_cli.py generate --size 1000 --output dados.json
  ```

- Iniciar modo interativo com árvore Rubro-Negra:
  ```bash
  python dns_cli.py interactive --tree-type red-black
  ```

- Carregar dados no modo interativo:
  ```bash
  python dns_cli.py interactive --load-data dados.json
  ```

- Analisar resultados de benchmark:
  ```bash
  python dns_cli.py analyze --input results/benchmark_results.json
  ```

---

## Estrutura dos Dados

Os dados de domínios são armazenados em formato JSON, por exemplo:

```json
[
  ["exemplo.com", ["192.168.1.1"]],
  ["empresa.org", ["8.8.8.8", "8.8.4.4"]]
]
```

---

## Resultados e Visualizações

Os resultados dos benchmarks e análises são salvos em `results/`. Gráficos e tabelas são gerados automaticamente, incluindo:

- Tempos de operação por tamanho de dados
- Comparação de throughput
- Altura das árvores
- Análise de rotações (AVL vs Rubro-Negra)
- Dashboards comparativos

Veja a pasta `results/analysis/` para exemplos de gráficos e relatórios.

---

## Valores Padrão dos Testes (Benchmarks)

Ao executar o comando de benchmark sem especificar opções, os seguintes valores padrão são utilizados:

- **Tamanhos dos conjuntos de dados (`--sizes`)**: `100,1000,10000`
- **Distribuição das operações (`--operations`)**: `90,5,3,2`  
  (90% busca, 5% inserção, 3% adição de IP, 2% remoção)
- **Número de operações por benchmark (`--operation-count`)**: 2x o tamanho do conjunto de dados
- **Arquivo de saída dos resultados (`--output`)**: `results/benchmark_results.json`
- **Semente para reprodutibilidade (`--seed`)**: aleatória (diferente a cada execução, a menos que especificado)
- **Número de execuções por benchmark (`--num-runs`)**: 5
- **Execuções de aquecimento descartadas (`--warmup-runs`)**: 2

Esses valores podem ser alterados conforme necessário utilizando as opções correspondentes no comando `benchmark`.
