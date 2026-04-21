<div align="center">

# Operacionalização de Modelos com MLOps
### Projeto de Disciplina — PD2

**Pós-Graduação em Machine Learning, Deep Learning e Inteligência Artificial**<br>
**Disciplina:** Operacionalização de Modelos com MLOps<br>
**Aluno:** Deyvison Ramos<br>
[**Projeto no Github**](https://github.com/deyvisonramos/fundamento_ml/tree/operacionalizacao_modelos)<br>

> **o projeto está na branch "operacionalizacao_modelos". O link leva diretamente para a branch correta.

[**Apresentação em Vídeo**](https://drive.google.com/drive/folders/1VdpkHmv2W1hJoCCDD7l8YlCVKDohCWGK?usp=sharing)

<p>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/MLflow-rastreamento-0194E2?style=flat-square" alt="MLflow">
  <img src="https://img.shields.io/badge/Streamlit-inferência-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

</div>

---

## Visão Geral

Este projeto representa a evolução natural do PD1: saímos de um notebook exploratório para um **sistema MLOps completo**, estruturado com a perspectiva de um engenheiro de machine learning.

O problema continua sendo o mesmo — prever se um cliente de banco irá subscrever um **depósito a prazo** (`deposit = yes/no`) com base em dados demográficos, financeiros e de histórico de campanha. O que muda é a forma como o trabalho é conduzido: código modular, experimentos rastreáveis, redução de dimensionalidade avaliada sistematicamente e modelo operacionalizado em interface de inferência.

**Dataset:** Bank Marketing | 11.162 registros | 16 features (7 numéricas + 9 categóricas)

---

## Estrutura do Projeto

```
├── config/
│   └── pipeline.yaml              # Fonte única de verdade — todos os parâmetros aqui
├── src/
│   ├── data_processing.py         # Ingestão, diagnóstico de qualidade e pré-processamento
│   ├── train.py                   # Treinamento dos 7 experimentos com rastreamento MLflow
│   └── inference.py               # Carregamento do modelo champion e predição
├── app/
│   └── app.py                     # Interface Streamlit (predição + monitoramento de drift)
├── reports/
│   ├── relatorio_tecnico.md       # ← Relatório técnico completo (decisões, análises, resultados)
│   └── metrics_experiments.csv    # Métricas de todos os experimentos (gerado automaticamente)
├── models/
│   └── champion_run_id.txt        # run_id do modelo champion selecionado (gerado)
├── data/raw/
│   └── bank.csv                   # Dataset bruto
├── notebooks/
│   └── projeto_ml_sklearn_bank_marketing.ipynb  # Exploração inicial — PD1
├── mlruns/                        # Artefatos MLflow (gerado automaticamente)
└── requirements.txt
```

> O relatório técnico com todas as decisões de projeto, análise comparativa dos experimentos e justificativa da abordagem final está em **[`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)**.

---

## Resultados dos Experimentos

Sete experimentos foram executados e rastreados no MLflow: três baselines e quatro variantes com redução de dimensionalidade (PCA e LDA).

| Experimento | Redução | F1 | ROC-AUC | Precisão | Recall | CV-F1 Médio | Tempo (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline_svm` ⭐ | — | **0.853** | 0.919 | 0.813 | 0.898 | **0.844±0.006** | 9.1 |
| `baseline_random_forest` | — | 0.847 | 0.917 | 0.808 | 0.890 | 0.842±0.005 | 0.7 |
| `baseline_gradient_boosting` | — | 0.845 | **0.926** | 0.830 | 0.860 | 0.841±0.005 | 1.2 |
| `rf_com_pca` | PCA (95%) | 0.836 | 0.903 | 0.800 | 0.875 | 0.833±0.004 | 2.0 |
| `gb_com_lda` | LDA (1 comp) | 0.821 | 0.903 | 0.797 | 0.846 | 0.820±0.006 | 0.5 |
| `gb_com_pca` | PCA (95%) | 0.818 | 0.902 | 0.803 | 0.833 | 0.830±0.007 | 4.7 |
| `rf_com_lda` | LDA (1 comp) | 0.817 | 0.899 | 0.785 | 0.851 | 0.815±0.006 | 0.7 |

**⭐ Champion: `baseline_svm`** — SVM com kernel RBF, maior F1 no conjunto de teste e validação cruzada estável. A redução de dimensionalidade causou queda de 1 a 3 pontos percentuais em todos os modelos, indicando que as 42 features pós-encoding são suficientemente discriminativas.

---

## Como Reproduzir

### Requisitos

- Python 3.11+
- Conda (recomendado)

### Configuração do Ambiente

```bash
conda create -n ml_project python=3.11 -y
conda activate ml_project
pip install -r requirements.txt
```

### Execução Passo a Passo

```bash
# 1. Diagnóstico dos dados
python src/data_processing.py

# 2. Treinamento dos 7 experimentos + seleção do champion
python src/train.py

# 3. Interface MLflow (visualização dos experimentos)
mlflow ui
# → http://localhost:5000

# 4. Interface de inferência + monitoramento
streamlit run app/app.py
# → http://localhost:8501

# 5. Teste de inferência via script (opcional)
python src/inference.py
```

---

## O Que Foi Entregue

Alinhado às competências avaliadas na rubrica:

| Competência | Entregável |
|---|---|
| Estruturação do projeto de ML | Código modular em `src/`, configuração centralizada em `config/pipeline.yaml`, separação clara entre dados, treino e inferência |
| Fundação de dados e diagnóstico | `src/data_processing.py` com qualidade, outliers, distribuição do target e riscos documentados |
| Experimentação sistemática com MLflow | 7 experimentos rastreados com parâmetros, métricas, artefatos e modelos serializados |
| Redução de dimensionalidade | PCA (95% de variância) e LDA (1 componente) integrados ao pipeline e avaliados comparativamente |
| Seleção e justificativa do champion | Critério F1, run_id persistido, justificativa técnica em `reports/relatorio_tecnico.md` |
| Operacionalização | Interface Streamlit com predição individual, valor esperado de negócio e monitoramento de drift via KS-test |

---

## Decisões Técnicas Relevantes

- **`duration` (risco de leakage):** Feature disponível apenas após a ligação — inapropriada para decisões pré-chamada em produção. Mantida para fins de comparação acadêmica, com limitação formalmente documentada no relatório.
- **Redução de dimensionalidade:** PCA e LDA reduziram o F1 em todos os cenários testados. A conclusão técnica é que as features pós-encoding são não-redundantes o suficiente para não se beneficiarem de compressão linear — detalhamento completo no relatório.
- **Champion por F1 vs ROC-AUC:** F1 foi escolhido como critério por refletir melhor o desempenho operacional no limiar de decisão padrão — mais alinhado ao contexto de campanha de marketing do que a capacidade discriminativa geral (ROC-AUC).
- **Reprodutibilidade:** `random_state=42` em todos os pontos, `StratifiedKFold(n_splits=5)` no cross-validation, parâmetros centralizados em `config/pipeline.yaml`.

---

## Evolução em Relação ao PD1

| Aspecto | PD1 (Notebook) | PD2 (MLOps) |
|---------|----------------|-------------|
| Código | Notebook único | Módulos Python separados |
| Configuração | Hardcoded | `config/pipeline.yaml` |
| Experimentos | Manual/sequencial | MLflow (rastreável e reproduzível) |
| Redução de dimensionalidade | Não avaliada | PCA + LDA integrados ao pipeline |
| Inferência | Somente no notebook | `src/inference.py` + Streamlit |
| Monitoramento | Nenhum | KS-test + dashboard Streamlit |
| Reprodutibilidade | Parcial | Total (config + MLflow) |

---

## Relatório Técnico

O relatório completo, com diagnóstico de dados, análise comparativa dos experimentos, avaliação das técnicas de redução de dimensionalidade, justificativa do champion e métricas de negócio projetadas, está disponível em:

**[`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)**

---


<div align="center">
  <small>Desenvolvido para fins acadêmicos — Abril / 2026</small>
</div>
