# Projeto ML Supervisionado com Scikit-Learn

Projeto de classificação binária supervisionada usando o dataset Bank Marketing. O objetivo é prever se um cliente vai aderir a um depósito a prazo (`deposit=yes/no`) a partir de variáveis demográficas, financeiras e do histórico da campanha.

## Estrutura

- `projeto_ml_sklearn_bank_marketing.ipynb`: notebook principal com a análise completa
- `dataset/bank.csv`: base de dados utilizada
- `outputs/`: artefatos e figuras mantidos no repositório
- `requirements.txt`: dependências do projeto

## Requisitos

- Python 3.11 ou superior
- `pip`

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Se preferir usar Conda:

```bash
conda create -n ml_project python=3.11 -y
conda activate ml_project
pip install -r requirements.txt
```

## Como executar

1. Abra o notebook:

```bash
jupyter notebook projeto_ml_sklearn_bank_marketing.ipynb
```

O notebook é autossuficiente e gera análises, métricas e gráficos diretamente a partir de `dataset/bank.csv`.

## Principais modelos avaliados

- Perceptron baseline
- Árvore de decisão default e otimizada
- SVM com busca de hiperparâmetros
- Random Forest
- Gradient Boosting
- Voting Ensemble

## Observações

- O projeto utiliza apenas modelos do ecossistema `scikit-learn`.
- O arquivo `.ipynb` já foi executado e inclui as saídas renderizadas para facilitar a avaliação no GitHub.
