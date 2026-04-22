# Relatório Técnico — Operacionalização de Modelos com MLOps
**Disciplina:** Operacionalização de Modelos com MLOps | **Aluno:** Deyvison Ramos  
**Dataset:** Bank Marketing (11.162 registros) | **Tarefa:** Classificação Binária

---

## 1. Contexto e Objetivos

### Problema de Negócio
O objetivo é prever se um cliente de um banco irá subscrever um depósito a prazo (`deposit = yes/no`) com base em dados demográficos, financeiros e histórico de contatos de campanha de marketing. A predição permite direcionar os recursos de campanha para clientes com maior probabilidade de conversão, reduzindo o custo operacional e aumentando a taxa de sucesso.

### Objetivos Técnicos
- Estruturar um projeto MLOps reproduzível com separação clara de responsabilidades (dados, treino, inferência, monitoramento)
- Registrar e comparar sistematicamente experimentos utilizando MLflow
- Avaliar o impacto de técnicas de redução de dimensionalidade (PCA e LDA) no pipeline de classificação
- Operacionalizar o modelo champion em uma interface de inferência funcional (Streamlit)

### Critérios de Sucesso
| Métrica | Critério |
|---------|----------|
| F1-score (macro) | > 0.85 |
| ROC-AUC | > 0.90 |
| Reprodutibilidade | 100% via `config/pipeline.yaml` + MLflow |

### Métricas de Negócio
- **Custo por contato de campanha:** $5
- **Receita por conversão:** $200
- **Valor esperado por cliente:** `P(yes) × $200 − $5`
- **Trade-off:** Recall alto reduz oportunidades perdidas; Precisão alta reduz o custo de campanha. O limiar de decisão é configurável na interface.

---

## 2. Diagnóstico de Dados

### Dataset: Bank Marketing
| Atributo | Valor |
|---------|-------|
| Registros | 11.162 |
| Features | 16 (7 numéricas + 9 categóricas) |
| Target | `deposit` (yes/no) |
| Balanceamento | ~53% yes / ~47% no |

### Qualidade dos Dados
- **Valores nulos:** Nenhum campo contém valores ausentes nativos no CSV. A feature `pdays = -1` codifica ausência de contato anterior, tratada por imputação de mediana.
- **Duplicatas:** Nenhuma linha duplicada identificada.
- **Outliers (IQR):** Presentes em `balance` (saldos extremos), `duration` (chamadas muito longas) e `pdays`. Tratados via `PowerTransformer + RobustScaler` no pipeline.

### Riscos e Limitações Documentados
1. **Data Leakage — `duration`:** A duração da ligação só é conhecida após o contato ocorrer. Usar essa feature em produção implicaria que o modelo só poderia ser aplicado *após* a ligação, não *antes*. Para uso operacional pré-chamada, um modelo sem `duration` seria mais adequado. No contexto desta avaliação acadêmica, a feature foi mantida para fins de comparação, com esta limitação documentada.
2. **Concept Drift:** Os dados são de uma campanha histórica (2008–2013). Em produção, o comportamento dos clientes pode mudar com o tempo — uma estratégia de re-treinamento periódico é necessária.
3. **Desbalanceamento Leve:** A proporção entre classes é relativamente equilibrada (~53/47), mas o uso do F1-score como métrica primária é mais adequado do que acurácia simples.
4. **Viés de Seleção:** Registros com chamadas muito curtas (`duration ≈ 0`) tendem a ser classificados como `no`. Isso reflete uma correlação causal, não apenas estatística.

### Pipeline de Pré-Processamento
```
Features Numéricas   → SimpleImputer(median) → PowerTransformer → RobustScaler
Features Categóricas → SimpleImputer(most_frequent) → OneHotEncoder(drop='first')
```
O `ColumnTransformer` é ajustado (`fit`) exclusivamente nos dados de treino e aplicado (`transform`) nos dados de teste, garantindo ausência de data leakage.

---

## 3. Experimentos Sistemáticos com MLflow

### Configuração Experimental
- **Experimento MLflow:** `bank_marketing_classification`
- **Validação cruzada:** `StratifiedKFold(n_splits=5, shuffle=True)`
- **Métrica primária:** F1-score (macro)
- **Split treino/teste:** 80/20 estratificado, `random_state=42`

### Resultados dos Experimentos

| Experimento | F1 | ROC-AUC | Precisão | Recall | CV-F1 Médio | Tempo (s) |
|-------------|-----|---------|----------|--------|-------------|-----------|
| baseline_gradient_boosting | 0.845 | 0.926 | 0.845 | 0.845 | 0.841±0.005 | 1.2 |
| baseline_random_forest | 0.847 | 0.917 | 0.847 | 0.847 | 0.842±0.005 | 0.7 |
| **baseline_svm** ⭐ | **0.853** | **0.919** | — | — | **0.844±0.006** | 9.1 |
| gb_com_pca | 0.818 | 0.902 | — | — | 0.830±0.007 | 4.7 |
| rf_com_pca | 0.836 | 0.903 | — | — | 0.833±0.004 | 2.0 |
| gb_com_lda | 0.821 | 0.903 | — | — | 0.820±0.006 | 0.5 |
| rf_com_lda | 0.817 | 0.899 | — | — | 0.815±0.006 | 0.7 |

*⭐ Champion selecionado por maior F1. Métricas completas em `reports/metrics_experiments.csv`. A interface do MLflow permite análise interativa de todos os experimentos.*

### Decisões Técnicas
- **Gradient Boosting** foi incluído como modelo primário por ter apresentado o melhor F1 no PD1 (0.866)
- **Random Forest** foi incluído como alternativa mais interpretável e com menor custo de inferência
- **SVM** foi incluído como baseline não-ensemble para referência
- Os hiperparâmetros foram mantidos fixos (configurados em `config/pipeline.yaml`) para isolar o impacto da redução de dimensionalidade como variável independente

---

## 4. Redução de Dimensionalidade

### Contexto
Após o pré-processamento, o `OneHotEncoder` expande as 9 features categóricas para aproximadamente 30–40 dimensões binárias, resultando em ~37–47 features totais. A redução de dimensionalidade foi avaliada como potencial estratégia para:
- Remover redundância entre features
- Reduzir overfitting
- Diminuir o custo computacional de modelos mais complexos

### Técnica 1: PCA (Análise de Componentes Principais)

**Tipo:** Não supervisionada — maximiza variância explicada independentemente do target.

**Configuração:** `n_components=0.95` — número de componentes suficiente para explicar 95% da variância total.

**Trade-offs:**
| Aspecto | Avaliação |
|---------|-----------|
| Interpretabilidade | Baixa — componentes são combinações lineares das features originais |
| Custo computacional | Moderado — etapa extra no pipeline, mas reduz a dimensionalidade para os classificadores |
| Aplicabilidade | Adequada para datasets com alta correlação entre features |
| Limitação | Não considera informação do target — pode descartar variância discriminativa |

Para este dataset, com features pouco correlacionadas entre si, o PCA não captura a estrutura discriminativa tão bem quanto técnicas supervisionadas.

### Técnica 2: LDA (Análise Discriminante Linear)

**Tipo:** Supervisionada — maximiza a separabilidade entre classes.

**Configuração:** `n_components=1` (fixo para classificação binária — máximo = n_classes − 1).

**Trade-offs:**
| Aspecto | Avaliação |
|---------|-----------|
| Interpretabilidade | Alta — único eixo que maximiza separação entre as classes |
| Custo computacional | Muito baixo na inferência — reduz tudo a 1 dimensão |
| Aplicabilidade | Ideal para classificação binária quando as classes são linearmente separáveis |
| Limitação | Assume distribuição gaussiana das features e covariâncias iguais entre classes |

O LDA produz uma representação altamente comprimida (1D) que, embora eficiente computacionalmente, pode perder nuances não-lineares que o Gradient Boosting captura nas features originais.

### Resultados Comparativos

| Configuração | F1 | ROC-AUC | Variação F1 vs baseline |
|---|---|---|---|
| GB baseline | 0.845 | 0.926 | — |
| GB + PCA (95%) | 0.818 | 0.902 | **−0.027** |
| GB + LDA (1 comp) | 0.821 | 0.903 | **−0.024** |
| RF baseline | 0.847 | 0.917 | — |
| RF + PCA (95%) | 0.836 | 0.903 | **−0.011** |
| RF + LDA (1 comp) | 0.817 | 0.899 | **−0.030** |

### Conclusão da Análise
- **PCA** reduziu o F1 do GB em 2,7 pontos percentuais. No Gradient Boosting — que se beneficia de interações não-lineares entre features — a projeção linear do PCA descarta variância que o modelo utilizaria para capturar padrões complexos.
- **LDA** comprimiu para 1 dimensão, causando perda similar (−2,4 para GB, −3,0 para RF). Apesar de ser supervisionado, a compressão para um único eixo linear é muito agressiva para um problema com 42 features de entrada.
- **Random Forest com PCA** apresentou a menor perda (−1,1 pp), sugerindo que o RF é mais robusto à compressão de dimensionalidade.
- **Conclusão:** Para este dataset, a redução de dimensionalidade não melhora a performance — as features pós-encoding são informativas e suficientemente não-redundantes para não se beneficiarem de compressão linear. Os baselines sem redução são superiores em todas as métricas avaliadas.

---

## 5. Modelo Champion Selecionado

O modelo champion é selecionado automaticamente pelo critério de maior F1-score no conjunto de teste, com o run_id persistido em `models/champion_run_id.txt`.

**Critério de seleção:** F1-score (métrica mais robusta para dados com algum desbalanceamento)

**Justificativa da escolha de F1 vs ROC-AUC:**
- ROC-AUC mede a capacidade discriminativa geral do modelo, mas não reflete o desempenho em um limiar específico
- F1-score reflete a performance operacional real (equilíbrio entre precisão e recall no limiar padrão)
- Para uso em campanha de marketing, onde tanto falsos positivos (custo) quanto falsos negativos (receita perdida) têm impacto direto, o F1 está mais alinhado ao objetivo de negócio

**Champion selecionado: `baseline_svm`**
- F1-score: **0.853** | ROC-AUC: **0.919** | CV-F1: 0.844±0.006
- run_id: `06b41c74913e44de9ff0f5a695f46535` (persistido em `models/champion_run_id.txt`)

**Justificativa:** O SVM com kernel RBF apresentou o maior F1 no conjunto de teste e validação cruzada estável (desvio padrão de ±0.006), indicando boa generalização. Apesar do maior tempo de treino (9,1s vs 1,2s do GB), o custo de inferência é comparável. A ausência de redução de dimensionalidade no champion confirma que as 42 features pós-encoding já contêm informação discriminativa suficiente — a compressão via PCA ou LDA resultou em perda de informação relevante para este problema.

---

## 6. Operacionalização

### Persistência do Modelo
- Todos os modelos são serializados via `mlflow.sklearn.log_model()` e associados a um `run_id` único
- O modelo champion é referenciado por `models/champion_run_id.txt`
- Carregamento via `mlflow.sklearn.load_model(f"runs:/{run_id}/model")` garante reprodutibilidade exata

### Interface de Inferência (Streamlit)
**Página 1 — Predição Individual:**
- Formulário completo com todos os 16 campos de entrada em português
- Predição em tempo real com probabilidade e classe prevista
- Cálculo do valor esperado de negócio por cliente
- Limiar de decisão configurável

**Página 2 — Monitoramento:**
- Tabela comparativa de todos os experimentos registrados
- Gráfico de barras F1 vs ROC-AUC por experimento
- Detecção de drift via KS-test (Kolmogorov-Smirnov)
- Visualização da distribuição de features: treino vs lote de produção simulado

### Monitoramento e Detecção de Drift
- **Método:** Teste de Kolmogorov-Smirnov entre distribuição de treino e lote de produção
- **Limiar de alerta:** p-valor < 0,05 (significância estatística)
- **Features monitoradas:** Todas as 7 features numéricas

### Estratégia de Re-Treinamento
1. **Gatilho de drift:** KS-test com p < 0,05 em ≥ 2 features numéricas consecutivamente
2. **Gatilho de degradação:** F1 em produção cai > 5% abaixo do baseline registrado
3. **Frequência mínima:** Re-treinamento semestral mesmo sem detecção de drift (modelos de marketing tendem a ter sazonalidade)
4. **Procedimento:** Re-executar `src/train.py` com dados atualizados → novo champion selecionado e registrado automaticamente no MLflow

---

## 7. Métricas de Negócio e Impacto

| Métrica | Definição | Monitoramento |
|---------|-----------|---------------|
| Custo por Contato | $5 por ligação realizada | Fixo |
| Receita por Conversão | $200 por depósito subscrito | Fixo |
| Valor Esperado por Cliente | `P(yes) × 200 − 5` | Calculado em tempo real |
| Taxa de Conversão Prevista | Proporção de clientes classificados como `yes` | Interface Streamlit |
| Precisão Operacional | Dos clientes contatados, quantos realmente converteram | Teste A/B pós-deploy |
| ROI da Campanha | `(conversões × receita − total_contatos × custo) / (total_contatos × custo)` | Pós-campanha |

**Exemplo:** Em uma campanha com 1.000 clientes selecionados pelo modelo com precisão de 70%:
- Conversões esperadas: 700
- Receita: $140.000
- Custo de campanha: $5.000
- **ROI estimado: 2.700%**

Sem modelo (taxa base ~53%): 530 conversões, receita $106.000, ROI: 2.020%.  
**Ganho do modelo: +$34.000 por campanha de 1.000 clientes.**
