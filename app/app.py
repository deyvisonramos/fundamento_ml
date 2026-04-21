"""
Interface Streamlit: Página 1 - Predição | Página 2 - Monitoramento e Drift
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy import stats

from inference import load_champion_model, predict_single

st.set_page_config(
    page_title="Bank Marketing - MLOps",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Mapeamentos PT-BR → valores originais do dataset
# ---------------------------------------------------------------------------
JOB_MAP = {
    "Administrativo": "admin.",
    "Operário": "blue-collar",
    "Empreendedor": "entrepreneur",
    "Empregada Doméstica": "housemaid",
    "Gerência": "management",
    "Aposentado(a)": "retired",
    "Autônomo": "self-employed",
    "Serviços": "services",
    "Estudante": "student",
    "Técnico": "technician",
    "Desempregado(a)": "unemployed",
    "Não Informado": "unknown",
}

MARITAL_MAP = {
    "Casado(a)": "married",
    "Solteiro(a)": "single",
    "Divorciado(a)": "divorced",
}

EDUCATION_MAP = {
    "Ensino Fundamental": "primary",
    "Ensino Médio": "secondary",
    "Ensino Superior": "tertiary",
    "Não Informado": "unknown",
}

SIM_NAO_MAP = {"Não": "no", "Sim": "yes"}

CONTACT_MAP = {
    "Celular": "cellular",
    "Telefone Fixo": "telephone",
    "Não Informado": "unknown",
}

MONTH_MAP = {
    "Janeiro": "jan",
    "Fevereiro": "feb",
    "Março": "mar",
    "Abril": "apr",
    "Maio": "may",
    "Junho": "jun",
    "Julho": "jul",
    "Agosto": "aug",
    "Setembro": "sep",
    "Outubro": "oct",
    "Novembro": "nov",
    "Dezembro": "dec",
}

POUTCOME_MAP = {
    "Não Informado": "unknown",
    "Sem Sucesso": "failure",
    "Outro": "other",
    "Bem-sucedido": "success",
}

FEATURE_LABELS = {
    "age": "Idade",
    "balance": "Saldo Médio Anual",
    "day": "Dia do Mês",
    "duration": "Duração da Chamada (seg)",
    "campaign": "Nº de Contatos na Campanha",
    "pdays": "Dias Desde Último Contato",
    "previous": "Contatos em Campanhas Anteriores",
}


@st.cache_resource(show_spinner="Carregando modelo champion...")
def get_model():
    config = load_config()
    return load_champion_model(config["mlflow"]["tracking_uri"])


@st.cache_data(show_spinner=False)
def load_train_data():
    config = load_config()
    df = pd.read_csv(config["data"]["raw_path"])
    return df


def load_config():
    with open("config/pipeline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics():
    path = Path("reports/metrics_experiments.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def page_prediction():
    st.title("🏦 Predição de Depósito Bancário")
    st.markdown(
        "Preencha os dados do cliente abaixo para obter a predição do modelo sobre a probabilidade de subscrição de um depósito a prazo."
    )

    config = load_config()
    model, run_id = get_model()

    with st.form("prediction_form"):
        st.subheader("Dados Demográficos")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Idade", 18, 95, 35)
            job_label = st.selectbox("Profissão", list(JOB_MAP.keys()))
            marital_label = st.selectbox("Estado Civil", list(MARITAL_MAP.keys()))
        with col2:
            education_label = st.selectbox("Escolaridade", list(EDUCATION_MAP.keys()))
            default_label = st.selectbox("Inadimplente?", list(SIM_NAO_MAP.keys()))
            housing_label = st.selectbox("Financiamento Habitacional?", list(SIM_NAO_MAP.keys()))
        with col3:
            loan_label = st.selectbox("Empréstimo Pessoal?", list(SIM_NAO_MAP.keys()))
            balance = st.number_input("Saldo Médio Anual (€)", value=1500, min_value=-10000, max_value=100000)

        st.subheader("Dados da Campanha")
        col4, col5, col6 = st.columns(3)
        with col4:
            contact_label = st.selectbox("Tipo de Contato", list(CONTACT_MAP.keys()))
            month_label = st.selectbox("Mês do Último Contato", list(MONTH_MAP.keys()))
            day = st.slider("Dia do Mês", 1, 31, 15)
        with col5:
            duration = st.slider(
                "Duração da Última Chamada (seg)",
                0, 4000, 300,
                help="⚠️ Informação disponível apenas após a ligação — não utilizar para decisões pré-chamada.",
            )
            campaign = st.slider("Nº de Contatos Nesta Campanha", 1, 50, 2)
        with col6:
            pdays = st.number_input(
                "Dias Desde o Último Contato (-1 = nunca contactado)",
                value=-1, min_value=-1, max_value=999,
            )
            previous = st.slider("Nº de Contatos em Campanhas Anteriores", 0, 50, 0)
            poutcome_label = st.selectbox("Resultado da Campanha Anterior", list(POUTCOME_MAP.keys()))

        threshold = st.slider(
            "Limiar de Decisão",
            0.1, 0.9, float(config["business_metrics"]["decision_threshold"]),
            step=0.05,
            help="Ajuste o equilíbrio entre precisão e recall conforme a estratégia da campanha.",
        )

        submitted = st.form_submit_button("Realizar Predição", use_container_width=True, type="primary")

    if submitted:
        input_dict = {
            "age": age,
            "balance": balance,
            "day": day,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "job": JOB_MAP[job_label],
            "marital": MARITAL_MAP[marital_label],
            "education": EDUCATION_MAP[education_label],
            "default": SIM_NAO_MAP[default_label],
            "housing": SIM_NAO_MAP[housing_label],
            "loan": SIM_NAO_MAP[loan_label],
            "contact": CONTACT_MAP[contact_label],
            "month": MONTH_MAP[month_label],
            "poutcome": POUTCOME_MAP[poutcome_label],
        }
        result = predict_single(model, input_dict)
        prob = result["probability_yes"]
        decision = "yes" if prob >= threshold else "no"

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            label = "✅ Vai subscrever" if decision == "yes" else "❌ Não vai subscrever"
            st.metric("Predição", label)
        with col_b:
            st.metric("Probabilidade de Depósito", f"{prob:.1%}")
        with col_c:
            cost = config["business_metrics"]["campaign_cost_per_contact"]
            revenue = config["business_metrics"]["revenue_per_conversion"]
            expected_value = prob * revenue - cost
            st.metric("Valor Esperado da Campanha", f"€{expected_value:.0f}")

        if decision == "yes":
            st.success(f"Recomendação: CONTATAR este cliente. Probabilidade estimada: {prob:.1%}")
        else:
            st.warning(f"Recomendação: Baixa prioridade para contato. Probabilidade estimada: {prob:.1%}")

        st.caption(f"Modelo champion: run_id `{run_id}` | Limiar aplicado: {threshold}")


def page_monitoring():
    st.title("📊 Monitoramento do Modelo")
    config = load_config()

    metrics_df = load_metrics()
    if metrics_df is not None:
        st.subheader("Comparação de Experimentos")
        cols = ["run_name", "test_f1", "test_roc_auc", "test_precision", "test_recall", "cv_f1_mean", "train_time_seconds"]
        display_df = metrics_df[cols].copy()
        display_df.columns = ["Experimento", "F1", "ROC-AUC", "Precisão", "Recall", "CV-F1 Médio", "Tempo de Treino (s)"]

        champion_idx = display_df["F1"].idxmax()

        def highlight_champion(row):
            return ["background-color: #d4edda" if row.name == champion_idx else "" for _ in row]

        st.dataframe(
            display_df.style.format({
                "F1": "{:.3f}", "ROC-AUC": "{:.3f}", "Precisão": "{:.3f}",
                "Recall": "{:.3f}", "CV-F1 Médio": "{:.3f}", "Tempo de Treino (s)": "{:.1f}",
            }).apply(highlight_champion, axis=1),
            use_container_width=True,
        )

        champion_row = display_df.loc[champion_idx]
        st.info(
            f"**Champion:** {champion_row['Experimento']} | F1={champion_row['F1']:.3f} | "
            f"ROC-AUC={champion_row['ROC-AUC']:.3f}"
        )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(display_df))
        width = 0.35
        ax.bar(x - width / 2, display_df["F1"], width, label="F1", color="steelblue")
        ax.bar(x + width / 2, display_df["ROC-AUC"], width, label="ROC-AUC", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(display_df["Experimento"], rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.legend()
        ax.set_title("F1 e ROC-AUC por Experimento")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Detecção de Desvio de Distribuição (KS-Test)")
    st.markdown(
        "Compare a distribuição de uma variável no conjunto de treino com um lote de produção simulado. "
        "O teste de Kolmogorov-Smirnov detecta se as distribuições são estatisticamente diferentes, "
        "indicando possível degradação do modelo por mudança nos dados de entrada."
    )

    df_train = load_train_data()
    numeric_features = config["features"]["numeric"]

    feature_options = {FEATURE_LABELS.get(f, f): f for f in numeric_features}

    col1, col2 = st.columns(2)
    with col1:
        feature_label = st.selectbox("Variável para análise de drift", list(feature_options.keys()))
        feature_to_test = feature_options[feature_label]
        batch_size = st.slider("Tamanho do lote simulado", 50, 500, 100)

    with col2:
        drift_magnitude = st.slider(
            "Magnitude do desvio simulado (desvios padrão)",
            0.0, 3.0, 0.0, step=0.25,
            help="0 = sem desvio (amostragem aleatória do treino). Valores maiores simulam drift artificial para fins de demonstração.",
        )

    if st.button("Executar Teste de Drift", use_container_width=True):
        train_dist = df_train[feature_to_test].dropna()
        rng = np.random.default_rng(42)

        if drift_magnitude == 0:
            batch_sample = rng.choice(train_dist.values, size=batch_size, replace=True)
        else:
            shift = drift_magnitude * train_dist.std()
            batch_sample = rng.choice(train_dist.values, size=batch_size, replace=True) + shift

        ks_stat, p_value = stats.ks_2samp(train_dist.values, batch_sample)
        alpha = 0.05
        has_drift = p_value < alpha

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Estatística KS", f"{ks_stat:.4f}")
        col_r2.metric("p-valor", f"{p_value:.4f}")
        col_r3.metric("Drift Detectado?", "⚠️ SIM" if has_drift else "✅ NÃO")

        if has_drift:
            st.error(
                f"Drift detectado em '{feature_label}' (p={p_value:.4f} < {alpha}). "
                "Recomenda-se re-treinar o modelo com dados mais recentes."
            )
        else:
            st.success(
                f"Nenhum drift significativo detectado em '{feature_label}' (p={p_value:.4f} ≥ {alpha})."
            )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(train_dist, bins=30, alpha=0.5, label="Treino", color="steelblue", density=True)
        ax.hist(batch_sample, bins=30, alpha=0.5, label=f"Lote de Produção (n={batch_size})", color="coral", density=True)
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Densidade")
        ax.set_title(f"Distribuição: {feature_label} — KS={ks_stat:.4f}, p={p_value:.4f}")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Métricas de Negócio")
    bm = config["business_metrics"]
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Custo por Contato", f"€{bm['campaign_cost_per_contact']:.0f}")
    col_b2.metric("Receita por Conversão", f"€{bm['revenue_per_conversion']:.0f}")
    col_b3.metric("Limiar de Decisão Padrão", f"{bm['decision_threshold']:.0%}")
    st.caption(
        "Estratégia de re-treinamento: monitorar KS-test semanalmente. "
        "Disparar re-treino se drift detectado em ≥ 2 variáveis ou se F1 em produção cair mais de 5% abaixo do baseline."
    )


pages = {
    "Predição": page_prediction,
    "Monitoramento": page_monitoring,
}

with st.sidebar:
    st.title("🏦 Bank Marketing MLOps")
    st.markdown("**Projeto de Pós-Graduação — MLOps**")
    st.divider()
    selected = st.radio("Navegação", list(pages.keys()))

pages[selected]()
