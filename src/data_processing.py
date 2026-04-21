"""
Módulo de fundação de dados: ingestão, diagnóstico de qualidade e construção do pipeline de pré-processamento.
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/pipeline.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Dataset carregado: %d linhas x %d colunas", *df.shape)
    logger.info("Tipos de dados:\n%s", df.dtypes.to_string())
    return df


def diagnose_quality(df: pd.DataFrame, config: dict) -> dict:
    """Diagnostica qualidade do dataset e retorna dicionário com os resultados."""
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    target = config["data"]["target"]

    report = {}

    # Valores nulos
    null_counts = df.isnull().sum()
    report["missing_values"] = null_counts[null_counts > 0].to_dict()

    # Duplicatas
    n_dup = df.duplicated().sum()
    report["duplicate_rows"] = int(n_dup)

    # Distribuição do target
    target_dist = df[target].value_counts(normalize=True).round(3).to_dict()
    report["target_distribution"] = target_dist

    # Outliers via IQR para features numéricas
    outlier_counts = {}
    for col in numeric_features:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if n_out > 0:
                outlier_counts[col] = n_out
    report["outlier_counts_iqr"] = outlier_counts

    # Estatísticas descritivas numéricas
    report["numeric_stats"] = df[numeric_features].describe().round(3).to_dict()

    # Cardinalidade das categóricas
    report["categorical_cardinality"] = {
        col: int(df[col].nunique()) for col in categorical_features if col in df.columns
    }

    # Riscos documentados
    report["documented_risks"] = [
        "Feature 'duration': disponível apenas após a ligação — usar com cautela (leakage operacional).",
        "Feature 'pdays': valor -1 codifica ausência de contato anterior — pode ser tratado como categoria separada.",
        "Desbalanceamento leve entre classes (~53% yes vs ~47% no) — monitorar F1 macro.",
        "Dados de campanha de marketing histórica — distribuição pode divergir em produção (concept drift).",
    ]

    # Log do relatório
    logger.info("=== DIAGNÓSTICO DE QUALIDADE ===")
    logger.info("Valores nulos: %s", report["missing_values"] or "Nenhum")
    logger.info("Linhas duplicadas: %d", report["duplicate_rows"])
    logger.info("Distribuição do target: %s", report["target_distribution"])
    logger.info("Outliers detectados (IQR): %s", report["outlier_counts_iqr"])
    logger.info("Riscos documentados:")
    for risk in report["documented_risks"]:
        logger.info("  - %s", risk)

    return report


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer de pré-processamento.
    Deve ser ajustado (fit) apenas nos dados de treino para evitar data leakage.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.extend([
            ("power", PowerTransformer()),
            ("scaler", RobustScaler()),
        ])
    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def prepare_splits(df: pd.DataFrame, config: dict):
    """Divide o dataset em treino/teste com estratificação."""
    from sklearn.model_selection import train_test_split

    target = config["data"]["target"]
    positive_label = config["data"]["positive_label"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X = df.drop(columns=[target])
    y = (df[target] == positive_label).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        "Split: treino=%d | teste=%d | positivos treino=%.1f%%",
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    config = load_config()
    df = load_data(config["data"]["raw_path"])
    report = diagnose_quality(df, config)
    X_train, X_test, y_train, y_test = prepare_splits(df, config)

    preprocessor = build_preprocessor(
        config["features"]["numeric"],
        config["features"]["categorical"],
    )
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    logger.info(
        "Pré-processamento concluído: treino=%s | teste=%s",
        X_train_proc.shape,
        X_test_proc.shape,
    )
