"""
Treinamento sistemático com MLflow: baselines + redução de dimensionalidade (PCA e LDA).
Registra parâmetros, métricas e artefatos para cada experimento.
"""
import logging
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import tempfile
import yaml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from data_processing import build_preprocessor, load_config, load_data, prepare_splits

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def save_confusion_matrix_plot(y_true, y_pred, run_name: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(run_name)
    tmpdir = tempfile.gettempdir()
    path = str(Path(tmpdir) / f"cm_{run_name.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


def save_comparison_plot(results: list[dict]) -> str:
    names = [r["run_name"] for r in results]
    f1s = [r["test_f1"] for r in results]
    aucs = [r["test_roc_auc"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, f1s, width, label="F1", color="steelblue")
    bars2 = ax.bar(x + width / 2, aucs, width, label="ROC-AUC", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Experimentos: F1 e ROC-AUC")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    tmpdir = tempfile.gettempdir()
    path = str(Path(tmpdir) / "comparacao_experimentos.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def run_experiment(
    run_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    extra_params: dict,
    cv_config: dict,
) -> dict:
    """Executa um experimento, loga no MLflow e retorna métricas."""
    with mlflow.start_run(run_name=run_name):
        # Parâmetros
        mlflow.log_params(extra_params)

        # Cross-validation no treino
        cv = StratifiedKFold(
            n_splits=cv_config["n_splits"],
            shuffle=cv_config["shuffle"],
            random_state=42,
        )
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=cv_config["scoring"], n_jobs=1
        )
        mlflow.log_metric("cv_f1_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_std", float(cv_scores.std()))

        # Treino final
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        mlflow.log_metric("train_time_seconds", round(train_time, 3))

        # Métricas de teste
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = pipeline.decision_function(X_test)
            y_score = 1.0 / (1.0 + np.exp(-y_score))

        metrics = compute_metrics(y_test, y_pred, y_score)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Artefato: confusion matrix
        cm_path = save_confusion_matrix_plot(y_test, y_pred, run_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Modelo serializado
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

        logger.info(
            "[%s] F1=%.3f | ROC-AUC=%.3f | CV-F1=%.3f±%.3f | tempo=%.1fs",
            run_name,
            metrics["f1"],
            metrics["roc_auc"],
            cv_scores.mean(),
            cv_scores.std(),
            train_time,
        )

    return {
        "run_name": run_name,
        "run_id": run_id,
        "test_f1": metrics["f1"],
        "test_roc_auc": metrics["roc_auc"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_accuracy": metrics["accuracy"],
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "train_time_seconds": round(train_time, 3),
    }


def build_experiments(
    config: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[dict]:
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    cv_config = config["cross_validation"]
    mc = config["models"]
    dr = config["dimensionality_reduction"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    gb_params = mc["gradient_boosting"]
    rf_params = mc["random_forest"]
    svm_params = mc["svm"]

    gb_clf = GradientBoostingClassifier(
        n_estimators=gb_params["n_estimators"],
        learning_rate=gb_params["learning_rate"],
        max_depth=gb_params["max_depth"],
        subsample=gb_params["subsample"],
        min_samples_leaf=gb_params["min_samples_leaf"],
        random_state=gb_params["random_state"],
    )
    rf_clf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        class_weight=rf_params["class_weight"],
        random_state=rf_params["random_state"],
    )
    svm_clf = SVC(
        kernel=svm_params["kernel"],
        C=svm_params["C"],
        class_weight=svm_params["class_weight"],
        probability=svm_params["probability"],
    )

    pca = PCA(n_components=dr["pca"]["variance_threshold"], random_state=42)
    lda = LDA(n_components=dr["lda"]["n_components"])

    experiments = [
        # --- Baselines ---
        {
            "run_name": "baseline_gradient_boosting",
            "pipeline": Pipeline([("preprocessor", build_preprocessor(numeric_features, categorical_features)), ("gb", gb_clf)]),
            "extra_params": {
                "model": "GradientBoosting",
                "dim_reduction": "none",
                **{f"gb_{k}": v for k, v in gb_params.items()},
            },
        },
        {
            "run_name": "baseline_random_forest",
            "pipeline": Pipeline([("preprocessor", build_preprocessor(numeric_features, categorical_features)), ("rf", rf_clf)]),
            "extra_params": {
                "model": "RandomForest",
                "dim_reduction": "none",
                **{f"rf_{k}": v for k, v in rf_params.items()},
            },
        },
        {
            "run_name": "baseline_svm",
            "pipeline": Pipeline([("preprocessor", build_preprocessor(numeric_features, categorical_features)), ("svm", svm_clf)]),
            "extra_params": {
                "model": "SVM",
                "dim_reduction": "none",
                **{f"svm_{k}": v for k, v in svm_params.items()},
            },
        },
        # --- PCA ---
        {
            "run_name": "gb_com_pca",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("pca", PCA(n_components=dr["pca"]["variance_threshold"], random_state=42)),
                ("gb", GradientBoostingClassifier(**{k: v for k, v in gb_params.items()})),
            ]),
            "extra_params": {
                "model": "GradientBoosting",
                "dim_reduction": "PCA",
                "pca_variance_threshold": dr["pca"]["variance_threshold"],
                **{f"gb_{k}": v for k, v in gb_params.items()},
            },
        },
        {
            "run_name": "rf_com_pca",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("pca", PCA(n_components=dr["pca"]["variance_threshold"], random_state=42)),
                ("rf", RandomForestClassifier(**{k: v for k, v in rf_params.items()})),
            ]),
            "extra_params": {
                "model": "RandomForest",
                "dim_reduction": "PCA",
                "pca_variance_threshold": dr["pca"]["variance_threshold"],
                **{f"rf_{k}": v for k, v in rf_params.items()},
            },
        },
        # --- LDA ---
        {
            "run_name": "gb_com_lda",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("lda", LDA(n_components=dr["lda"]["n_components"])),
                ("gb", GradientBoostingClassifier(**{k: v for k, v in gb_params.items()})),
            ]),
            "extra_params": {
                "model": "GradientBoosting",
                "dim_reduction": "LDA",
                "lda_n_components": dr["lda"]["n_components"],
                **{f"gb_{k}": v for k, v in gb_params.items()},
            },
        },
        {
            "run_name": "rf_com_lda",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("lda", LDA(n_components=dr["lda"]["n_components"])),
                ("rf", RandomForestClassifier(**{k: v for k, v in rf_params.items()})),
            ]),
            "extra_params": {
                "model": "RandomForest",
                "dim_reduction": "LDA",
                "lda_n_components": dr["lda"]["n_components"],
                **{f"rf_{k}": v for k, v in rf_params.items()},
            },
        },
    ]

    results = []
    for exp in experiments:
        logger.info("Executando: %s", exp["run_name"])
        result = run_experiment(
            run_name=exp["run_name"],
            pipeline=exp["pipeline"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            extra_params=exp["extra_params"],
            cv_config=cv_config,
        )
        results.append(result)

    return results


def select_champion(results: list[dict], config: dict) -> str:
    """Seleciona o melhor modelo por F1 e persiste o run_id."""
    champion = max(results, key=lambda r: r["test_f1"])
    champion_path = MODELS_DIR / "champion_run_id.txt"
    champion_path.write_text(champion["run_id"])
    logger.info(
        "Champion: %s | F1=%.3f | ROC-AUC=%.3f | run_id=%s",
        champion["run_name"],
        champion["test_f1"],
        champion["test_roc_auc"],
        champion["run_id"],
    )
    return champion["run_id"]


def save_metrics_csv(results: list[dict]):
    df = pd.DataFrame(results)
    path = Path("reports/metrics_experiments.csv")
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Métricas salvas em %s", path)
    print("\n" + df[["run_name", "test_f1", "test_roc_auc", "cv_f1_mean", "train_time_seconds"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    config = load_config()
    df = load_data(config["data"]["raw_path"])
    X_train, X_test, y_train, y_test = prepare_splits(df, config)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Iniciando %d experimentos MLflow...", 7)
    results = build_experiments(config, X_train, X_test, y_train, y_test)

    champion_run_id = select_champion(results, config)
    save_metrics_csv(results)

    # Gráfico comparativo como artefato extra
    plot_path = save_comparison_plot(results)
    with mlflow.start_run(run_name="comparacao_final"):
        mlflow.log_artifact(plot_path, artifact_path="plots")
        for r in results:
            mlflow.log_metric(f"{r['run_name']}_f1", r["test_f1"])
        mlflow.log_param("champion_run_id", champion_run_id)
    logger.info("Gráfico comparativo salvo em %s", plot_path)
    logger.info("Treinamento concluído. Execute 'mlflow ui' para ver os experimentos.")
