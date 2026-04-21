"""
Módulo de inferência: carrega o modelo champion do MLflow e executa predições.
"""
import logging
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/pipeline.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_champion_model(tracking_uri: str = "mlruns", champion_run_id: str | None = None):
    """Carrega o modelo champion registrado no MLflow."""
    mlflow.set_tracking_uri(tracking_uri)

    if champion_run_id is None:
        champion_path = Path("models/champion_run_id.txt")
        if not champion_path.exists():
            raise FileNotFoundError(
                "models/champion_run_id.txt não encontrado. Execute src/train.py primeiro."
            )
        champion_run_id = champion_path.read_text().strip()

    model_uri = f"runs:/{champion_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Modelo carregado: run_id=%s", champion_run_id)
    return model, champion_run_id


def predict(model, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Executa predição sobre um DataFrame com as features brutas.
    Retorna DataFrame com colunas: predicted_class, probability_yes.
    """
    y_pred = model.predict(input_data)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(input_data)[:, 1]
    else:
        import numpy as np
        scores = model.decision_function(input_data)
        y_prob = 1.0 / (1.0 + np.exp(-scores))

    return pd.DataFrame({
        "predicted_class": ["yes" if p == 1 else "no" for p in y_pred],
        "probability_yes": y_prob.round(4),
    })


def predict_single(model, input_dict: dict) -> dict:
    """Converte um dicionário de features em predição única."""
    df = pd.DataFrame([input_dict])
    result = predict(model, df)
    return {
        "predicted_class": result["predicted_class"].iloc[0],
        "probability_yes": float(result["probability_yes"].iloc[0]),
    }


if __name__ == "__main__":
    config = load_config()
    model, run_id = load_champion_model(config["mlflow"]["tracking_uri"])

    # Exemplo de predição com valores típicos
    example = {
        "age": 35, "balance": 1500, "day": 15, "duration": 300,
        "campaign": 2, "pdays": -1, "previous": 0,
        "job": "management", "marital": "married", "education": "tertiary",
        "default": "no", "housing": "yes", "loan": "no",
        "contact": "cellular", "month": "may", "poutcome": "unknown",
    }
    result = predict_single(model, example)
    print(f"Predição: {result['predicted_class']} | Probabilidade: {result['probability_yes']:.1%}")
