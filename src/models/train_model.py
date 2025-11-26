from pathlib import Path
import joblib
import pandas as pd

from src.models.pipeline import create_model_pipeline, ModelType


def train_and_save_model(
    X_train: pd.DataFrame,
    y_train,
    model_path: str | Path,
    model_type: ModelType = "xgb",
):
    """Entrena el modelo y guarda el pipeline completo."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    clf = create_model_pipeline(X_train, model_type=model_type)
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    return clf
