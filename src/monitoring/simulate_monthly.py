from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def simular_nuevos_datos_mensuales(
    df_base: pd.DataFrame,
    pipeline,
    target_col: str,
    date_col: str = "order_purchase_timestamp",
    features_drop: list[str] | None = None,
) -> pd.DataFrame:
    """Simula el desempeño mensual del modelo entrenado."""
    df = df_base.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "purchase_ym" not in df.columns:
        df["purchase_ym"] = df[date_col].dt.to_period("M").astype(str).str.replace("-", "")

    if features_drop is None:
        features_drop = [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "is_canceled_strict",
            target_col,
        ]

    resultados = []

    for mes, df_mes in df.groupby("purchase_ym"):
        X_mes = df_mes.drop(columns=[*features_drop, "purchase_ym"], errors="ignore")
        y_mes = df_mes[target_col].astype(int)

        y_pred = pipeline.predict(X_mes)
        y_proba = pipeline.predict_proba(X_mes)[:, 1]

        cancel_rate_real = y_mes.mean()
        cancel_rate_pred = (y_pred == 1).mean()

        acc = accuracy_score(y_mes, y_pred)
        prec = precision_score(y_mes, y_pred, zero_division=0)
        rec = recall_score(y_mes, y_pred, zero_division=0)
        f1 = f1_score(y_mes, y_pred, zero_division=0)

        resultados.append({
            "purchase_ym": mes,
            "n_pedidos": len(df_mes),
            "cancel_rate_real": cancel_rate_real,
            "cancel_rate_predicha": cancel_rate_pred,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    monitor_mensual = pd.DataFrame(resultados).sort_values("purchase_ym")
    return monitor_mensual
