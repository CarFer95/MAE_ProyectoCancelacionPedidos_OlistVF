import pandas as pd
import numpy as np


def create_features(
    df: pd.DataFrame,
    order_items: pd.DataFrame,
    geolocation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aplica el feature engineering sobre la tabla master.

    Aquí debes completar con todas las transformaciones de tu notebook.
    """
    df = df.copy()

    # 1) Features temporales de compra
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_week"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["is_weekend_purchase"] = df["purchase_weekday"].isin([5, 6]).astype(int)

    # 2) Diferencias de fechas (ejemplos)
    if "order_delivered_customer_date" in df.columns:
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.days

    if "order_approved_at" in df.columns:
        df["approval_delay_hours"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 3600.0

    # 3) Features de monto
    if "payment_value" in df.columns:
        df["log_payment_value"] = np.log1p(df["payment_value"])

    if "price" in df.columns:
        df["log_price"] = np.log1p(df["price"])

    # 4) Historial del cliente
    if "customer_unique_id" in df.columns and "order_id" in df.columns:
        df["customer_num_orders"] = (
            df.groupby("customer_unique_id")["order_id"]
            .transform("nunique")
        )

    # Aquí agrega TODAS las demás features que ya tienes en tu notebook.

    return df
