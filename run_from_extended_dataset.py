from pathlib import Path
import pandas as pd

from src.data.build_dataset import build_orders_extended
from src.features.select_features import prepare_xy
from src.models.train_model import train_and_save_model
from src.models.evaluate import compute_classification_metrics
from src.monitoring.simulate_monthly import simular_nuevos_datos_mensuales


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def main():
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    master_path = processed_dir / "orders_extended_master.csv"

    if not master_path.exists():
        print("⚙️  Generando orders_extended_master.csv a partir de los CSV crudos...")
        master_path = build_orders_extended(raw_dir, processed_dir)
    else:
        print(f"✅ Usando master existente: {master_path}")

    df_master = pd.read_csv(master_path, parse_dates=["order_purchase_timestamp"])

    X_train, y_train, X_backtest, y_backtest, X_final, y_final = prepare_xy(df_master)

    model_path = models_dir / "cancel_model.joblib"
    print("\n🚀 Entrenando modelo XGBoost final...")
    clf = train_and_save_model(X_train, y_train, model_path, model_type="xgb")
    print(f"Modelo guardado en: {model_path}")

    print("\n📊 Métricas TRAIN:")
    print(compute_classification_metrics(clf, X_train, y_train))

    print("\n📊 Métricas BACKTEST:")
    print(compute_classification_metrics(clf, X_backtest, y_backtest))

    print("\n📊 Métricas FINAL TEST:")
    print(compute_classification_metrics(clf, X_final, y_final))

    print("\n📈 Simulación mensual y monitor...")
    monitor_mensual = simular_nuevos_datos_mensuales(
        df_base=df_master,
        pipeline=clf,
        target_col="order_canceled_extended",
        date_col="order_purchase_timestamp",
    )

    monitor_path = processed_dir / "monitor_mensual.csv"
    monitor_mensual.to_csv(monitor_path, index=False)
    print(f"Monitor mensual guardado en: {monitor_path}")


if __name__ == "__main__":
    main()
