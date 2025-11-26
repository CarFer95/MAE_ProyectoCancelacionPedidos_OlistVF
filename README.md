# Proyecto: Predicción de Cancelaciones Olist

Pipeline modular para:
1. Construir un dataset extendido de pedidos (`orders_extended_master.csv`).
2. Entrenar un modelo de clasificación (XGBoost) para predecir cancelaciones.
3. Evaluar el desempeño en train, backtest y test final.
4. Simular el desempeño mensual del modelo (`monitor_mensual.csv`).

## Estructura

- `data/raw/`: CSV originales de Olist.
- `data/processed/`: datasets generados.
- `src/`: código fuente (data, features, models, monitoring).
- `models/`: modelos entrenados.
- `notebooks/`: notebooks de exploración.
- `docs/`: documentación.

## Ejecución

1. Copiar los CSV de Olist en `data/raw/`.
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar:

   ```bash
   python run_from_extended_dataset.py
   ```
