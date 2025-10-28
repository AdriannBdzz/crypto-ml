# 📘 README.md — CryptoML Predictor  

### Predicción de precios de criptomonedas con Machine Learning y Validación Walk-Forward  
**Autor:** _[Tu nombre]_  
**Tecnologías:** Python · FastAPI · scikit-learn · ccxt · pandas · Optuna · PyTorch (futuro)

---

## 🧠 Descripción general

**CryptoML Predictor** es una aplicación de *machine learning* diseñada para **predecir la evolución de precios de criptomonedas** (como BTC/USDT) a partir de datos históricos de mercado y variables técnicas, macro y de sentimiento.  

El proyecto integra:
- Un **pipeline completo de ML para series temporales financieras**,  
- Validación **walk-forward purged + embargo** (anti-overfitting),  
- **API REST en FastAPI** para predicciones en tiempo real,  
- Soporte para **reentrenamiento automático** y evaluación de drift,  
- Arquitectura modular lista para producción o despliegue en la nube.

---

## ⚙️ Arquitectura del proyecto

```
crypto_ml/
│
├── configs/              # Configuración YAML (símbolo, horizonte, modelo, etc.)
├── src/
│   ├── data/             # Recolección de datos (ccxt, Glassnode, etc.)
│   │   └── fetch_ccxt.py
│   ├── features/         # Generación y selección de features
│   │   ├── pipeline.py
│   │   ├── selection.py
│   │   └── importance.py
│   ├── models/           # Entrenamiento de modelos ML
│   │   ├── train_rf.py   # Random Forest + Walk-Forward Validation
│   │   └── train_lr.py   # (Opcional) Logistic Regression / XGBoost
│   ├── eval/             # Evaluación fuera de muestra
│   │   └── evaluate_rf.py
│   ├── serve/            # API REST con FastAPI
│   │   └── api.py
│   └── utils/            # Funciones auxiliares y validación
│
├── artifacts/            # Modelos y reportes generados (joblib + JSON)
├── requirements-win.txt  # Dependencias
└── README.md
```

---

## 🚀 Flujo de trabajo

### 1️⃣ Recolección de datos
- Se utilizan APIs públicas (Binance vía **ccxt**).  
- Paginación para obtener hasta **10.000 velas históricas** por símbolo y timeframe.  
- Estructura OHLCV (open, high, low, close, volume).  

```python
from src.data.fetch_ccxt import fetch_ohlcv
df = fetch_ohlcv(symbol="BTC/USDT", exchange_id="binance", timeframe="1h", max_candles=5000)
```

---

### 2️⃣ Generación de features
- Indicadores técnicos: RSI, MACD, medias móviles, spreads.  
- Volatilidad realizada (1D y 7D).  
- Volumen relativo.  
- Limpieza y *shift* temporal para evitar **data leakage**.  

```python
from src.features.pipeline import make_features
features = make_features(df, decision_at="close")
```

---

### 3️⃣ Etiquetado y horizonte de predicción
- Clasificación direccional:  
  \( y_t = 1 \text{ si } \log(\frac{P_{t+h}}{P_t}) > 0 \)  
- Horizonte configurable (`horizon=6` horas por defecto).

---

### 4️⃣ Entrenamiento y validación

#### ✅ Validación Walk-Forward con Purge + Embargo  
Simula predicción real, entrenando en una ventana y evaluando en la siguiente, eliminando fugas temporales.

```bash
python -m src.models.train_rf   --symbol BTC/USDT   --exchange binance   --timeframe 1h   --horizon 6   --train_size 1000   --test_size 200   --step 200   --embargo 6
```

📊 Salida típica:

```
[fold 01] acc=0.53 mcc=0.12 auc=0.59 brier=0.27 | sharpe=0.30
[fold 02] acc=0.48 mcc=0.08 auc=0.54 brier=0.26 | sharpe=0.52
...
```

🧾 Reporte JSON en `artifacts/wfv_report_binance_BTC-USDT_1h_6h.json`.

---

### 5️⃣ Evaluación fuera de muestra

```bash
python -m src.eval.evaluate_rf --symbol BTC/USDT --exchange binance --timeframe 1h --horizon 6
```

Métricas incluidas:
- Accuracy, F1, MCC, AUC, Brier  
- Sharpe-like ratio y max drawdown  
- KS-test de drift por feature  

---

### 6️⃣ API REST en FastAPI

Arranca el servicio:

```bash
uvicorn src.serve.api:app --reload --port 8000
```

Endpoints:
- `GET /health` → Estado del modelo, features, meta.  
- `GET /predict` → Predicción en vivo:
  ```json
  {
    "symbol": "BTC/USDT",
    "prob_up": 0.57,
    "prediction": 1,
    "latest_price": 61200.0
  }
  ```
- `GET /docs` → Swagger UI interactivo.  

---

## 📈 Próximas mejoras

- Integrar **Optuna** para ajuste automático de hiperparámetros.  
- Añadir modelos alternativos (XGBoost, LightGBM, LSTM, Transformers).  
- Incorporar **sentimiento social** (Twitter, Google Trends) y **factores macroeconómicos**.  
- Implementar **reentrenamiento programado** y **detección de drift** continua.  
- Conexión opcional con **bots de trading** vía websockets.  

---

## 💡 Principales aprendizajes técnicos

- ✅ Cómo estructurar un proyecto de ML financiero **modular y escalable**.  
- ✅ Evitar *data leakage* (purge, embargo, shift, scaling aislado).  
- ✅ Validar correctamente con **Walk-Forward Validation**.  
- ✅ Calibrar modelos para probabilidades realistas (Brier, Sharpe).  
- ✅ Integrar modelo en **API de predicción en tiempo real**.  

---

## 🧩 Stack tecnológico

| Componente | Tecnología |
|-------------|-------------|
| **Lenguaje** | Python 3.11+ |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Validación avanzada** | custom Walk-Forward purged CV |
| **APIs de mercado** | ccxt (Binance, etc.) |
| **Backend** | FastAPI + Uvicorn |
| **Visualización / análisis** | matplotlib, seaborn |
| **Optimización (futuro)** | Optuna |
| **Infraestructura (sugerida)** | AWS Lambda / EC2 / GCP / Heroku |

---

## 📂 Resultados ejemplo

| Métrica | Promedio (19 folds) |
|----------|--------------------|
| Accuracy | 0.52 |
| MCC | 0.10 |
| AUC | 0.58 |
| Brier | 0.27 |
| Sharpe-like | ≈ 0.25 |

📉 **Conclusión:**  
El modelo ofrece una señal **débil pero consistente**, sin sobreajuste y con buena estabilidad temporal.  
Base sólida para incorporar señales más ricas (macro, on-chain, social).

---

## 📜 Licencia

MIT License – uso libre con atribución.
