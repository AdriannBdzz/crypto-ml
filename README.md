# Crypto ML Predictor

Pipeline de ML para predicción direccional de criptomonedas con:
- Ingesta OHLCV vía CCXT
- Feature engineering reproducible
- Entrenamiento (RandomForest, baseline Prophet opcional, LSTM PyTorch opcional)
- Backtesting simple
- API FastAPI para predicciones en tiempo real
- Docker para despliegue

## Rápido inicio

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.models.train_rf  # entrena y guarda artifacts/
uvicorn src.serve.api:app --reload
