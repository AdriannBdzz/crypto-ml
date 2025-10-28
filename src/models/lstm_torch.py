# opcional: ejemplo minimal PyTorch (no usado por defecto en API)
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from joblib import dump
from src.data.fetch_ccxt import fetch_ohlcv
from src.features.pipeline import make_features, make_labels
from src.utils.io import load_yaml, ensure_dir

class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len=48):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, i):
        return self.X[i:i+self.seq_len], self.y[i+self.seq_len]

class LSTMClassifier(nn.Module):
    def __init__(self, n_feats, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw = fetch_ohlcv(cfg["data"]["symbol"], cfg["data"]["exchange_id"], cfg["data"]["timeframe"], cfg["data"]["limit"])
    feat = make_features(raw, **cfg["features"])
    X, y = make_labels(feat, cfg["label"]["horizon"])

    ds = SeqDataset(X, y, args.seq_len)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    model = LSTMClassifier(n_feats=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(args.epochs):
        loss_ep = 0.0
        for xb, yb in dl:
            opt.zero_grad(); logits = model(xb); loss = crit(logits, yb); loss.backward(); opt.step()
            loss_ep += loss.item()
        print(f"epoch {ep+1}: loss {loss_ep/len(dl):.4f}")

    ensure_dir("artifacts")
    torch.save(model.state_dict(), "artifacts/lstm_cls.pt")
    dump({"features": list(X.columns)}, "artifacts/lstm_meta.joblib")

if __name__ == "__main__":
    main()
