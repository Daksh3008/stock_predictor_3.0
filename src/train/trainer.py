"""Training utilities: trainer loop with early stopping, checkpointing and optional MC-dropout for uncertainty.
Uses PyTorch.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, loss_fn=nn.MSELoss(), device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, train_dataset, val_dataset=None, epochs=20, batch_size=64,
            ckpt_path='models/checkpoint.pt', early_stop=5):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        best_val = np.inf
        patience = 0
        for ep in range(1, epochs + 1):
            self.model.train()
            losses = []
            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                preds, _ = self.model(X)
                loss = self.loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            train_loss = np.mean(losses)

            if val_dataset is not None:
                val_loss = self._evaluate(val_loader)
                print(f"Epoch {ep} Train MSE={train_loss:.6f} Val MSE={val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({'model_state': self.model.state_dict()}, ckpt_path)
                    patience = 0
                else:
                    patience += 1
                if patience >= early_stop:
                    print("Early stopping")
                    break
            else:
                print(f"Epoch {ep} Train MSE={train_loss:.6f}")
        return ckpt_path

    def _evaluate(self, loader):
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                out, _ = self.model(X)
                preds.append(out.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mse = mean_squared_error(trues, preds)
        return mse