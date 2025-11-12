"""
src/pipeline/train_model.py

Main CLI script for training the multivariate LSTM + Attention model.
Reads settings from config/default.yaml but allows CLI overrides for key params.

Usage:
    python -m src.pipeline.train_model --epochs 30 --seq_len 60 --pred_horizon 1
"""
import os
import sys
import argparse
import yaml
from src.train.experiment_runner import run_experiment

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")


def load_config(path=CONFIG_PATH):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train multivariate LSTM + Attention model")
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--seq_len", type=int, help="Sequence length (lookback window)")
    parser.add_argument("--pred_horizon", type=int, help="Days ahead to predict")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    args = parser.parse_args()

    # Load config
    cfg = load_config()
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    exp_cfg = cfg.get("experiment", {})

    # Apply CLI overrides if provided
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.seq_len:
        model_cfg["seq_len"] = args.seq_len
    if args.pred_horizon:
        model_cfg["pred_horizon"] = args.pred_horizon
    if args.lr:
        train_cfg["lr"] = args.lr

    print("=" * 80)
    print("📈 Starting Deepak Nitrite Multivariate LSTM Training")
    print("=" * 80)
    print(f"Sequence Length: {model_cfg['seq_len']} | Prediction Horizon: {model_cfg['pred_horizon']}")
    print(f"Epochs: {train_cfg['epochs']} | Batch Size: {train_cfg['batch_size']} | LR: {train_cfg['lr']}")
    print("Saving models to:", exp_cfg["save_dir"])
    print("-" * 80)

    try:
        result = run_experiment(
            processed_csv=os.path.join(cfg["data"]["processed_dir"], "feature_matrix.csv"),
            seq_len=model_cfg["seq_len"],
            pred_horizon=model_cfg["pred_horizon"],
            hidden=model_cfg["hidden_dim"],
            layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            use_attention=model_cfg["use_attention"],
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch_size"],
            lr=train_cfg["lr"],
            explain=True,
            save_dir=exp_cfg["save_dir"],
            shap_sample=exp_cfg.get("shap_sample", 200)
        )

        print("\n✅ Training completed successfully!")
        print("📊 Metrics Summary:")
        for k, v in result["metrics"].items():
            print(f"   {k:25s}: {v:.6f}")
        print("📜 Correlation Report:", result["report"])
        print("💾 Model Checkpoint  :", result["checkpoint"])
        print("=" * 80)

    except Exception as e:
        print("❌ Training failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
