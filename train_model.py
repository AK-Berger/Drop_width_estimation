#!/usr/bin/env python3
import os
import warnings
import argparse
from pathlib import Path
import pandas as pd
from src.drop_width.preprocessing import DropDataProcessor
from src.drop_width.model import build_lstm_model

# Suppress TensorFlow and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for drop width estimation."
    )
    parser.add_argument(
        "--data-file", "-d",
        type=Path,
        default=Path("data/dataset.xlsx"),
        help="Path to the Excel dataset file."
    )
    parser.add_argument(
        "--model-output", "-m",
        type=Path,
        default=Path("models/model_lstm.h5"),
        help="Where to save the trained model (.h5)."
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=2500,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        "--window-size", "-s",
        type=int,
        default=20,
        help="Sliding window size."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading data from {args.data_file}...")
    df = pd.read_excel(args.data_file)
    processor = DropDataProcessor(df)
    train_data, _, train_label, _, _ = processor.extract_data_label()

    print("Slicing data...")
    train_data_sliced, train_label_sliced = processor.slice_patch(
        args.window_size, train_data, train_label
    )

    print("Building model...")
    model, X_train, X_val, y_train, y_val = build_lstm_model(
        train_data_sliced, train_label_sliced, args.window_size
    )

    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        shuffle=True,
        verbose=1
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)
    print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    main()
