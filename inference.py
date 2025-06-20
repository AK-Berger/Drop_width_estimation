#!/usr/bin/env python3
import os
import warnings
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow messages and deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Your project imports
from src.drop_width.preprocessing import DropDataProcessor
from src.drop_width.model import build_lstm_model
from src.drop_width.side_to_width import width_estim

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate drop widths from video frames using a pre-trained LSTM model."
    )
    parser.add_argument(
        "--data-file", "-d",
        type=Path,
        default=Path("data/dataset.xlsx"),
        help="Path to the Excel dataset file."
    )
    parser.add_argument(
        "--weights-file", "-w",
        type=Path,
        default=Path("models/lstm_weights.h5"),
        help="Path to the pre-trained LSTM weights (.h5)."
    )
    parser.add_argument(
        "--video-id", "-v",
        type=int,
        required=True,
        help="Video ID to filter the dataset for inference."
    )
    parser.add_argument(
        "--window-size", "-s",
        type=int,
        default=20,
        help="Sliding window size for slicing the time series."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="If provided, save the plot to this path (PNG). Otherwise, display it."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and preprocess
    df = pd.read_excel(args.data_file)
    processor = DropDataProcessor(df)
    train_data, _, train_label, _, train_data_notnormalized = processor.extract_data_label()

    # Slice with sliding window
    train_data_sliced, train_label_sliced = processor.slice_patch(
        args.window_size, train_data, train_label
    )

    # Build and load model
    model, _, _, _, _ = build_lstm_model(
        train_data_sliced, train_label_sliced, args.window_size
    )
    model.load_weights(args.weights_file)

    # Filter for chosen video
    input_data = df[df["Video ID"] == args.video_id]
    if input_data.empty:
        raise ValueError(f"No data found for Video ID {args.video_id}")

    # Run inference
    y_pred, measured_width = width_estim(
        input_data, model, processor, train_data_notnormalized, args.window_size
    )

    # Align and store prediction
    y_pred_trimmed = y_pred[:-1]
    start = args.window_size // 2
    end = -args.window_size // 2 if args.window_size % 2 == 0 else -(args.window_size // 2)
    input_data_trimmed = input_data.iloc[start:end].copy()
    input_data_trimmed["Estimated Width"] = y_pred_trimmed
    input_data_trimmed["Drop width (cm)"]=measured_width

    # Save to Excel
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    excel_path = output_dir / f"{args.data_file.stem}_{args.video_id}.xlsx"
    input_data_trimmed.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    # Plotting configuration
    plt.rcParams["figure.figsize"] = (20, 5)
    font_size = 32

    # Plot the data
    plt.xlabel('Frames', fontdict={'size': font_size})
    plt.ylabel('Width (μm)', fontdict={'size': font_size})
    plt.plot(y_pred_trimmed, color='red', linestyle='dotted', linewidth=5, label="Estimated")
    plt.plot(measured_width, "black", linewidth=5, label="Measured")
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.legend(fontsize=24, loc='upper left')
    plt.ylim(3550, 4200)
    plt.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
