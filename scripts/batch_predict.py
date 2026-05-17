#!/usr/bin/env python3
"""
Batch Prediction Script
Process multiple predictions from a CSV file
"""

import pandas as pd
import joblib
import sys
import argparse
from pathlib import Path


def batch_predict(input_file, output_file, model_path="models/best_model.pkl"):
    """Run batch predictions"""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    required_cols = ['Payload Mass (kg)', 'Year', 'Grid_Fins', 'Legs',
                    'Booster Version', 'Launch Site']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    print(f"Making predictions for {len(df)} records...")
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    df['Prediction'] = ['Success' if p == 1 else 'Failure' for p in predictions]
    df['Success_Probability'] = probabilities[:, 1]
    df['Failure_Probability'] = probabilities[:, 0]

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    print("\n📊 Summary:")
    print(f"  Total: {len(df)}")
    print(f"  Success: {(predictions == 1).sum()}")
    print(f"  Failure: {(predictions == 0).sum()}")


def main():
    parser = argparse.ArgumentParser(description='Batch prediction script')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('-o', '--output', default='predictions.csv',
                       help='Output CSV file')
    parser.add_argument('-m', '--model', default='models/best_model.pkl',
                       help='Model file path')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)

    batch_predict(args.input, args.output, args.model)


if __name__ == "__main__":
    main()