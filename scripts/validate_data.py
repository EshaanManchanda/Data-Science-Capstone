#!/usr/bin/env python3
"""
Data Validation Script
Validates input data for predictions
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime


def validate_data(data, rules):
    """Validate data against rules"""
    errors = []
    warnings = []

    for field, rule in rules.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue

        value = data[field]

        if 'required' in rule and rule['required'] and pd.isna(value):
            errors.append(f"Required field is empty: {field}")

        if 'type' in rule:
            expected_type = rule['type']
            if expected_type == 'number' and not isinstance(value, (int, float)):
                try:
                    float(value)
                except:
                    errors.append(f"Field must be a number: {field}")

        if 'min' in rule and isinstance(value, (int, float)):
            if value < rule['min']:
                errors.append(f"Value for {field} below minimum: {rule['min']}")

        if 'max' in rule and isinstance(value, (int, float)):
            if value > rule['max']:
                errors.append(f"Value for {field} above maximum: {rule['max']}")

        if 'choices' in rule and value not in rule['choices']:
            errors.append(f"Invalid choice for {field}: {value}. Must be one of {rule['choices']}")

    return errors, warnings


def validate_csv(filepath):
    """Validate a CSV file"""
    print(f"Validating {filepath}...")

    rules = {
        'Payload Mass (kg)': {
            'required': True,
            'type': 'number',
            'min': 0,
            'max': 10000
        },
        'Year': {
            'required': True,
            'type': 'number',
            'min': 2010,
            'max': 2030
        },
        'Grid_Fins': {
            'type': 'number',
            'min': 0,
            'max': 1
        },
        'Legs': {
            'type': 'number',
            'min': 0,
            'max': 1
        },
        'Booster Version': {
            'choices': ['F9 v1.0', 'F9 v1.1', 'F9 FT', 'B5']
        },
        'Launch Site': {
            'choices': ['CCAFS LC-40', 'KSC LC-39A', 'VAFB SLC-4E']
        }
    }

    try:
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} rows")

        for idx, row in df.iterrows():
            errors, warnings = validate_data(row.to_dict(), rules)
            if errors:
                print(f"  Row {idx+2}: Errors - {errors}")
            if warnings:
                print(f"  Row {idx+2}: Warnings - {warnings}")

        print("\n✅ Validation complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Data validation')
    parser.add_argument('file', help='CSV file to validate')
    args = parser.parse_args()

    validate_csv(args.file)


if __name__ == "__main__":
    main()