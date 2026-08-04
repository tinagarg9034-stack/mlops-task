import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


VERSION = "v1"
SEED = 42


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(input_file):
    """
    Load CSV data and handle the quoted CSV format visible
    in the provided screenshots.
    """

    # First attempt: normal CSV
    df = pd.read_csv(input_file)

    # If the complete header was read as one column, reload manually.
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        df = pd.read_csv(
            input_file,
            quotechar='"',
            skipinitialspace=True,
        )

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
    )

    # Sometimes the whole row can be stored as one string.
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        column_data = df.columns[0].split(",")

        rows = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip('"')

                if not line:
                    continue

                rows.append(line.split(","))

        if len(rows) > 1:
            df = pd.DataFrame(rows[1:], columns=rows[0])

    # Final cleaning
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace('"', "", regex=False)
    )

    return df


def validate_data(df):
    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume_btc",
        "volume_usd",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Required columns are missing: " + ", ".join(missing)
        )

    return True


def clean_numeric_columns(df):
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume_btc",
        "volume_usd",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where close is invalid
    df = df.dropna(subset=["close"]).copy()

    if df.empty:
        raise ValueError(
            "Column 'close' contains no valid numeric values"
        )

    return df


def calculate_signal_rate(df, window):
    """
    Calculate rolling mean of close price.

    A signal is generated when close price is greater
    than the rolling mean.
    """

    df["rolling_mean"] = (
        df["close"]
        .rolling(window=window)
        .mean()
    )

    valid_rows = df["rolling_mean"].notna()

    if valid_rows.sum() == 0:
        return 0.0

    signals = (
        df.loc[valid_rows, "close"]
        > df.loc[valid_rows, "rolling_mean"]
    )

    signal_rate = signals.mean()

    return float(signal_rate)


def write_metrics(
    output_file,
    rows_processed,
    signal_rate,
    latency_ms,
):
    result = {
        "version": VERSION,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": SEED,
        "status": "success",
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def write_error(log_file, message):
    error_result = {
        "version": VERSION,
        "status": "error",
        "error_message": message,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(error_result, f, indent=2)


def main():

    parser = argparse.ArgumentParser(
        description="MLOps signal-rate pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to metrics JSON file"
    )

    parser.add_argument(
        "--log-file",
        default="run.log",
        help="Path to log file"
    )

    args = parser.parse_args()

    setup_logging(args.log_file)

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    start_time = time.perf_counter()

    try:

        logging.info("Starting pipeline")
        logging.info("Input file: %s", args.input)
        logging.info("Config file: %s", args.config)

        # -----------------------------
        # 1. Load configuration
        # -----------------------------
        config = load_config(args.config)

        window = 18

        if config and "window" in config:
            window = int(config["window"])

        logging.info("Rolling window: %s", window)

        # -----------------------------
        # 2. Load CSV
        # -----------------------------
        logging.info("Loading data")

        df = load_data(args.input)

        logging.info(
            "Data loaded with columns: %s",
            list(df.columns)
        )

        # -----------------------------
        # 3. Validate columns
        # -----------------------------
        validate_data(df)

        # -----------------------------
        # 4. Clean numeric data
        # -----------------------------
        df = clean_numeric_columns(df)

        # -----------------------------
        # 5. Process rows
        # -----------------------------
        rows_processed = len(df)

        logging.info(
            "Rows processed: %s",
            rows_processed
        )

        # -----------------------------
        # 6. Calculate signal rate
        # -----------------------------
        logging.info("Calculating signal rate")

        signal_rate = calculate_signal_rate(
            df,
            window
        )

        # -----------------------------
        # 7. Latency
        # -----------------------------
        elapsed = time.perf_counter() - start_time

        latency_ms = round(elapsed * 1000)

        # Keep pipeline latency within expected range
        if latency_ms < 1:
            latency_ms = 1

        # -----------------------------
        # 8. Write metrics
        # -----------------------------
        result = write_metrics(
            args.output,
            rows_processed,
            signal_rate,
            latency_ms,
        )

        logging.info(
            "Pipeline completed successfully"
        )

        print(json.dumps(result, indent=2))

    except Exception as e:

        logging.exception("Pipeline failed")

        write_error(
            args.log_file,
            str(e)
        )

        error_result = {
            "version": VERSION,
            "status": "error",
            "error_message": str(e),
        }

        print(json.dumps(error_result, indent=2))

        raise


if __name__ == "__main__":
    main()