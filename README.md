# MLOps Task – Signal Rate Pipeline

## Project Overview

This project implements a simple MLOps data-processing pipeline using Python.

The pipeline:

- Reads input data from a CSV file
- Validates the required columns
- Processes the `close` price column
- Calculates a rolling mean
- Generates a signal rate metric
- Records execution metrics
- Logs execution details
- Supports configuration through a YAML file
- Can be executed locally or using Docker

---

## Project Structure

```text
mlops-task/
│
├── run.py
├── data.csv
├── config.yaml
├── requirements.txt
├── Dockerfile
├── README.md
├── metrics.json
└── run.log