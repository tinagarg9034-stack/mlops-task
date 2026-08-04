# MLOps Task – Signal Rate Pipeline

A simple end-to-end MLOps project that demonstrates data processing, rolling mean calculation, signal generation, metrics tracking, logging, configuration management, reproducibility, and Docker containerization.

---

## 📌 Project Overview

This project implements a reproducible data-processing pipeline using Python and Pandas.

The pipeline:

1. Reads input data from a CSV file.
2. Validates the required columns.
3. Converts the `close` column into numeric values.
4. Calculates a rolling mean.
5. Generates trading signals.
6. Calculates the signal rate.
7. Measures pipeline latency.
8. Stores execution metrics in `metrics.json`.
9. Records execution logs in `run.log`.
10. Supports configurable parameters through `config.yaml`.
11. Runs locally using Python.
12. Runs inside a Docker container.

The project is designed to demonstrate basic MLOps practices such as reproducibility, configuration management, logging, metrics tracking, validation, and containerization.

---

## 🛠️ Technologies Used

- Python 3.12
- Pandas
- PyYAML
- Docker
- Git
- GitHub
- PowerShell

---

## 📂 Project Structure

```text
mlops-task/
│
├── data.csv
├── config.yaml
├── requirements.txt
├── run.py
├── Dockerfile
├── .dockerignore
├── metrics.json
├── run.log
└── README.md