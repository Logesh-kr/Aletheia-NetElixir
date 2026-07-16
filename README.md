# Aletheia

Aletheia is an AI-powered marketing analytics pipeline developed for the NetElixir Hackathon. It forecasts campaign revenue by combining advertising data from Google Ads, Meta Ads, and Microsoft Ads using a LightGBM model.

## Features

- Multi-platform data ingestion
- Automated feature engineering
- LightGBM revenue forecasting
- Inference-only prediction pipeline
- Model artifact loading
- Unit tested (177 tests passing)

## Project Structure

```
.
├── aletheia/
├── data/
├── pickle/
├── output/
├── tests/
├── main.py
├── predict.py
├── run.sh
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py
```

## Inference

```bash
python predict.py --data-dir data --output output/predictions.csv
```

or

```bash
./run.sh
```

## Model

The trained model is stored at:

```
pickle/model.pkl
```

## Output

Predictions are written to:

```
output/predictions.csv
```

## Tech Stack

- Python
- Pandas
- NumPy
- LightGBM

## Testing

```bash
pytest
```