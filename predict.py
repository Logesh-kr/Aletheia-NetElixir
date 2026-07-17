import argparse
import json
import logging
import time
from pathlib import Path
from textwrap import dedent

from aletheia.features.pipeline import FeaturePipeline
from aletheia.ingestion import IngestionPipeline
from aletheia.models.lightgbm_model import LightGBMModel
from aletheia.models.predictor import ModelPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Aletheia Inference Pipeline")

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing google_ads.csv, meta_ads.csv, and bing_ads.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output predictions CSV",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="pickle/model.pkl",
        help="Path to the trained model artifact",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    model_path = Path(args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading trained model from %s", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = LightGBMModel()
    predictor = ModelPredictor(model)
    predictor.load_model(model_path)

    logger.info("Ingesting data from %s", data_dir)

    ingestion = IngestionPipeline()

    unified_df = ingestion.run(
        google_ads_path=data_dir / "google_ads.csv",
        meta_ads_path=data_dir / "meta_ads.csv",
        bing_ads_path=data_dir / "bing_ads.csv",
    )

    logger.info("Generating features...")

    feature_pipeline = FeaturePipeline()
    feature_df = feature_pipeline.transform(unified_df)

    logger.info("Running inference...")

    predictions_df = predictor.predict(feature_df)

    prediction_series = predictions_df["prediction"]

    print("\n==============================")
    print("Prediction Statistics")
    print("==============================")
    print(f"Rows Processed      : {len(predictions_df)}")
    print(f"Total Revenue       : ${prediction_series.sum():,.2f}")
    print(f"Mean Prediction     : ${prediction_series.mean():,.2f}")
    print(f"Median Prediction   : ${prediction_series.median():,.2f}")
    print(f"Std Deviation       : ${prediction_series.std():,.2f}")
    print(f"Minimum Prediction  : ${prediction_series.min():,.2f}")
    print(f"Maximum Prediction  : ${prediction_series.max():,.2f}")

    # -----------------------------
    # AI Business Insights
    # -----------------------------

    insights_df = unified_df.copy()
    insights_df["prediction"] = prediction_series.values

    date_range = "N/A"
    if "date" in insights_df.columns and not insights_df["date"].empty:
        min_date = insights_df["date"].min().strftime("%Y-%m-%d")
        max_date = insights_df["date"].max().strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    total_predicted_revenue = prediction_series.sum()
    platform_summary = (
        insights_df.groupby("platform")["prediction"]
        .agg(["sum", "mean"])
        .sort_values("sum", ascending=False)
    )

    platform_lines = []
    for platform, row in platform_summary.iterrows():
        pct = (row["sum"] / total_predicted_revenue) * 100 if total_predicted_revenue > 0 else 0
        platform_lines.append(f"{platform.ljust(15)} : ${row['sum']:>12,.2f} ({pct:>5.1f}%) | Mean: ${row['mean']:>10,.2f}")
    platform_summary_str = "\n".join(platform_lines)

    campaign_summary = (
        insights_df.groupby("campaign_name")["prediction"]
        .sum()
        .sort_values(ascending=False)
    )

    top_5_campaigns = campaign_summary.head(5)
    top_5_lines = []
    for i, (camp, rev) in enumerate(top_5_campaigns.items(), 1):
        top_5_lines.append(f"{i}. {camp.ljust(30)} : ${rev:>12,.2f}")
    top_5_str = "\n".join(top_5_lines)

    top_platform = platform_summary.index[0]
    bottom_platform = platform_summary.index[-1]
    top_campaign = campaign_summary.index[0]

    recs = [
        f"- Double down on the highest performing campaign: {top_campaign}.",
        f"- Analyze {bottom_platform} to improve ROI, as it currently has the lowest predicted revenue.",
        f"- Investigate top performing campaigns in {top_platform} to identify successful patterns.",
        "- Use these forecasts to guide future budget allocation.",
        "- Retrain the forecasting model periodically with fresh marketing data."
    ]
    recommendations_str = "\n".join(recs)

    elapsed_time = time.time() - start_time

    insights = f"""==============================
Aletheia AI Business Insights
==============================

Overall Forecast
----------------
Date Range                : {date_range}
Total Predicted Revenue   : ${total_predicted_revenue:,.2f}
Average Predicted Revenue : ${prediction_series.mean():,.2f}

Platform Performance
--------------------
Best Performing Platform  : {top_platform}
Lowest Performing Platform: {bottom_platform}

Revenue Contribution by Platform
--------------------------------
{platform_summary_str}

Top 5 Campaigns by Revenue
--------------------------
{top_5_str}

Recommendations
---------------
{recommendations_str}

Execution Time
--------------
{elapsed_time:.2f} seconds"""

    print("\n")
    print(insights)

    insights_path = output_path.parent / "insights.txt"

    with open(insights_path, "w", encoding="utf-8") as file:
        file.write(insights + "\n")

    logger.info("Business insights saved to %s", insights_path)

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_range": date_range,
        "total_predicted_revenue": float(total_predicted_revenue),
        "mean_prediction": float(prediction_series.mean()),
        "platform_summary": {
            platform: {
                "total_revenue": float(row["sum"]),
                "mean_revenue": float(row["mean"])
            }
            for platform, row in platform_summary.iterrows()
        },
        "top_5_campaigns": {
            camp: float(rev) for camp, rev in top_5_campaigns.items()
        },
        "execution_time_seconds": float(elapsed_time)
    }

    summary_path = output_path.parent / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=4)

    logger.info("JSON summary saved to %s", summary_path)

    logger.info("Saving predictions to %s", output_path)

    predictions_df.to_csv(output_path, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()