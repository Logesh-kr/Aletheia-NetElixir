import argparse
import logging
from pathlib import Path

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

    logger.info("Saving predictions to %s", output_path)
    predictions_df.to_csv(output_path, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
