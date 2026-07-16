from aletheia.config import AletheiaConfig
from aletheia.pipeline import AletheiaPipeline


def main() -> None:
    config = AletheiaConfig()

    pipeline = AletheiaPipeline(config)

    result = pipeline.run(
        google_ads_path=config.google_ads_path,
        meta_ads_path=config.meta_ads_path,
        bing_ads_path=config.bing_ads_path,
        save_model=True,
    )

    print("\n========== Aletheia ==========")
    print("Pipeline completed successfully.\n")

    print("Training Metrics")
    print("----------------")
    print(f"RMSE : {result.training_result.rmse:.4f}")
    print(f"MAE  : {result.training_result.mae:.4f}")
    print(f"R²   : {result.training_result.r2:.4f}")

    print("\nRows Processed")
    print("----------------")
    print(f"Unified rows : {len(result.unified_df)}")
    print(f"Feature rows : {len(result.feature_df)}")


if __name__ == "__main__":
    main()