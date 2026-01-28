from pathlib import Path
import pandas as pd
from cts_recommender.preprocessing import programming, movie_features, ML_feature_processing
from cts_recommender.io.writers import atomic_write_parquet
from cts_recommender.settings import get_settings

cfg = get_settings()

def run_original_Rdata_programming_pipeline( raw_file: Path | None = None, out_file: Path | None = None, rdata_key: str = 'mydata') -> tuple[pd.DataFrame, Path]:

    if raw_file is None:
        raw_file = cfg.original_rdata_programming
    if out_file is None:
        out_file = cfg.processed_dir / "preprocessed_programming.parquet"
    df_raw = programming.load_original_programming(raw_file, key=rdata_key)
    df = programming.preprocess_programming(df_raw)

    atomic_write_parquet(df, out_file)
    return df, out_file

def run_enrich_programming_with_movie_metadata_pipeline( processed_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:

    if processed_file is None:
        processed_file = cfg.processed_dir / "preprocessed_programming.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "enriched_programming.parquet"
    df_processed = movie_features.load_processed_programming(processed_file)
    df_enriched = movie_features.enrich_programming_with_movie_metadata(df_processed)

    atomic_write_parquet(df_enriched, out_file)

    return df_enriched, out_file

def run_ML_feature_processing_pipeline( processed_file: Path | None = None, enriched_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:

    if processed_file is None:
        processed_file = cfg.processed_dir / "preprocessed_programming.parquet"
    if enriched_file is None:
        enriched_file = cfg.processed_dir / "enriched_programming.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "ML_features.parquet"


    processed_df, enriched_df = ML_feature_processing.load_processed_and_enriched_programming(
        data_path_processed=processed_file,
        data_path_enriched=enriched_file
    )

    ML_df = ML_feature_processing.build_X_features(processed_df, enriched_df)

    atomic_write_parquet(ML_df, out_file)
    
    return ML_df, out_file