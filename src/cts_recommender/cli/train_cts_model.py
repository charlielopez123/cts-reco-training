"""
CLI for training the Contextual Thompson Sampler with warm-start.
"""

import argparse
import logging
from pathlib import Path

from cts_recommender.pipelines import CTS_warmstart_training_pipeline
from cts_recommender.settings import get_settings
from cts_recommender import logging_setup

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Contextual Thompson Sampler with warm-start on historical data"
    )

    cfg = get_settings()

    # Input/Output paths
    parser.add_argument(
        "--training-data",
        type=Path,
        default=cfg.processed_dir / "IL" / "training_data.joblib",
        help="Path to IL training data (.joblib)"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.models_dir / "cts_model.npz",
        help="Path to save trained CTS model (.npz)"
    )

    parser.add_argument(
        "--curator-model",
        type=Path,
        default=cfg.models_dir / "curator_logistic_model.joblib",
        help="Path to trained curator model (.joblib) for computing curator signal"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Warm-start hyperparameters
    parser.add_argument(
        "--lr-warmstart",
        type=float,
        default=0.005,
        help="Learning rate for warm-start training (default: 0.005)"
    )

    parser.add_argument(
        "--lr-bias-ratio",
        type=float,
        default=0.1,
        help="Bias learning rate ratio during warm-start (default: 0.1 = 10x slower). "
             "Set to 1.0 for normal bias learning."
    )

    parser.add_argument(
        "--expl-scale-warmstart",
        type=float,
        default=1e-4,
        help="Exploration noise scale during warm-start (default: 1e-4)"
    )

    parser.add_argument(
        "--ema-decay-warmstart",
        type=float,
        default=0.9999,
        help="EMA decay for Hessian during warm-start (default: 0.9999)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for warm-start training (default: 1)"
    )

    # Online hyperparameters (stored in model for later use)
    parser.add_argument(
        "--lr-online",
        type=float,
        default=0.05,
        help="Learning rate for online learning (default: 0.05)"
    )

    parser.add_argument(
        "--expl-scale-online",
        type=float,
        default=0.001,
        help="Exploration noise scale for online learning (default: 0.001)"
    )

    parser.add_argument(
        "--ema-decay-online",
        type=float,
        default=0.999,
        help="EMA decay for Hessian during online learning (default: 0.999)"
    )

    # Model hyperparameters (fixed)
    parser.add_argument(
        "--h0",
        type=float,
        default=0.1,
        help="Initial Hessian diagonal value (default: 0.1)"
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Softmax temperature (default: 2.0)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Uniform mixing weight (default: 0.3)"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization strength (default: 1e-4)"
    )

    parser.add_argument(
        "--match-loss-weight",
        type=float,
        default=0.5,
        help="Weight for signal matching loss (default: 0.5)"
    )

    # Training options
    parser.add_argument(
        "--monitor-every",
        type=int,
        default=500,
        help="Record diagnostics every N updates (default: 500, 0 to disable)"
    )

    args = parser.parse_args()

    # Initialize logging
    logging_setup.setup_logging(args.log_level)

    # Convert monitor_every=0 to None
    monitor_every = args.monitor_every if args.monitor_every > 0 else None

    # Convert lr_bias_ratio=1.0 to None (normal learning)
    lr_bias_ratio = args.lr_bias_ratio if args.lr_bias_ratio != 1.0 else None

    try:
        cts, model_path = CTS_warmstart_training_pipeline.run_CTS_warmstart_training_pipeline(
            training_data_file=args.training_data,
            model_output_file=args.out,
            curator_model_file=args.curator_model,
            lr_warmstart=args.lr_warmstart,
            lr_bias_ratio=lr_bias_ratio,
            lr_online=args.lr_online,
            expl_scale_warmstart=args.expl_scale_warmstart,
            expl_scale_online=args.expl_scale_online,
            ema_decay_warmstart=args.ema_decay_warmstart,
            ema_decay_online=args.ema_decay_online,
            h0=args.h0,
            tau=args.tau,
            alpha=args.alpha,
            weight_decay=args.weight_decay,
            match_loss_weight=args.match_loss_weight,
            epochs=args.epochs,
            monitor_every=monitor_every,
            verbose=False
        )

        logger.info(f"SUCCESS: CTS model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
