"""
CTS Training Analysis Script

Runs CTS warm-start training with enhanced monitoring to visualize:
- Signal weight evolution throughout training
- Parameter health (exploding/vanishing detection)
- Training signal distributions

This script does NOT modify any existing pipelines - it runs training
independently with custom monitoring hooks.

Usage:
    cd ~/python/cts-reco-training
    uv run python experiments/scripts/analyze_cts_training.py [--mode standard|curtain]

Options:
    --mode standard  Use standard 18-dim context (default)
    --mode curtain   Use curtain 21-dim context

Outputs saved to: experiments/outputs/training_diagnostics/{mode}/
"""

import sys
from pathlib import Path
import logging
import argparse
import joblib
import numpy as np

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.models.slow_bias_thompson_sampler import SlowBiasThompsonSampler
from cts_recommender.imitation_learning.IL_constants import SIGNAL_NAMES, TARGET_SIGNAL_WEIGHTS
from cts_recommender.pipelines.CTS_warmstart_training_pipeline import _extract_signal_matrix

# Import plotting utilities from same directory
# Note: SIGNAL_NAMES from IL_constants is lowercase for data keys
# The plots module has its own title-case version for display labels
from cts_training_plots import (
    plot_weight_evolution,
    plot_parameter_health,
    plot_signal_distributions,
    plot_training_summary,
    generate_analysis_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_context_features(training_samples: list) -> np.ndarray:
    """
    Extract context features from training samples.

    Returns array of shape (n_samples, context_dim)
    """
    contexts = []
    for sample in training_samples:
        contexts.append(sample['context_features'])
    return np.array(contexts)


def extract_signals_dict(training_samples: list) -> dict:
    """
    Extract signals as a dictionary for distribution plotting.

    Keys match SIGNAL_NAMES order: curator, audience, competition, diversity, novelty, rights
    """
    signals = {name: [] for name in SIGNAL_NAMES}

    for sample in training_samples:
        value_signals = sample['value_signals']
        signals['curator'].append(float(sample['selected']))
        signals['audience'].append(value_signals['audience'])
        signals['competition'].append(value_signals['competition'])
        signals['diversity'].append(value_signals['diversity'])
        signals['novelty'].append(value_signals['novelty'])
        signals['rights'].append(value_signals['rights'])

    return {key: np.array(val) for key, val in signals.items()}


def run_training_with_weight_monitoring(
    contexts: np.ndarray,
    signals: np.ndarray,
    rewards: np.ndarray,
    target_weights: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1,
    monitor_every: int = 100,
    n_weight_samples: int = 100,
    lr_bias_ratio: float | None = None,
) -> tuple[ContextualThompsonSampler, dict]:
    """
    Run CTS training with enhanced monitoring for weight evolution.

    This wraps the standard warm_start with additional weight tracking
    at each monitoring checkpoint.

    Args:
        contexts: Context features, shape (n_samples, context_dim)
        signals: Signal matrix, shape (n_samples, num_signals)
        rewards: Reward targets, shape (n_samples,)
        target_weights: Target weight distribution for initialization
        lr: Learning rate
        epochs: Number of training epochs
        monitor_every: Frequency of monitoring checkpoints
        n_weight_samples: Number of context samples for weight computation
        lr_bias_ratio: If provided, use SlowBiasThompsonSampler with slower bias learning

    Returns:
        Tuple of (trained_model, enhanced_history)
    """
    num_samples, context_dim = contexts.shape
    num_signals = signals.shape[1]

    logger.info(f"Training data: {num_samples} samples, {context_dim} context dims, {num_signals} signals")

    # Initialize CTS model
    cts = ContextualThompsonSampler(
        num_signals=num_signals,
        context_dim=context_dim,
        expl_scale=1e-4,
        ema_decay=0.9999,
        h0=0.1,
        tau=2.0,
        alpha=0.3,
        weight_decay=1e-4,
        match_loss_weight=0.5,
        random_state=42
    )

    # Initialize with target weights
    cts.initialize_with_target_weights(target_weights)
    logger.info("Model initialized with target weights")

    # Run standard warm_start with monitoring
    logger.info(f"Starting training: {epochs} epoch(s), lr={lr}, monitor_every={monitor_every}")

    history = cts.warm_start(
        contexts=contexts,
        signals=signals,
        rewards=rewards,
        epochs=epochs,
        lr=lr,
        expl_scale=1e-4,
        ema_decay=0.9999,
        weight_decay=1e-4,
        monitor_every=monitor_every,
        verbose=True
    )

    # Now enhance history with weight evolution data
    # We need to re-run through the checkpoints to compute weights
    # Since we can't hook into warm_start, we'll compute weights post-hoc
    # by sampling contexts and computing weights with the final model parameters

    # For proper weight evolution, we need to re-train and capture at each checkpoint
    # Let's do a custom training loop
    logger.info("Re-running training with weight evolution tracking...")

    enhanced_history = run_custom_training_with_weights(
        contexts=contexts,
        signals=signals,
        rewards=rewards,
        target_weights=target_weights,
        lr=lr,
        epochs=epochs,
        monitor_every=monitor_every,
        n_weight_samples=n_weight_samples,
        lr_bias_ratio=lr_bias_ratio,
    )

    # Merge standard history metrics into enhanced history
    for key in ['U_norm', 'b_norm', 'grad_U_norm', 'grad_b_norm',
                'h_U_mean', 'h_U_min', 'h_U_max', 'loss']:
        if key in history and key not in enhanced_history:
            enhanced_history[key] = history[key]

    return cts, enhanced_history


def run_custom_training_with_weights(
    contexts: np.ndarray,
    signals: np.ndarray,
    rewards: np.ndarray,
    target_weights: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1,
    monitor_every: int = 100,
    n_weight_samples: int = 100,
    lr_bias_ratio: float | None = None,
) -> dict:
    """
    Custom training loop that captures weight evolution at each checkpoint.

    Args:
        contexts: Context features
        signals: Signal matrix
        rewards: Reward targets
        target_weights: Target weights for initialization
        lr: Learning rate
        epochs: Training epochs
        monitor_every: Checkpoint frequency
        n_weight_samples: Contexts to sample for weight computation
        lr_bias_ratio: If provided, use SlowBiasThompsonSampler with slower bias learning

    Returns:
        History dict with weight evolution data
    """
    num_samples, context_dim = contexts.shape
    num_signals = signals.shape[1]

    # Initialize model - use SlowBiasThompsonSampler if lr_bias_ratio provided
    common_params = {
        "num_signals": num_signals,
        "context_dim": context_dim,
        "expl_scale": 1e-4,
        "ema_decay": 0.9999,
        "h0": 0.1,
        "tau": 2.0,
        "alpha": 0.3,
        "weight_decay": 1e-4,
        "match_loss_weight": 0.5,
        "random_state": 42,
    }

    if lr_bias_ratio is not None:
        logger.info(f"Using SlowBiasThompsonSampler with lr_bias_ratio={lr_bias_ratio}")
        cts = SlowBiasThompsonSampler(lr_bias_ratio=lr_bias_ratio, **common_params)
    else:
        cts = ContextualThompsonSampler(**common_params)

    cts.initialize_with_target_weights(target_weights)

    # Initialize history
    history = {
        'update_steps': [],
        'signal_weights_mean': [],
        'signal_weights_std': [],
        'bias_values': [],
        'U_norm': [],
        'b_norm': [],
        'U_std': [],
        'b_std': [],
        'grad_U_norm': [],
        'grad_b_norm': [],
        'h_U_mean': [],
        'h_U_min': [],
        'h_U_max': [],
        'loss': []
    }

    # Random generator for sampling
    rng = np.random.default_rng(42)

    # Sample indices for weight computation (fixed throughout training)
    weight_sample_indices = rng.choice(num_samples, size=min(n_weight_samples, num_samples), replace=False)

    def compute_weight_stats():
        """Compute weight statistics across sampled contexts."""
        sampled_weights = np.zeros((len(weight_sample_indices), num_signals))
        for idx, ctx_idx in enumerate(weight_sample_indices):
            context = contexts[ctx_idx]
            weights, _ = cts._compute_w(cts.U, cts.b, context)
            sampled_weights[idx] = weights

        return sampled_weights.mean(axis=0), sampled_weights.std(axis=0)

    def record_checkpoint(step: int, grad_u: np.ndarray, grad_b: np.ndarray, loss_val: float):
        """Record all metrics at a checkpoint."""
        weights_mean, weights_std = compute_weight_stats()

        history['update_steps'].append(step)
        history['signal_weights_mean'].append(weights_mean)
        history['signal_weights_std'].append(weights_std)
        history['bias_values'].append(cts.b.copy())
        history['U_norm'].append(np.linalg.norm(cts.U))
        history['b_norm'].append(np.linalg.norm(cts.b))
        history['U_std'].append(np.std(cts.U))
        history['b_std'].append(np.std(cts.b))
        history['grad_U_norm'].append(np.linalg.norm(grad_u))
        history['grad_b_norm'].append(np.linalg.norm(grad_b))
        history['h_U_mean'].append(np.mean(cts.h_U))
        history['h_U_min'].append(np.min(cts.h_U))
        history['h_U_max'].append(np.max(cts.h_U))
        history['loss'].append(loss_val)

    # Record initial state
    zero_grad = np.zeros_like(cts.U)
    zero_grad_b = np.zeros_like(cts.b)
    record_checkpoint(0, zero_grad, zero_grad_b, 0.0)

    # Training loop - uses cts.update() directly for accurate behavior
    update_count = 0
    last_diagnostics = {"grad_U_norm": 0.0, "grad_b_norm": 0.0, "loss": 0.0}

    for epoch in range(epochs):
        # Shuffle indices
        perm = rng.permutation(num_samples)

        for idx in perm:
            context = contexts[idx]
            signal = signals[idx]
            reward = rewards[idx]

            # Use the model's actual update method for accurate behavior
            diagnostics = cts.update(
                c_t=context,
                s_chosen=signal,
                r=reward,
                lr=lr,
                return_diagnostics=True,
            )

            update_count += 1
            last_diagnostics = diagnostics

            # Record checkpoint
            if update_count % monitor_every == 0:
                # Create gradient arrays from diagnostics for checkpoint recording
                grad_u_placeholder = np.full_like(cts.U, diagnostics["grad_U_norm"] / np.sqrt(cts.U.size))
                grad_b_placeholder = np.full_like(cts.b, diagnostics["grad_b_norm"] / np.sqrt(cts.b.size))
                record_checkpoint(update_count, grad_u_placeholder, grad_b_placeholder, diagnostics["loss"])

                # Log progress (indices match SIGNAL_NAMES: curator=0, audience=1, rights=5)
                weights_mean = history['signal_weights_mean'][-1]
                logger.info(
                    f"Step {update_count}: "
                    f"Curator={weights_mean[0]:.1%}, "
                    f"Audience={weights_mean[1]:.1%}, "
                    f"Rights={weights_mean[5]:.1%}, "
                    f"||U||={history['U_norm'][-1]:.4f}"
                )

    # Record final state if not already recorded
    if update_count % monitor_every != 0:
        grad_u_placeholder = np.full_like(cts.U, last_diagnostics["grad_U_norm"] / np.sqrt(cts.U.size))
        grad_b_placeholder = np.full_like(cts.b, last_diagnostics["grad_b_norm"] / np.sqrt(cts.b.size))
        record_checkpoint(update_count, grad_u_placeholder, grad_b_placeholder, last_diagnostics["loss"])

    return history


def main():
    """Main entry point for training analysis."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CTS Training Analysis")
    parser.add_argument(
        "--mode",
        choices=["standard", "curtain"],
        default="standard",
        help="Context mode: standard (18-dim) or curtain (21-dim). Default: standard"
    )
    parser.add_argument(
        "--curator-model",
        type=Path,
        default=None,
        help="Path to curator model. Default: auto-select based on mode"
    )
    args = parser.parse_args()

    # Paths based on mode
    data_dir = PROJECT_ROOT / "data" / "processed" / "IL"
    models_dir = PROJECT_ROOT / "data" / "models"

    if args.mode == "curtain":
        training_samples_file = data_dir / "curtain_mode" / "training_samples_curtain.joblib"
        curator_model_file = args.curator_model or models_dir / "curtain_mode" / "curator_logistic_model_curtain.joblib"
        output_subdir = "curtain_mode"
    else:
        training_samples_file = data_dir / "training_samples.joblib"
        curator_model_file = args.curator_model or models_dir / "curator_logistic_model.joblib"
        output_subdir = "standard_mode"

    output_dir = PROJECT_ROOT / "experiments" / "outputs" / "training_diagnostics" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running analysis in {args.mode} mode")
    logger.info(f"Training samples: {training_samples_file}")
    logger.info(f"Curator model: {curator_model_file}")
    logger.info(f"Output directory: {output_dir}")

    # Load curator model
    logger.info(f"Loading curator model from {curator_model_file}")
    curator_model = joblib.load(curator_model_file)

    # Load training samples
    logger.info(f"Loading training samples from {training_samples_file}")
    training_samples = joblib.load(training_samples_file)
    logger.info(f"Loaded {len(training_samples)} training samples")

    # Extract data
    logger.info("Extracting signals and contexts...")
    signals = _extract_signal_matrix(training_samples, curator_model)
    contexts = extract_context_features(training_samples)
    signals_dict = extract_signals_dict(training_samples)

    # Use the blended pseudo-reward from IL pipeline (gamma-balanced, not raw curator)
    rewards = np.array([s['reward'] for s in training_samples])

    logger.info(f"Signals shape: {signals.shape}")
    logger.info(f"Contexts shape: {contexts.shape}")

    # Load target weights from IL_constants (defined in TARGET_SIGNAL_WEIGHTS)
    target_weights = np.array([TARGET_SIGNAL_WEIGHTS[name] for name in SIGNAL_NAMES])

    logger.info("Target weights (from TARGET_SIGNAL_WEIGHTS):")
    for idx, name in enumerate(SIGNAL_NAMES):
        logger.info(f"  [{idx}] {name}: {target_weights[idx]:.1%}")

    # Run training with weight monitoring
    # Set to None for standard CTS, or a float (e.g., 0.1) for slower bias learning
    # NOTE: Must match Makefile CTS_LR_BIAS_RATIO setting for reproducible results
    lr_bias_ratio = None  # No slow bias - matches Makefile (lr_bias_ratio=1.0)
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING WITH WEIGHT MONITORING")
    if lr_bias_ratio is not None:
        logger.info(f"Using lr_bias_ratio={lr_bias_ratio} (bias learns {int(1/lr_bias_ratio)}x slower)")
    else:
        logger.info("Using standard ContextualThompsonSampler (no slow bias)")
    logger.info("=" * 60)

    _, history = run_training_with_weight_monitoring(
        contexts=contexts,
        signals=signals,
        rewards=rewards,
        target_weights=target_weights,
        lr=0.01,  # Testing higher learning rate
        epochs=1,
        monitor_every=100,
        n_weight_samples=100,
        lr_bias_ratio=lr_bias_ratio,
    )

    # Save history for future re-plotting
    history_file = output_dir / "training_history.joblib"
    joblib.dump(history, history_file)
    logger.info(f"Saved training history to {history_file}")

    # Generate plots
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 60)

    # Plot 1: Weight evolution
    logger.info("Generating weight evolution plot...")
    plot_weight_evolution(
        history=history,
        target_weights=target_weights,
        output_path=output_dir / "weight_evolution.png"
    )

    # Plot 2: Parameter health
    logger.info("Generating parameter health plot...")
    plot_parameter_health(
        history=history,
        output_path=output_dir / "parameter_health.png"
    )

    # Plot 3: Signal distributions
    logger.info("Generating signal distributions plot...")
    selected_mask = signals_dict['curator'] > 0.5
    plot_signal_distributions(
        signals=signals_dict,
        selected_mask=selected_mask,
        output_path=output_dir / "signal_distributions.png"
    )

    # Plot 4: Training summary dashboard
    logger.info("Generating training summary dashboard...")
    plot_training_summary(
        history=history,
        target_weights=target_weights,
        output_path=output_dir / "training_summary.png"
    )

    # Generate text report
    logger.info("Generating analysis report...")
    report = generate_analysis_report(
        history=history,
        target_weights=target_weights,
        output_path=output_dir / "analysis_report.txt"
    )
    print("\n" + report)

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
