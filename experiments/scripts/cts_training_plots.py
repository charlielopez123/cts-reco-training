"""
Plotting utilities for CTS training analysis.

Provides visualization functions for analyzing CTS model training:
- Weight evolution over training steps
- Parameter health monitoring (exploding/vanishing detection)
- Training signal distributions
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cts_recommender.imitation_learning.IL_constants import SIGNAL_NAMES as _SIGNAL_NAMES

# Title-case version for plot labels (matches SIGNAL_NAMES order: curator first)
SIGNAL_NAMES = [name.title() for name in _SIGNAL_NAMES]
# Colors: curator, audience, competition, diversity, novelty, rights
SIGNAL_COLORS = ['#AA3377', '#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311']


def plot_weight_evolution(
    history: Dict[str, Any],
    target_weights: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 10),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot signal weight evolution during training.

    Creates a 6-panel subplot (2x3) showing how each signal's learned weight
    changes over training steps. Target weights shown as reference lines.

    Args:
        history: Training history dict containing:
            - 'update_steps': List of step numbers
            - 'signal_weights_mean': List of arrays, shape (num_signals,) each
            - 'signal_weights_std': List of arrays, shape (num_signals,) each
        target_weights: Target weight distribution for reference lines (num_signals,)
        figsize: Figure size (width, height)
        output_path: If provided, save figure to this path

    Returns:
        Tuple of (figure, axes array)
    """
    steps = np.array(history['update_steps'])
    weights_mean = np.array(history['signal_weights_mean'])
    weights_std = np.array(history['signal_weights_std'])
    num_signals = weights_mean.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_signals):
        ax = axes[idx]
        color = SIGNAL_COLORS[idx]
        name = SIGNAL_NAMES[idx]

        # Plot mean weight with std shading
        mean_vals = weights_mean[:, idx]
        std_vals = weights_std[:, idx]

        ax.plot(steps, mean_vals, color=color, linewidth=2, label='Learned weight')
        ax.fill_between(
            steps,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.2,
            label='Context std'
        )

        # Plot target weight as reference
        if target_weights is not None:
            target = target_weights[idx]
            ax.axhline(y=target, color='black', linestyle='--', linewidth=1.5,
                       label=f'Target: {target:.1%}')

            # Calculate final deviation
            final_weight = mean_vals[-1]
            deviation = final_weight - target
            deviation_pct = (deviation / target) * 100 if target > 0 else 0
            ax.set_title(f'{name}\nFinal: {final_weight:.1%} ({deviation_pct:+.0f}% from target)')
        else:
            ax.set_title(f'{name}\nFinal: {mean_vals[-1]:.1%}')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Signal Weight')
        ax.set_ylim(0, 0.55)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Signal Weight Evolution During Training', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_parameter_health(
    history: Dict[str, Any],
    figsize: Tuple[int, int] = (16, 10),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot parameter health metrics to detect exploding/vanishing issues.

    Creates a 4-panel plot showing:
    1. Parameter norms (||U||, ||b||) - detect explosion
    2. Gradient norms - detect vanishing gradients
    3. Parameter std - detect weight collapse
    4. Curvature (h_U) - should stabilize

    Args:
        history: Training history dict with standard monitoring keys
        figsize: Figure size (width, height)
        output_path: If provided, save figure to this path

    Returns:
        Tuple of (figure, axes array)
    """
    steps = np.array(history['update_steps'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Panel 1: Parameter norms
    ax = axes[0]
    ax.plot(steps, history['U_norm'], label='||U|| (weight matrix)', color='#0077BB', linewidth=2)
    ax.plot(steps, history['b_norm'], label='||b|| (bias)', color='#EE7733', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Parameter Magnitudes\n(Explosion: unbounded growth)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Check for explosion warning
    u_norm_final = history['U_norm'][-1]
    u_norm_initial = history['U_norm'][0] if history['U_norm'][0] > 0 else 1e-6
    if u_norm_final / u_norm_initial > 100:
        ax.annotate('Warning: Possible explosion!',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=10, color='red', ha='center')

    # Panel 2: Gradient norms
    ax = axes[1]
    ax.plot(steps, history['grad_U_norm'], label='||grad_U||', color='#0077BB', linewidth=2)
    ax.plot(steps, history['grad_b_norm'], label='||grad_b||', color='#EE7733', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitudes\n(Vanishing: approaching 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Check for vanishing warning
    grad_final = history['grad_U_norm'][-1]
    if grad_final < 1e-8:
        ax.annotate('Warning: Vanishing gradients!',
                    xy=(0.5, 0.05), xycoords='axes fraction',
                    fontsize=10, color='red', ha='center')

    # Panel 3: Parameter std (detect collapse)
    ax = axes[2]
    if 'U_std' in history:
        ax.plot(steps, history['U_std'], label='std(U)', color='#0077BB', linewidth=2)
    if 'b_std' in history:
        ax.plot(steps, history['b_std'], label='std(b)', color='#EE7733', linewidth=2)

    # If std not in history, compute from bias values
    if 'bias_values' in history:
        bias_stds = [np.std(b) for b in history['bias_values']]
        ax.plot(steps, bias_stds, label='std(b) computed', color='#EE7733', linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Parameter Spread\n(Collapse: std approaching 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Curvature (Hessian approximation)
    ax = axes[3]
    ax.plot(steps, history['h_U_mean'], label='h_U mean', color='#0077BB', linewidth=2)
    if 'h_U_min' in history and 'h_U_max' in history:
        ax.fill_between(steps, history['h_U_min'], history['h_U_max'],
                        color='#0077BB', alpha=0.2, label='h_U range')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Curvature Estimate')
    ax.set_title('Hessian Approximation (h_U)\n(Should stabilize, not collapse)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('Parameter Health Monitoring', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_signal_distributions(
    signals: Dict[str, np.ndarray],
    selected_mask: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 10),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot histograms of training signal values.

    Creates a 6-panel plot showing the distribution of each signal,
    optionally split by selected/not-selected status.

    Args:
        signals: Dict mapping signal names to value arrays
        selected_mask: Boolean array indicating which samples were selected by curator
        figsize: Figure size (width, height)
        output_path: If provided, save figure to this path

    Returns:
        Tuple of (figure, axes array)
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    signal_keys = ['audience', 'competition', 'diversity', 'novelty', 'rights', 'curator']

    for idx, key in enumerate(signal_keys):
        ax = axes[idx]
        color = SIGNAL_COLORS[idx]
        name = SIGNAL_NAMES[idx]

        if key not in signals:
            ax.set_title(f'{name}\n(not available)')
            ax.axis('off')
            continue

        values = np.array(signals[key])

        if selected_mask is not None:
            # Split by selection status
            selected_vals = values[selected_mask]
            not_selected_vals = values[~selected_mask]

            # Use same bins for both
            all_vals = np.concatenate([selected_vals, not_selected_vals])
            unique_vals = np.unique(all_vals)

            if len(unique_vals) <= 10:
                # Discrete signal - use bar chart
                bins = np.sort(unique_vals)
                bin_edges = np.concatenate([bins - 0.025, [bins[-1] + 0.025]])

                ax.hist(not_selected_vals, bins=bin_edges, alpha=0.6,
                        label=f'Not selected (n={len(not_selected_vals)})', color='gray')
                ax.hist(selected_vals, bins=bin_edges, alpha=0.8,
                        label=f'Selected (n={len(selected_vals)})', color=color)
            else:
                # Continuous signal
                ax.hist(not_selected_vals, bins=30, alpha=0.6,
                        label=f'Not selected (n={len(not_selected_vals)})', color='gray')
                ax.hist(selected_vals, bins=30, alpha=0.8,
                        label=f'Selected (n={len(selected_vals)})', color=color)

            # Calculate statistics
            mean_selected = np.mean(selected_vals) if len(selected_vals) > 0 else 0
            mean_not_selected = np.mean(not_selected_vals) if len(not_selected_vals) > 0 else 0
            ratio = mean_selected / mean_not_selected if mean_not_selected > 0 else float('inf')

            ax.set_title(f'{name}\nSelected mean: {mean_selected:.3f} ({ratio:.2f}x higher)')
        else:
            # No selection info
            unique_vals = np.unique(values)
            if len(unique_vals) <= 10:
                bins = np.sort(unique_vals)
                bin_edges = np.concatenate([bins - 0.025, [bins[-1] + 0.025]])
                ax.hist(values, bins=bin_edges, alpha=0.8, color=color)
            else:
                ax.hist(values, bins=30, alpha=0.8, color=color)

            ax.set_title(f'{name}\nMean: {np.mean(values):.3f}, Unique: {len(unique_vals)}')

        ax.set_xlabel('Signal Value')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Signal Distributions', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_training_summary(
    history: Dict[str, Any],
    target_weights: np.ndarray,
    figsize: Tuple[int, int] = (16, 12),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, List]:
    """
    Create a summary dashboard of CTS training analysis.

    Combines key metrics into a single figure:
    - Weight evolution (highlighted: rights)
    - Final vs target comparison table
    - Parameter health summary
    - Warning indicators

    Args:
        history: Training history dict with all monitoring keys
        target_weights: Target weight distribution (num_signals,)
        figsize: Figure size (width, height)
        output_path: If provided, save figure to this path

    Returns:
        Tuple of (figure, axes list)
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Panel 1: All weights on one plot (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    steps = np.array(history['update_steps'])
    weights_mean = np.array(history['signal_weights_mean'])

    for idx in range(len(SIGNAL_NAMES)):
        linewidth = 3 if idx == 5 else 1.5  # Highlight rights (index 5 with curator-first order)
        alpha = 1.0 if idx == 5 else 0.7
        ax1.plot(steps, weights_mean[:, idx], color=SIGNAL_COLORS[idx],
                 linewidth=linewidth, alpha=alpha, label=SIGNAL_NAMES[idx])

        # Add target reference lines
        ax1.axhline(y=target_weights[idx], color=SIGNAL_COLORS[idx],
                    linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Signal Weight', fontsize=12)
    ax1.set_title('Signal Weight Evolution (Rights highlighted)', fontsize=14)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.55)

    # Panel 2: Summary table (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')

    final_weights = weights_mean[-1]
    initial_weights = weights_mean[0]

    table_data = []
    for idx, name in enumerate(SIGNAL_NAMES):
        initial = initial_weights[idx]
        final = final_weights[idx]
        target = target_weights[idx]
        deviation = final - target
        pct_change = ((final - initial) / initial * 100) if initial > 0 else 0

        table_data.append([
            name,
            f'{initial:.1%}',
            f'{final:.1%}',
            f'{target:.1%}',
            f'{deviation:+.1%}',
            f'{pct_change:+.0f}%'
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Signal', 'Initial', 'Final', 'Target', 'Deviation', 'Change'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color the Rights row (row index 6 = header row + 6th signal with curator-first order)
    for col_idx in range(6):
        table[(6, col_idx)].set_facecolor('#FFE0E0')  # Light red for Rights row

    ax2.set_title('Weight Summary', fontsize=12, pad=20)

    # Panel 3: Parameter health indicators (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    # Calculate health metrics
    u_norm_ratio = history['U_norm'][-1] / (history['U_norm'][0] + 1e-10)
    grad_final = history['grad_U_norm'][-1]
    h_final = history['h_U_mean'][-1]

    health_status = []
    health_status.append(('U norm ratio', f'{u_norm_ratio:.1f}x',
                          'OK' if u_norm_ratio < 100 else 'WARNING'))
    health_status.append(('Final gradient', f'{grad_final:.2e}',
                          'OK' if grad_final > 1e-8 else 'WARNING'))
    health_status.append(('Final curvature', f'{h_final:.2e}',
                          'OK' if h_final > 1e-6 else 'WARNING'))

    health_text = "Parameter Health:\n\n"
    for metric, value, status in health_status:
        symbol = 'OK' if status == 'OK' else 'WARNING'
        health_text += f"  {metric}: {value} [{symbol}]\n"

    ax3.text(0.1, 0.5, health_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax3.transAxes)
    ax3.set_title('Health Check', fontsize=12, pad=20)

    # Panel 4: Loss curve (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    if 'loss' in history:
        ax4.plot(steps, history['loss'], color='#0077BB', linewidth=2)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Loss not tracked', ha='center', va='center',
                 transform=ax4.transAxes)
        ax4.axis('off')

    # Panel 5: Rights weight zoom (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    rights_idx = 5  # Index 5 with curator-first order
    rights_weights = weights_mean[:, rights_idx]
    rights_target = target_weights[rights_idx]

    ax5.plot(steps, rights_weights, color=SIGNAL_COLORS[rights_idx], linewidth=2)
    ax5.axhline(y=rights_target, color='black', linestyle='--', linewidth=1.5,
                label=f'Target: {rights_target:.1%}')
    ax5.fill_between(steps, rights_target, rights_weights,
                     where=(rights_weights > rights_target),
                     color='red', alpha=0.3, label='Excess weight')
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Rights Weight')
    ax5.set_title('Rights Signal Weight (Zoomed)')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)

    plt.suptitle('CTS Training Analysis Summary', fontsize=16, y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, fig.axes


def generate_analysis_report(
    history: Dict[str, Any],
    target_weights: np.ndarray,
    output_path: Path
) -> str:
    """
    Generate a text summary report of training analysis.

    Args:
        history: Training history dict
        target_weights: Target weight distribution
        output_path: Path to save the report

    Returns:
        Report text
    """
    steps = np.array(history['update_steps'])
    weights_mean = np.array(history['signal_weights_mean'])

    final_weights = weights_mean[-1]
    initial_weights = weights_mean[0]

    report = []
    report.append("=" * 60)
    report.append("CTS TRAINING ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total training steps: {steps[-1]}")
    report.append(f"Monitoring checkpoints: {len(steps)}")
    report.append("")

    report.append("-" * 60)
    report.append("SIGNAL WEIGHT EVOLUTION")
    report.append("-" * 60)
    report.append("")
    report.append(f"{'Signal':<12} {'Initial':>10} {'Final':>10} {'Target':>10} {'Deviation':>12}")
    report.append("-" * 56)

    for idx, name in enumerate(SIGNAL_NAMES):
        initial = initial_weights[idx]
        final = final_weights[idx]
        target = target_weights[idx]
        deviation = final - target

        report.append(f"{name:<12} {initial:>10.1%} {final:>10.1%} {target:>10.1%} {deviation:>+12.1%}")

    report.append("")

    # Find when rights exceeded thresholds (index 5 with curator-first order)
    rights_weights = weights_mean[:, 5]

    exceeded_20pct = np.where(rights_weights > 0.20)[0]
    exceeded_30pct = np.where(rights_weights > 0.30)[0]
    exceeded_40pct = np.where(rights_weights > 0.40)[0]

    report.append("-" * 60)
    report.append("RIGHTS WEIGHT MILESTONES")
    report.append("-" * 60)
    report.append("")
    if len(exceeded_20pct) > 0:
        report.append(f"Rights exceeded 20% at step: {steps[exceeded_20pct[0]]}")
    if len(exceeded_30pct) > 0:
        report.append(f"Rights exceeded 30% at step: {steps[exceeded_30pct[0]]}")
    if len(exceeded_40pct) > 0:
        report.append(f"Rights exceeded 40% at step: {steps[exceeded_40pct[0]]}")

    report.append("")
    report.append("-" * 60)
    report.append("PARAMETER HEALTH")
    report.append("-" * 60)
    report.append("")

    u_norm_ratio = history['U_norm'][-1] / (history['U_norm'][0] + 1e-10)
    report.append(f"U norm growth: {u_norm_ratio:.1f}x {'(OK)' if u_norm_ratio < 100 else '(WARNING: possible explosion)'}")

    grad_final = history['grad_U_norm'][-1]
    report.append(f"Final gradient norm: {grad_final:.2e} {'(OK)' if grad_final > 1e-8 else '(WARNING: vanishing)'}")

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)

    with open(output_path, 'w') as file:
        file.write(report_text)

    return report_text
