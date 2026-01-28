import numpy as np
from typing import Optional, Union
from pathlib import Path
import pickle

from cts_recommender.utils.reproducibility import get_reproducible_random_state


def sigmoid(x: float) -> float:
    """Convert logit to probability using sigmoid function."""
    return 1 / (1 + np.exp(-x))

class ContextualThompsonSampler:
    """
    Learns content recommendations by weighting value signals based on context.

    Uses Thompson Sampling to balance exploration and exploitation. The model learns
    which value signals (audience, competition, diversity, novelty, rights) matter most
    for different contexts (time of day, day of week, etc.).
    """

    def __init__(self,
                num_signals: int,
                context_dim: int,
                lr: float = 0.01,
                expl_scale: float = 1.0,
                ema_decay: float = 1-1e-3,
                h0: float = 1e-3,
                tau: float = 2.0,
                alpha: float = 0.2,
                match_loss_weight: float = 0.5,
                weight_decay: float = 0.0,
                random_state: Optional[int] = None) -> None:
        """
        Initialize the sampler with model dimensions and hyperparameters.

        Args:
            num_signals: Number of value signals (e.g., audience, diversity, rights)
            context_dim: Number of context features (e.g., hour, weekday, season)
            lr: Learning rate for gradient updates
            expl_scale: Exploration noise scale (higher = more exploration)
            ema_decay: Exponential moving average decay for gradient statistics (closer to 1 = slower updates)
            h0: Initial value for Hessian approximation (prevents division by zero)
            tau: Temperature for softmax (higher = more uniform signal weights)
            alpha: Uniform mixing weight (0 = pure softmax, 1 = uniform)
            match_loss_weight: Weight for signal matching loss when curator signals provided
            weight_decay: L2 regularization strength (0 = no regularization)
            random_state: Random seed for reproducibility (None uses default seed)
        """
        self.num_signals = num_signals
        self.context_dim = context_dim
        self.random_state = random_state
        self.rng = get_reproducible_random_state(random_state)
        self.h0 = h0
        self.lr = lr
        self.expl_scale = expl_scale
        self.ema_decay = ema_decay
        self.tau = tau
        self.alpha = alpha
        self.match_loss_weight = match_loss_weight
        self.weight_decay = weight_decay
        self.eps = 1e-8  # for numerical stability


        # initialize U and b to ZERO for perfectly uniform starting weights
        self.U: np.ndarray = np.zeros((num_signals, context_dim))
        self.b: np.ndarray = np.zeros(num_signals)

        # running squared-grad estimate for approximate precision (diagonal Hessian)
        self.h_U: np.ndarray = np.ones_like(self.U) * self.h0  # avoid divide-by-zero
        self.h_b: np.ndarray = np.ones_like(self.b) * self.h0

    def initialize_with_target_weights(self, target_weights: np.ndarray) -> None:
        """
        Initialize bias parameters to produce target weight distribution.

        This sets the bias vector b such that the model produces the specified
        target weights (approximately) across all contexts. Useful for:
        - Starting with uniform weights (1/num_signals for each signal)
        - Preventing one signal from dominating from the start
        - Warm-starting with domain knowledge about signal importance

        Args:
            target_weights: Target weight distribution (num_signals,). Must sum to ~1.
                          e.g., [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] for uniform weights

        Example:
            >>> # Initialize with uniform weights
            >>> cts = ContextualThompsonSampler(num_signals=6, context_dim=16)
            >>> uniform_weights = np.ones(6) / 6
            >>> cts.initialize_with_target_weights(uniform_weights)
            >>>
            >>> # Initialize with custom priorities
            >>> custom_weights = np.array([0.2, 0.1, 0.2, 0.2, 0.1, 0.2])
            >>> cts.initialize_with_target_weights(custom_weights)

        Note:
            - U is kept at 0, so weights are context-independent initially
            - Only b is set to produce target weights via softmax
            - After warm-start training, weights will adapt based on data
        """
        # Validate input
        target_weights = np.asarray(target_weights, dtype=np.float64)
        if target_weights.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"target_weights must have length {self.b.shape[0]} "
                f"(num_signals), got {target_weights.shape[0]}"
            )
        if not np.isclose(target_weights.sum(), 1.0, atol=1e-3):
            raise ValueError(
                f"target_weights must sum to 1.0, got {target_weights.sum():.6f}"
            )
        if np.any(target_weights <= 0):
            raise ValueError("target_weights must all be positive")

        # Convert target weights to logits via inverse softmax
        # We want: w = (1-alpha) * softmax(z/tau) + alpha/K ≈ target_weights
        # Solve for z (bias b) that produces these weights

        # Account for uniform mixing
        K = len(target_weights)

        # Solve: w = (1-alpha) * softmax(z/tau) + alpha/K = target
        # => softmax(z/tau) = (target - alpha/K) / (1-alpha)
        adjusted_target = (target_weights - self.alpha / K) / (1 - self.alpha)

        # Clip to valid probability range (handles cases where target is outside achievable range)
        # This automatically handles:
        # - target < alpha/K (min achievable) → clips to small positive value
        # - target > (1-alpha) + alpha/K (max achievable) → clips to 1.0
        adjusted_target = np.clip(adjusted_target, 1e-10, 1.0)

        # Renormalize to ensure it's a valid probability distribution
        adjusted_target = adjusted_target / adjusted_target.sum()

        # Inverse softmax: z = tau * log(p) + constant
        # Since softmax is translation-invariant, we center at 0 for numerical stability
        log_adjusted = np.log(adjusted_target)
        logits = self.tau * (log_adjusted - log_adjusted.mean())

        # Set bias to produce these logits (U @ c = 0 initially)
        self.b = logits

    def _compute_w(self, U: np.ndarray, b: np.ndarray, c_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute signal weights from context using tempered softmax.

        Converts context features into weights for each value signal. Higher weights
        mean the signal is more important for this context.

        Args:
            U: Weight matrix (num_signals, context_dim)
            b: Bias vector (num_signals,)
            c_t: Context feature vector (context_dim,)

        Returns:
            w: Signal weights (num_signals,) - positive, sum to ~1
            q: Raw softmax probabilities before uniform mixing
        """
        z_t = U @ c_t + b  # Utility logits for each signal given context
        tempered_z_t = z_t / self.tau  # Temperature scaling
        ex = np.exp(tempered_z_t - tempered_z_t.max())  # Numerical stability
        q = ex / ex.sum()  # Tempered softmax
        w = (1-self.alpha)*q + self.alpha/len(q)  # Mix with uniform distribution
        return w, q

    def thompson_sample_w(self, c_t: np.ndarray) -> np.ndarray:
        """
        Sample signal weights with exploration noise (Thompson Sampling).

        Adds Gaussian noise to model parameters based on uncertainty, then computes
        weights. This balances using what we've learned (exploitation) with trying
        new things (exploration).

        Args:
            c_t: Context feature vector (context_dim,)

        Returns:
            w_tilde: Sampled signal weights (num_signals,)
        """
        std_U = np.sqrt(self.expl_scale / (self.h_U + self.eps))
        std_b = np.sqrt(self.expl_scale / (self.h_b + self.eps))
        # Add Gaussian noise scaled by uncertainty
        U_tilde = self.U + self.rng.normal(size=self.U.shape) * std_U
        b_tilde = self.b + self.rng.normal(size=self.b.shape) * std_b
        # Compute weights from noisy parameters
        w_tilde, _ = self._compute_w(U_tilde, b_tilde, c_t)
        return w_tilde 
        
    def score_candidates(self, c_t: np.ndarray, S_matrix: np.ndarray, K: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Score candidate content and return top-K recommendations.

        Each candidate has a vector of signal values (e.g., audience=0.8, diversity=0.6).
        We compute a weighted sum using context-dependent signal weights, then return
        the highest scoring candidates.

        Args:
            c_t: Context feature vector (context_dim,)
            S_matrix: Signal values for each candidate (M candidates, num_signals)
            K: Number of top candidates to return

        Returns:
            topk: Indices of top-K candidates
            topk_scores: Scores for top-K candidates
            w_tilde: Sampled signal weights used for scoring
            scores: All candidate scores
        """
        w_tilde = self.thompson_sample_w(c_t)  # Sample context-dependent weights
        scores = S_matrix @ w_tilde  # Weighted sum of signals for each candidate
        idxs = np.argpartition(scores, -K)[-K:]  # Get top-K indices efficiently
        topk = idxs[np.argsort(scores[idxs])[::-1]]  # Sort top-K by score
        return topk, scores[topk], w_tilde, scores
    
    def update(
        self,
        c_t: np.ndarray,
        s_chosen: np.ndarray,
        r: float,
        y_signals: Optional[np.ndarray] = None,
        lr: Optional[float] = None,
        lr_bias_ratio: Optional[float] = None,
        return_diagnostics: bool = False,
        ) -> Optional[dict]:
        """
        Update model based on outcome of a recommendation.

        Learns from whether a recommendation succeeded (r=1) or failed (r=0).
        Optionally learns which signals the curator valued when y_signals provided.

        Args:
            c_t: Context features when recommendation was made (context_dim,)
            s_chosen: Signal values of chosen content (num_signals,)
            r: Binary reward (1=success, 0=failure)
            y_signals: Optional curator-indicated signals that mattered (num_signals binary vector)
            lr: Override learning rate for this update (None uses default)
            lr_bias_ratio: Ratio for bias learning rate (None = same as lr, 0.1 = 10x slower).
                          Useful during warm-start to prevent bias from dominating.
            return_diagnostics: If True, return dict with gradient and parameter info

        Returns:
            Optional dict with diagnostics if return_diagnostics=True
        """
        # Compute current signal weights for this context
        w, raw = self._compute_w(self.U, self.b, c_t)

        # ---- 1) Reward loss ----
        # How well did we predict success? Update weights based on error
        theta = w.dot(s_chosen)  # Predicted utility
        p = sigmoid(theta)  # Convert to probability
        dL_dtheta = (p - r)  # Error: prediction - actual
        grad_w_L = dL_dtheta * s_chosen
        J_softm = 1/self.tau * (np.diag(w) - np.outer(w, w))  # Softmax Jacobian
        grad_z_w = (1-self.alpha) * J_softm

        # ---- 2) Signal matching loss ----
        # If curator indicated which signals mattered, push weights toward those
        if y_signals is not None and y_signals.sum() > 0:
            y_norm = y_signals / (y_signals.sum() + self.eps)  # Normalize to distribution
            grad_w_L_signal_match = self.match_loss_weight * (w - y_norm)
            grad_w_L += grad_w_L_signal_match

        # ---- 3) Backpropagate through softmax ----
        grad_z_L = grad_z_w.T @ grad_w_L  # (num_signals,)

        # ---- 4) Compute gradients for model parameters ----
        grad_U_L = np.outer(grad_z_L, c_t)
        grad_b_L = grad_z_L

        # ---- 4b) Add L2 regularization (weight decay) ----
        if self.weight_decay > 0:
            grad_U_L += self.weight_decay * self.U
            grad_b_L += self.weight_decay * self.b

        # ---- 5) Update curvature estimates (for uncertainty) ----
        self.h_U = self.ema_decay * self.h_U + (1 - self.ema_decay) * (grad_U_L**2)
        self.h_b = self.ema_decay * self.h_b + (1 - self.ema_decay) * (grad_b_L**2)

        # ---- 6) Apply gradient descent ----
        if lr is None:
            lr = self.lr
        self.U -= lr * grad_U_L
        # Apply slower learning rate to bias if ratio provided (used during warm-start)
        lr_b = lr * lr_bias_ratio if lr_bias_ratio is not None else lr
        self.b -= lr_b * grad_b_L

        # ---- 7) Return diagnostics if requested ----
        if return_diagnostics:
            return {
                'grad_U_norm': np.linalg.norm(grad_U_L),
                'grad_b_norm': np.linalg.norm(grad_b_L),
                'grad_U_max': np.abs(grad_U_L).max(),
                'grad_b_max': np.abs(grad_b_L).max(),
                'loss': 0.5 * (p - r)**2,  # MSE loss
            }

    def warm_start(self,
                contexts: Union[np.ndarray, float],
                signals: Union[np.ndarray, float],
                rewards: Union[np.ndarray, float],
                epochs: int = 1,
                lr: float = 1e-3,
                lr_bias_ratio: Optional[float] = None,
                expl_scale: Optional[float] = None,
                ema_decay: Optional[float] = None,
                tau: Optional[float] = None,
                alpha: Optional[float] = None,
                weight_decay: Optional[float] = None,
                monitor_every: Optional[int] = None,
                verbose: bool = False) -> Optional[dict]:
        """
        Pre-train model on historical curator decisions.

        Initializes model parameters from past data before online learning.
        Can process either a single sample or a batch with multiple epochs.

        Hyperparameters can be temporarily overridden during warm-start to reduce
        exploration noise and speed up learning, then restored for online learning.

        Args:
            contexts: Context features - single (context_dim,) or batch (n_samples, context_dim)
            signals: Signal values - single (num_signals,) or batch (n_samples, num_signals)
            rewards: Binary rewards - single float or batch (n_samples,)
            epochs: Number of passes through batch data (only used for batches)
            lr: Learning rate for warm-start (typically lower than online lr)
            lr_bias_ratio: Ratio for bias learning rate during warm-start (None = same as lr).
                          Use 0.01-0.1 to slow bias learning, preventing signal weight collapse
                          when training on large datasets. Only applies during warm-start;
                          online updates use full learning rate for bias.
            expl_scale: Override exploration scale during warm-start (None = keep current)
            ema_decay: Override EMA decay during warm-start (None = keep current)
            tau: Override softmax temperature during warm-start (None = keep current)
            alpha: Override uniform mixing weight during warm-start (None = keep current)
            weight_decay: Override L2 regularization during warm-start (None = keep current)
            monitor_every: If provided, collect diagnostics every N updates (None = no monitoring)
            verbose: If True, print progress during training

        Returns:
            Optional dict with training history if monitor_every is set

        Example:
            >>> # Monitor training every 500 updates with regularization and slow bias
            >>> history = cts.warm_start(contexts, signals, rewards, epochs=5, lr=1e-3,
            ...                          lr_bias_ratio=0.01, expl_scale=0.01,
            ...                          weight_decay=1e-4, monitor_every=500, verbose=True)
            >>> # Plot training curves
            >>> plt.plot(history['update_steps'], history['grad_U_norm'])
        """
        # Save original hyperparameters
        original_expl_scale = self.expl_scale
        original_ema_decay = self.ema_decay
        original_tau = self.tau
        original_alpha = self.alpha
        original_weight_decay = self.weight_decay

        # Apply temporary overrides if provided
        if expl_scale is not None:
            self.expl_scale = expl_scale
        if ema_decay is not None:
            self.ema_decay = ema_decay
        if tau is not None:
            self.tau = tau
        if alpha is not None:
            self.alpha = alpha
        if weight_decay is not None:
            self.weight_decay = weight_decay

        # Initialize monitoring if requested
        history = None
        if monitor_every is not None:
            history = {
                'update_steps': [],
                'epochs': [],
                'U_norm': [],
                'b_norm': [],
                'U_max': [],
                'b_max': [],
                'h_U_mean': [],
                'h_U_median': [],
                'h_U_min': [],
                'h_U_max': [],
                'h_b_mean': [],
                'grad_U_norm': [],
                'grad_b_norm': [],
                'grad_U_max': [],
                'grad_b_max': [],
                'loss': [],
                'noise_std_U': [],
                'noise_std_b': [],
            }

        try:
            # Single sample case
            if isinstance(rewards, float):
                c_t = contexts
                s_i = signals
                r = rewards
                self.update(c_t, s_i, r, lr=lr, lr_bias_ratio=lr_bias_ratio)
            # Batch case - run for specified epochs
            else:
                n = len(rewards)
                update_count = 0

                for epoch in range(epochs):
                    perm = self.rng.permutation(n)  # Shuffle each epoch
                    epoch_losses = []

                    for i, idx in enumerate(perm):
                        c_t = contexts[idx]
                        s_i = signals[idx]
                        r = rewards[idx]

                        # Update with diagnostics if monitoring
                        if monitor_every is not None:
                            diag = self.update(c_t, s_i, r, lr=lr, lr_bias_ratio=lr_bias_ratio, return_diagnostics=True)
                            epoch_losses.append(diag['loss'])

                            # Record metrics every N updates
                            if (update_count + 1) % monitor_every == 0:
                                history['update_steps'].append(update_count + 1)
                                history['epochs'].append(epoch + (i + 1) / n)
                                history['U_norm'].append(np.linalg.norm(self.U))
                                history['b_norm'].append(np.linalg.norm(self.b))
                                history['U_max'].append(np.abs(self.U).max())
                                history['b_max'].append(np.abs(self.b).max())
                                history['h_U_mean'].append(self.h_U.mean())
                                history['h_U_median'].append(np.median(self.h_U))
                                history['h_U_min'].append(self.h_U.min())
                                history['h_U_max'].append(self.h_U.max())
                                history['h_b_mean'].append(self.h_b.mean())
                                history['grad_U_norm'].append(diag['grad_U_norm'])
                                history['grad_b_norm'].append(diag['grad_b_norm'])
                                history['grad_U_max'].append(diag['grad_U_max'])
                                history['grad_b_max'].append(diag['grad_b_max'])
                                history['loss'].append(diag['loss'])
                                history['noise_std_U'].append(np.sqrt(self.expl_scale / self.h_U.mean()))
                                history['noise_std_b'].append(np.sqrt(self.expl_scale / self.h_b.mean()))
                        else:
                            self.update(c_t, s_i, r, lr=lr, lr_bias_ratio=lr_bias_ratio)

                        update_count += 1

                    # Print epoch summary if verbose
                    if verbose:
                        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                        print(f"Epoch {epoch + 1}/{epochs}: "
                              f"avg_loss={avg_loss:.6f}, "
                              f"|U|={np.linalg.norm(self.U):.4f}, "
                              f"|b|={np.linalg.norm(self.b):.4f}, "
                              f"h_U_median={np.median(self.h_U):.6f}")

        finally:
            # Always restore original hyperparameters
            self.expl_scale = original_expl_scale
            self.ema_decay = original_ema_decay
            self.tau = original_tau
            self.alpha = original_alpha
            self.weight_decay = original_weight_decay

        return history


    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk for later use.

        Saves model parameters, hyperparameters, and random state so you can
        resume training or make predictions later with identical behavior.

        Args:
            path: File path (will create .npz for params and .rng.pkl for random state)

        Files created:
            <path>.npz: Model parameters (U, b, h_U, h_b) and hyperparameters
            <path>.rng.pkl: Random number generator state
        """
        path = Path(path)

        np.savez(
            path.with_suffix(path.suffix or ".npz"),
            U=self.U,
            b=self.b,
            h_U=self.h_U,
            h_b=self.h_b,
            lr=np.array(self.lr),
            expl_scale=np.array(self.expl_scale),
            ema_decay=np.array(self.ema_decay),
            h0=np.array(self.h0),
            alpha=np.array(self.alpha),
            tau=np.array(self.tau),
            match_loss_weight=np.array(self.match_loss_weight),
            weight_decay=np.array(self.weight_decay)
        )

        # Save RNG state for reproducibility
        rng_state = self.rng.get_state()
        with open(path.with_suffix(".rng.pkl"), "wb") as f:
            pickle.dump(rng_state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContextualThompsonSampler":
        """
        Load a saved model from disk.

        Restores model parameters, hyperparameters, and random state exactly
        as they were when saved.

        Args:
            path: File path (same as used in save())

        Returns:
            Loaded ContextualThompsonSampler instance
        """
        path = Path(path)
        npz = path.with_suffix(path.suffix or ".npz")
        data = np.load(npz)

        # Reconstruct sampler with saved hyperparameters
        sampler = cls(
            num_signals=data["U"].shape[0],
            context_dim=data["U"].shape[1],
            lr=float(data["lr"]),
            expl_scale=float(data["expl_scale"]),
            ema_decay=float(data["ema_decay"]),
            h0=float(data["h0"]),
            tau=float(data["tau"]),
            alpha=float(data["alpha"]),
            match_loss_weight=float(data["match_loss_weight"]),
            weight_decay=float(data.get("weight_decay", 0.0)),  # Backward compatible
            random_state=None,  # Will restore RNG state below
        )

        # Restore learned parameters
        sampler.U = data["U"]
        sampler.b = data["b"]
        sampler.h_U = data["h_U"]
        sampler.h_b = data["h_b"]

        # Restore RNG state
        rng_file = path.with_suffix(".rng.pkl")
        if rng_file.exists():
            with open(rng_file, "rb") as f:
                state = pickle.load(f)
            sampler.rng.set_state(state)

        return sampler




