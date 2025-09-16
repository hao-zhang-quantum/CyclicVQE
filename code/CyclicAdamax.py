# =====================================================
#  Cyclic Adamax Updater
# =====================================================
import numpy as np

class CyclicAdamaxUpdater:
    """
    Cyclic Adamax optimizer with soft reset and optional adaptive restart.
    """

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 restart_period=200, soft_reset_alpha=0,
                 use_adaptive_restart=False, grad_window=20, plateau_tol=1e-3):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.u = None
        self.t = 0

        # Restart parameters
        self.restart_period = restart_period
        self.soft_reset_alpha = soft_reset_alpha

        # Adaptive restart parameters
        self.use_adaptive_restart = use_adaptive_restart
        self.grad_window = grad_window
        self.plateau_tol = plateau_tol
        self.grad_history = []

    def _expand(self, new_shape):
        """Expand moment vectors if new gradient shape is larger."""
        if self.m is None:
            self.m = np.zeros(new_shape)
            self.u = np.zeros(new_shape)
        elif self.m.shape != new_shape:
            old_size = self.m.shape[0]
            new_size = new_shape[0]
            if new_size > old_size:
                self.m = np.pad(self.m, (0, new_size - old_size), constant_values=0)
                self.u = np.pad(self.u, (0, new_size - old_size), constant_values=0)

    def reset(self):
        """Manually trigger restart (optional)"""
        self.m = np.zeros_like(self.m)
        self.u = np.zeros_like(self.u)
        self.t = 0

    def soft_reset(self, reset_alpha=None):
        """Soft reset: shrink moment estimates instead of zeroing them."""
        if reset_alpha is None:
            reset_alpha = self.soft_reset_alpha
        self.m *= reset_alpha
        self.u *= reset_alpha
        self.t = 0  # reset time step

    def reset_idx(self, idx, reset_alpha=0):
        """Reset moment vectors for specific indices."""
        self.m[idx] *= reset_alpha
        self.u[idx] *= reset_alpha

    def _adaptive_restart_trigger(self, grad_norm):
        """Check if adaptive restart should be triggered (plateau detection only)."""
        if not self.use_adaptive_restart:
            return False

        self.grad_history.append(grad_norm)
        if len(self.grad_history) > self.grad_window:
            self.grad_history.pop(0)

            # Plateau detection: gradient norm variation is very small
            if np.std(self.grad_history) < self.plateau_tol:
                return True

        return False

    def step(self, x, grad):
        """Perform one optimization step."""
        grad = np.nan_to_num(grad, nan=0.0)
        grad = np.clip(grad, -0.1, 0.1)

        self._expand(grad.shape)
        self.t += 1

        grad_norm = np.linalg.norm(grad)

        # Periodic restart
        if self.t % self.restart_period == 0:
            print("⚡ AdamaxUpdater triggered soft reset (periodic)!")
            self.soft_reset()
            return x, grad_norm

        # Adaptive restart
        if self._adaptive_restart_trigger(grad_norm):
            print("🔄 AdamaxUpdater triggered adaptive soft reset!")
            self.soft_reset()
            return x, grad_norm

        # Standard Adamax update
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.u = np.maximum(self.beta2 * self.u, np.abs(grad))
        m_hat = self.m / (1 - self.beta1**self.t)

        x = x - self.lr * m_hat / (self.u + self.epsilon)
        x = x / np.linalg.norm(x)

        return x, grad_norm