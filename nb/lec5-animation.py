import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate true underlying data
np.random.seed(42)
X_true = np.linspace(0, 1, 100)
y_true = 2 * X_true**2 - 3 * X_true + 1 + 0.1 * np.random.randn(100)

# True function (without noise)
def true_function(X):
    return 2 * X**2 - 3 * X + 1

# Parameters
num_datasets = 100
X_plot = np.linspace(0, 1, 100)
y_true_func = true_function(X_true)


def make_animation(fit_func, filename, title_prefix):
    """Create a bias-variance animation for a given fitting function."""
    np.random.seed(42)
    all_preds = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold')

    def init():
        for ax in axes:
            ax.clear()
        return axes

    def animate(frame_num):
        indices = np.sort(np.random.choice(len(X_true), size=20, replace=False))
        X_train = X_true[indices]
        y_train = y_true[indices] + 0.1 * np.random.randn(20)

        y_pred = fit_func(X_train, y_train, X_plot)
        all_preds.append(y_pred)

        # Left plot: current dataset and fit
        axes[0].clear()
        axes[0].plot(X_true, y_true_func, color='blue', label='True function', linewidth=2)
        axes[0].scatter(X_train, y_train, color='red', label='Current dataset', s=20)
        axes[0].plot(X_plot, y_pred, color='black', label='Current fit')
        axes[0].set_title('Current Dataset and Fit')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].legend()
        axes[0].set_ylim(-2, 2)

        # Middle plot: all previous fits (variance)
        axes[1].clear()
        axes[1].plot(X_true, y_true_func, color='blue', linewidth=2)
        for j, prev_pred in enumerate(all_preds[:-1]):
            axes[1].plot(X_plot, prev_pred, color='gray', alpha=0.5,
                         label='Previous fit' if j == 0 else "")
        axes[1].set_title('Previous Fits (Variance)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('y')
        axes[1].set_ylim(-2, 2)

        # Right plot: average of fitted lines (bias)
        axes[2].clear()
        axes[2].plot(X_true, y_true_func, color='blue', linewidth=2)
        if all_preds:
            avg_pred = np.mean(np.array(all_preds), axis=0)
            axes[2].plot(X_plot, avg_pred, color='green', label='Average fit')
        axes[2].set_title('Average of Fits (Bias)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('y')
        axes[2].legend()
        axes[2].set_ylim(-2, 2)

        return axes

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_datasets,
                         interval=1000, blit=False)
    anim.save(filename, writer='pillow', fps=1)
    plt.close(fig)


# --- Fitting functions ---

def fit_linear(X_train, y_train, X_plot):
    coeffs = np.polyfit(X_train, y_train, 1)
    return np.polyval(coeffs, X_plot)

def fit_spline_regularized(X_train, y_train, X_plot):
    model = UnivariateSpline(X_train, y_train, k=3, s=1.0)
    return model(X_plot)

def fit_spline_unregularized(X_train, y_train, X_plot):
    model = UnivariateSpline(X_train, y_train, k=3, s=0)
    return model(X_plot)


# --- Generate all three animations ---

make_animation(fit_linear, 'fig/bias-variance-simple.gif', 'Linear Regression')
make_animation(fit_spline_regularized, 'fig/bias-variance-regularized.gif', 'Spline (Regularized)')
make_animation(fit_spline_unregularized, 'fig/bias-variance-complex.gif', 'Spline (No Regularization)')
