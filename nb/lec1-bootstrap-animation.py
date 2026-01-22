import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from IPython.display import HTML
import matplotlib.animation as animation

np.random.seed(42)
n = 30
x = np.random.exponential(scale=1, size=n)  

# calculate sample estimate
muhat = np.mean(x)

n_boot = 1000

# Set up the figure and subplots (3 columns now)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))

# Precompute bootstrap samples for animation
bootstrap_samples = [np.random.choice(x, n, replace=True) for _ in range(n_boot)]
bootstrap_means = [np.mean(sample) for sample in bootstrap_samples]

# Bins for histograms
bins_sample = np.linspace(x.min(), x.max(), 20)
bins_means = np.linspace(min(bootstrap_means), max(bootstrap_means), 20)

# ax0: original sample
ax0.hist(x, bins=bins_sample, alpha=0.7, color='gray', label='Original Sample')
ax0.axvline(np.mean(x), color='red', linestyle='--', linewidth=2)
ax0.text(np.mean(x) + 0.1, ax0.get_ylim()[1] * 0.8, f'Mean: {np.mean(x):.3f}', fontsize=12, color='red')
ax0.set_title('Original Sample')
ax0.set_xlabel('Value')
ax0.set_ylabel('Frequency')
ax0.legend()

# ax1: first bootstrap sample
hist_sample = ax1.hist(bootstrap_samples[0], bins=bins_sample, alpha=0.7, color='blue', label='Bootstrap Sample')
mean_line = ax1.axvline(np.mean(bootstrap_samples[0]), color='red', linestyle='--', linewidth=2)
mean_text = ax1.text(np.mean(bootstrap_samples[0]) + 0.1, ax1.get_ylim()[1] * 0.8, f'Mean: {np.mean(bootstrap_samples[0]):.3f}', fontsize=12, color='red')
ax1.set_title('Bootstrap Sample')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.legend()

# ax2: histogram of bootstrap means (empty initially)
hist_means = ax2.hist([], bins=bins_means, alpha=0.7, color='green', label='Bootstrap Means')
ax2.set_title('Histogram of Bootstrap Means')
ax2.set_xlabel('Mean Value')
ax2.set_ylabel('Frequency')
ax2.legend()

def animate(frame):
    # ax1: update bootstrap sample
    ax1.clear()
    ax1.hist(bootstrap_samples[frame], bins=bins_sample, alpha=0.7, color='blue', label='Bootstrap Sample')
    current_mean = np.mean(bootstrap_samples[frame])
    ax1.axvline(current_mean, color='red', linestyle='--', linewidth=2)
    ax1.text(current_mean + 0.1, ax1.get_ylim()[1] * 0.8, f'Mean: {current_mean:.3f}', fontsize=12, color='red')
    ax1.set_title('Bootstrap Sample')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # ax2: update histogram of means
    ax2.clear()
    ax2.hist(bootstrap_means[:frame+1], bins=bins_means, alpha=0.7, color='green', label='Bootstrap Means')
    ax2.set_title('Histogram of Bootstrap Means')
    ax2.set_xlabel('Mean Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # ax0: keep original sample unchanged
    ax0.clear()
    ax0.hist(x, bins=bins_sample, alpha=0.7, color='gray', label='Original Sample')
    ax0.axvline(np.mean(x), color='red', linestyle='--', linewidth=2)
    ax0.text(np.mean(x) + 0.1, ax0.get_ylim()[1] * 0.8, f'Mean: {np.mean(x):.3f}', fontsize=12, color='red')
    ax0.set_title('Original Sample')
    ax0.set_xlabel('Value')
    ax0.set_ylabel('Frequency')
    ax0.legend()

anim = animation.FuncAnimation(fig, animate, frames=n_boot, interval=300, repeat=False)
anim.save('bootstrap_animation.gif', writer='pillow')
