import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from IPython.display import HTML
import matplotlib.animation as animation

np.random.seed(42)
n = 30
n_samp = 1000

# get population distribution for plotting
x_vals = np.random.exponential(size=1000000)

# Set up the figure and subplots (3 columns now)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))

# Precompute samples for animation
samples = [np.random.exponential(size=n) for _ in range(n_samp)]
medians = [np.median(sample) for sample in samples]

# Bins for histograms
bins_sample = np.linspace(0, 10, 20)
bins_means = np.linspace(min(medians), max(medians), 20)

# ax0: population
ax0.hist(x_vals, bins=500, alpha=0.7, color='gray', label='Population', density=True)
ax0.axvline(np.median(x_vals), color='red', linestyle='--', linewidth=2)
ax0.text(np.median(x_vals) + 0.1, ax0.get_ylim()[1] * 0.8, f'Median: {np.median(x_vals):.2f}', fontsize=12, color='red')
ax0.set_title('Population')
ax0.set_xlabel('Value')
ax0.set_ylabel('Density')


# ax1: first sample
hist_sample = ax1.hist(samples[0], bins=bins_sample, alpha=0.7, color='blue', label='Sample')
mean_line = ax1.axvline(np.median(samples[0]), color='red', linestyle='--', linewidth=2)
mean_text = ax1.text(np.median(samples[0]) + 0.1, ax1.get_ylim()[1] * 0.8, f'Median: {np.median(samples[0]):.3f}', fontsize=12, color='red')
ax1.set_title('Sample')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.legend()

# ax2: histogram of  means (empty initially)
hist_means = ax2.hist([], bins=bins_means, alpha=0.7, color='green', label='Sample Medians')
ax2.set_title('Histogram of Medians')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.legend()

def animate(frame):
    # ax1: update bootstrap sample
    ax1.clear()
    ax1.hist(samples[frame], bins=bins_sample, alpha=0.7, color='blue', label='Sample from Population')
    current_mean = np.median(samples[frame])
    ax1.axvline(current_mean, color='red', linestyle='--', linewidth=2)
    ax1.text(current_mean + 0.1, ax1.get_ylim()[1] * 0.8, f'Median: {current_mean:.3f}', fontsize=12, color='red')
    ax1.set_title('Sample from Population')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # ax2: update histogram of means
    ax2.clear()
    ax2.hist(medians[:frame+1], bins=bins_means, alpha=0.7, color='green', label='Sample Medians')
    ax2.set_title('Histogram of Sample Medians')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # ax0: keep original sample unchanged
    ax0.clear()
    ax0.hist(x_vals, bins=500, alpha=0.7, color='gray', label='Population', density=True)
    ax0.axvline(np.median(x_vals), color='red', linestyle='--', linewidth=2)
    ax0.text(np.median(x_vals) + 0.1, ax0.get_ylim()[1] * 0.8, f'Median: {np.median(x_vals):.2f}', fontsize=12, color='red')
    ax0.set_title('Population')
    ax0.set_xlabel('Value')
    ax0.set_ylabel('Density')


anim = animation.FuncAnimation(fig, animate, frames=n_samp, interval=300, repeat=False)
anim.save('nb/fig/sample_animation.gif', writer='pillow')
