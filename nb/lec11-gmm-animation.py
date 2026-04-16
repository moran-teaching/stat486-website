import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec


# --- Generate synthetic 2D data from 3 Gaussians ---
np.random.seed(1)
K = 3
n_per = 80
true_mus = np.array([[-2.5, -2.0], [2.5, -1.5], [0.0, 2.8]])
true_covs = np.array([
    [[1.0, 0.3], [0.3, 0.6]],
    [[0.7, -0.2], [-0.2, 1.0]],
    [[0.9, 0.0], [0.0, 0.9]],
])

X = np.vstack([
    np.random.multivariate_normal(true_mus[k], true_covs[k], n_per)
    for k in range(K)
])
N = X.shape[0]


# --- EM for GMM ---
def gaussian_pdf(X, mu, cov):
    d = X - mu
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (2 * np.pi * np.sqrt(det))
    exponent = -0.5 * np.einsum('ni,ij,nj->n', d, inv, d)
    return norm * np.exp(exponent)


def e_step(X, pis, mus, covs):
    N = X.shape[0]
    K = len(pis)
    r = np.zeros((N, K))
    for k in range(K):
        r[:, k] = pis[k] * gaussian_pdf(X, mus[k], covs[k])
    r /= r.sum(axis=1, keepdims=True) + 1e-12
    return r


def m_step(X, r):
    N, K = r.shape
    Nk = r.sum(axis=0)
    pis = Nk / N
    mus = (r.T @ X) / Nk[:, None]
    covs = np.zeros((K, 2, 2))
    for k in range(K):
        d = X - mus[k]
        covs[k] = (r[:, k:k+1] * d).T @ d / Nk[k]
        covs[k] += 1e-6 * np.eye(2)
    return pis, mus, covs


# --- Initialization (deliberately poor so we can see convergence) ---
np.random.seed(7)
init_idx = np.random.choice(N, K, replace=False)
mus = X[init_idx].copy()
covs = np.array([np.eye(2) * 2.0 for _ in range(K)])
pis = np.ones(K) / K


# --- Run EM, capturing state at each half-step ---
# We alternate: show E-step result (responsibilities with current params),
# then M-step result (new params). Each frame is one half-step.
n_iters = 20
states = []

# Initial E-step with initial params
r = e_step(X, pis, mus, covs)
states.append(('E', pis.copy(), mus.copy(), covs.copy(), r.copy()))

for it in range(n_iters):
    pis, mus, covs = m_step(X, r)
    states.append(('M', pis.copy(), mus.copy(), covs.copy(), r.copy()))
    r = e_step(X, pis, mus, covs)
    states.append(('E', pis.copy(), mus.copy(), covs.copy(), r.copy()))


# --- Plotting ---
colors = ['#e41a1c', '#377eb8', '#4daf4a']


def cov_ellipse(mu, cov, ax, color, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    e = Ellipse(xy=mu, width=width, height=height, angle=angle,
                edgecolor=color, facecolor='none', lw=2, **kwargs)
    ax.add_patch(e)


fig = plt.figure(figsize=(9, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1], hspace=0.3, wspace=0.25)
ax_main = fig.add_subplot(gs[0, :])
ax_resp = [fig.add_subplot(gs[1, k]) for k in range(K)]

xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)


def animate(frame):
    step_type, pis_f, mus_f, covs_f, r_f = states[frame]

    ax_main.clear()
    ax_main.scatter(X[:, 0], X[:, 1], c='lightgray', s=18, edgecolor='none')
    for k in range(K):
        ax_main.scatter(mus_f[k, 0], mus_f[k, 1], marker='X',
                        s=220, c=colors[k], edgecolor='black', lw=1.5,
                        zorder=5)
        cov_ellipse(mus_f[k], covs_f[k], ax_main, colors[k])
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.set_xlabel('$x_1$')
    ax_main.set_ylabel('$x_2$')
    iter_num = frame // 2
    label = 'E-step' if step_type == 'E' else 'M-step'
    ax_main.set_title(f'GMM via EM — iter {iter_num}, {label}',
                      fontsize=13, fontweight='bold')

    for k in range(K):
        ax_resp[k].clear()
        sizes = 5 + 60 * r_f[:, k]
        alphas = np.clip(r_f[:, k], 0.05, 1.0)
        ax_resp[k].scatter(X[:, 0], X[:, 1], s=sizes, c=colors[k],
                           alpha=alphas, edgecolor='none')
        ax_resp[k].scatter(mus_f[k, 0], mus_f[k, 1], marker='X',
                           s=120, c=colors[k], edgecolor='black', lw=1.2)
        ax_resp[k].set_xlim(xlim)
        ax_resp[k].set_ylim(ylim)
        ax_resp[k].set_xticks([])
        ax_resp[k].set_yticks([])
        ax_resp[k].set_title(f'$p(z={k+1}\\mid x,\\theta)$', fontsize=11)

    return [ax_main] + ax_resp


anim = FuncAnimation(fig, animate, frames=len(states),
                     interval=700, blit=False)
anim.save('fig/lec11-gmm-em.gif', writer='pillow', fps=1.5)
plt.close(fig)

print(f'Saved fig/lec11-gmm-em.gif ({len(states)} frames)')
