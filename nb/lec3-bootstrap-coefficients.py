import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from IPython.display import HTML
import matplotlib.animation as animation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

house = pd.read_csv("nb/data/rainier_valley_house.csv")
# We will consider this subset of features
features = ["floors", "grade", "condition", "view", "sqft_living",
            "sqft_lot", "sqft_basement", "yr_built", "yr_renovated",
            "bedrooms", "bathrooms", "lat", "long"]

# filter houses greater than 1 mil
house = house[house['price'] < 1e6]

Y = house['price'] / 1000
X = house[features]

## Standardize the features
scaler = StandardScaler()
X_stan = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_stan, Y, test_size=100, train_size=400, random_state=2)
Y_train = Y_train.to_numpy()

n_train = X_train.shape[0]

linreg = LinearRegression()
linreg.fit(X_train, Y_train)
beta = np.zeros((1+X_train.shape[1]))
beta[0] = linreg.intercept_
beta[1:] = linreg.coef_

n_boot = 1000

# Precompute bootstrap samples for animation
bootstrap_inds = [np.random.choice(range(1, n_train), n_train, replace=True) for _ in range(n_boot)]

boot_bhats = np.zeros((n_boot, 1+X_train.shape[1]))
for b in range(n_boot):
    X_boot = X_train[bootstrap_inds[b]]
    Y_boot = Y_train[bootstrap_inds[b]]
    linreg = LinearRegression()
    linreg.fit(X_boot, Y_boot)
    boot_bhats[b, 0] = linreg.intercept_
    boot_bhats[b, 1:] = linreg.coef_

bins_beta = np.linspace(boot_bhats[:, 4].min(), boot_bhats[:, 4].max(), 20)


# Set up the figure and subplots (3 columns now)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))

# ax0: original sample
ax0.scatter(X_train[:, 4], Y_train)
ax0.axline((0,beta[0]), slope=beta[0], color='red')

ax0.set_ylim(Y_train.min(), Y_train.max())
ax0.set_xlim(X_train[:,4].min(), X_train[:,4].max())

ax0.set_title('Original Sample')
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.legend()

# ax1: first bootstrap sample
ax1.scatter(X_train[bootstrap_inds[0], 4], Y_train[bootstrap_inds[0]])
ax1.axline((0,boot_bhats[0, 0]), slope=boot_bhats[0, 4], color='red')
ax1.set_ylim(Y_train.min(), Y_train.max())
ax1.set_xlim(X_train[:,4].min(), X_train[:,4].max())

ax1.set_title('Bootstrap Sample')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

# ax2: histogram of bootstrap medians (empty initially)
hist_medians = ax2.hist([], bins=bins_beta, alpha=0.7, color='green', label='Bootstrap Betas')
ax2.set_title('Histogram of Bootstrap Betas')
ax2.set_xlabel('Beta')
ax2.set_ylabel('Frequency')
ax2.legend()

def animate(frame):
    # ax1: first bootstrap sample
    ax1.clear()
    ax1.scatter(X_train[bootstrap_inds[frame], 4], Y_train[bootstrap_inds[frame]])
    ax1.axline((0,boot_bhats[frame, 0]), slope=boot_bhats[frame, 4], color='red')
    ax1.set_ylim(Y_train.min(), Y_train.max())
    ax1.set_xlim(X_train[:,4].min(), X_train[:,4].max())

    ax1.set_title('Bootstrap Sample')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # ax2: histogram of bootstrap medians (empty initially)
    ax2.clear()
    hist_medians = ax2.hist(boot_bhats[:frame+1, 4], bins=bins_beta, alpha=0.7, color='green', label='Bootstrap Beta')
    ax2.set_title('Histogram of Bootstrap Betas')
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    ax0.clear()
    # ax0: original sample
    ax0.scatter(X_train[:, 4], Y_train)
    ax0.axline((0,beta[0]), slope=beta[4], color='red')
    ax0.set_ylim(Y_train.min(), Y_train.max())
    ax0.set_xlim(X_train[:,4].min(), X_train[:,4].max())

    ax0.set_title('Original Sample')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.legend()


anim = animation.FuncAnimation(fig, animate, frames=n_boot, interval=300, repeat=False)
anim.save('nb/fig/boot_linreg.gif', writer='pillow')
