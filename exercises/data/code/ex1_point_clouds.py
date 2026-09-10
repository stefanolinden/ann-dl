"""Exercise 1 - Point Clouds: Geometry and Spread in 2D.

Generates Figures 1, 2 and 3 and prints all the numbers required in the report
(mixing rates and separation ratios r_ij).
"""
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "../figures"
rng = np.random.default_rng(42)

# ---- Class parameters (base standard deviations, s = 1.0) ----
means = {
    0: np.array([2.0, 3.0]),
    1: np.array([5.0, 6.0]),
    2: np.array([8.0, 1.0]),
    3: np.array([15.0, 4.0]),
}
stds = {
    0: np.array([0.8, 2.5]),
    1: np.array([1.2, 1.9]),
    2: np.array([0.9, 0.9]),
    3: np.array([0.5, 2.0]),
}
n_per_class = 100
colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def generate_clouds(scale, rng):
    """Draw n_per_class Gaussian points per class with std multiplied by scale."""
    data = {}
    for c in means:
        pts = rng.normal(loc=means[c], scale=stds[c] * scale, size=(n_per_class, 2))
        data[c] = pts
    return data


# ---------------------------------------------------------------
# Item A - Figure 1: the four clouds at s = 1
# ---------------------------------------------------------------
data_s1 = generate_clouds(1.0, rng)

fig, ax = plt.subplots(figsize=(7, 6))
for c, pts in data_s1.items():
    ax.scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.6, color=colors[c], label=f"Class {c}")
    ax.scatter(*means[c], color=colors[c], marker="X", s=200, edgecolor="black", linewidth=1.5)
ax.set_title("Figure 1 - Point clouds for the 4 classes (s = 1.0)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Item B - Figure 2: 4 subplots for s in {0.5, 1.0, 2.0, 4.0}
# ---------------------------------------------------------------
scales = [0.5, 1.0, 2.0, 4.0]
datasets_by_scale = {}

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
for ax, s in zip(axes, scales):
    data_s = generate_clouds(s, rng)
    datasets_by_scale[s] = data_s
    for c, pts in data_s.items():
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, color=colors[c], label=f"Class {c}")
    ax.set_title(f"s = {s}")
    ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()
fig.suptitle("Figure 2 - Point clouds for different spread factors s")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure2.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Item B.2 - separation ratio r_ij for s = 1.0
# ---------------------------------------------------------------
def mean_std(c):
    return (stds[c][0] + stds[c][1]) / 2.0


pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
print("\nSeparation ratios r_ij at s = 1.0:")
r_values = {}
for i, j in pairs:
    dist = np.linalg.norm(means[i] - means[j])
    denom = mean_std(i) + mean_std(j)
    r = dist / denom
    r_values[(i, j)] = r
    print(f"  r_{i}{j} = {r:.3f}  (dist={dist:.3f}, sigma_bar_sum={denom:.3f})")

min_pair = min(r_values, key=r_values.get)
min_r = r_values[min_pair]
print(f"\nSmallest r_ij at s=1.0: pair {min_pair} with r = {min_r:.3f}")
print(f"Predicted smallest r_ij at s=2.0 (scales as 1/s): {min_r / 2:.3f}")

# ---------------------------------------------------------------
# Item B.3/B.4 - mixing rate per scale + Figure 3
# ---------------------------------------------------------------
def mixing_rate(data_s):
    """Fraction of points whose nearest class mean is not their own class."""
    all_means = np.array([means[c] for c in sorted(means)])
    n_wrong = 0
    n_total = 0
    for c, pts in data_s.items():
        dists = np.linalg.norm(pts[:, None, :] - all_means[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        n_wrong += np.sum(nearest != c)
        n_total += len(pts)
    return n_wrong / n_total


print("\nMixing rate per scale:")
mixing_rates = []
for s in scales:
    rate = mixing_rate(datasets_by_scale[s])
    mixing_rates.append(rate)
    print(f"  s = {s}: mixing rate = {rate:.3f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(scales, mixing_rates, marker="o", color="tab:purple", label="Mixing rate")
ax.set_title("Figure 3 - Mixing rate vs spread factor s")
ax.set_xlabel("Spread factor s")
ax.set_ylabel("Mixing rate")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure3.png", dpi=150)
plt.close(fig)

print("\nDone. Figures saved to", FIG_DIR)
