"""Exercise 2 - Non-Linearity in Higher Dimensions.

Generates Dataset I (shifted Gaussians) and Dataset II (concentric shells) in
5D, projects them to 2D with PCA (Figure 4), and plots radius histograms
(Figure 5).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FIG_DIR = "../figures"
rng = np.random.default_rng(42)
n_per_class = 500

# ---------------------------------------------------------------
# Item A - Dataset I: shifted Gaussians
# ---------------------------------------------------------------
mu_A = np.zeros(5)
mu_B = np.full(5, 1.5)

cov_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
cov_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

class_A = rng.multivariate_normal(mu_A, cov_A, size=n_per_class)
class_B = rng.multivariate_normal(mu_B, cov_B, size=n_per_class)
dataset1 = np.vstack([class_A, class_B])
labels1 = np.array([0] * n_per_class + [1] * n_per_class)

# ---------------------------------------------------------------
# Item B - Dataset II: concentric shells
# ---------------------------------------------------------------
def sample_shell(n, radius_mean, radius_std, rng, dim=5):
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.normal(loc=radius_mean, scale=radius_std, size=n)
    return directions * radii[:, None]


class_C = sample_shell(n_per_class, 2.0, 0.4, rng)  # core
class_D = sample_shell(n_per_class, 5.0, 0.4, rng)  # shell
dataset2 = np.vstack([class_C, class_D])
labels2 = np.array([0] * n_per_class + [1] * n_per_class)

# ---------------------------------------------------------------
# Item C - PCA projection (Figure 4) + explained variance
# ---------------------------------------------------------------
pca1 = PCA(n_components=2)
proj1 = pca1.fit_transform(dataset1)

pca2 = PCA(n_components=2)
proj2 = pca2.fit_transform(dataset2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for lbl, name in [(0, "A"), (1, "B")]:
    mask = labels1 == lbl
    axes[0].scatter(proj1[mask, 0], proj1[mask, 1], s=12, alpha=0.6, label=f"Class {name}")
axes[0].set_title("Dataset I (shifted Gaussians) - PCA")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

for lbl, name in [(0, "C (core)"), (1, "D (shell)")]:
    mask = labels2 == lbl
    axes[1].scatter(proj2[mask, 0], proj2[mask, 1], s=12, alpha=0.6, label=f"Class {name}")
axes[1].set_title("Dataset II (concentric shells) - PCA")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

fig.suptitle("Figure 4 - PCA projection of Dataset I and Dataset II")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure4.png", dpi=150)
plt.close(fig)

var1 = pca1.explained_variance_ratio_.sum()
var2 = pca2.explained_variance_ratio_.sum()
print(f"Explained variance (PC1+PC2) - Dataset I:  {var1:.3f}")
print(f"Explained variance (PC1+PC2) - Dataset II: {var2:.3f}")

# ---------------------------------------------------------------
# Item C - center distance and radius histograms (Figure 5)
# ---------------------------------------------------------------
dist1 = np.linalg.norm(class_A.mean(axis=0) - class_B.mean(axis=0))
dist2 = np.linalg.norm(class_C.mean(axis=0) - class_D.mean(axis=0))
print(f"Distance between class centers - Dataset I:  {dist1:.3f}")
print(f"Distance between class centers - Dataset II: {dist2:.3f}")

radii1_A = np.linalg.norm(class_A, axis=1)
radii1_B = np.linalg.norm(class_B, axis=1)
radii2_C = np.linalg.norm(class_C, axis=1)
radii2_D = np.linalg.norm(class_D, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(radii1_A, bins=30, alpha=0.6, label="Class A")
axes[0].hist(radii1_B, bins=30, alpha=0.6, label="Class B")
axes[0].set_title("Dataset I - ||x|| per class")
axes[0].set_xlabel("||x||")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].hist(radii2_C, bins=30, alpha=0.6, label="Class C (core)")
axes[1].hist(radii2_D, bins=30, alpha=0.6, label="Class D (shell)")
axes[1].set_title("Dataset II - ||x|| per class")
axes[1].set_xlabel("||x||")
axes[1].set_ylabel("Count")
axes[1].legend()

fig.suptitle("Figure 5 - Radius histograms")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure5.png", dpi=150)
plt.close(fig)

print("\nDone. Figures saved to", FIG_DIR)
