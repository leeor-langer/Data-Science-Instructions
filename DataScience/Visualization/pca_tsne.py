import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set(color_codes=True)


wine_quality_data = np.loadtxt('winequality/winequality-red.csv', skiprows=1, delimiter=";")

# Choose elements
high_quality = np.where(wine_quality_data[:, -1] >= 6)[0]
low_quality = np.where(wine_quality_data[:, -1] < 6)[0]


# PCA
pca = PCA(n_components=2)
pca.fit(wine_quality_data[:, :-1], wine_quality_data[:, -1])
X_embedded = pca.fit_transform(wine_quality_data[:, :-1])

plt.scatter(X_embedded[high_quality, 0], X_embedded[high_quality, 1], alpha=0.2, color="green")
plt.scatter(X_embedded[low_quality, 0], X_embedded[low_quality, 1], alpha=0.2, color="red")
plt.axis('equal')
plt.show()


# TSNE
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(wine_quality_data[:, [3, 10]])

plt.scatter(X_embedded[high_quality, 0], X_embedded[high_quality, 1], alpha=0.2, color="green")
plt.scatter(X_embedded[low_quality, 0], X_embedded[low_quality, 1], alpha=0.2, color="red")
plt.axis('equal')
plt.show()
