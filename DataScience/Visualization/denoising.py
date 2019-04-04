import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

sns.set(color_codes=True)

wine_quality_data = np.loadtxt('winequality/winequality-red.csv', skiprows=1, delimiter=";")

# Choose elements
high_quality = np.where(wine_quality_data[:, -1] >= 6)[0]
low_quality = np.where(wine_quality_data[:, -1] < 6)[0]

# PCA
pca = PCA(n_components=3)
pca.fit(wine_quality_data[:, :-1])
print(pca.explained_variance_ratio_)
print('-----------------------------')
print(abs( pca.components_ ))
X_embedded = pca.fit_transform(wine_quality_data[:, :-1])

fig = pyplot.figure()
ax = Axes3D(fig)
plt.scatter(X_embedded[high_quality, 0], X_embedded[high_quality, 1], X_embedded[high_quality, 2], alpha=0.2, color="green")
plt.scatter(X_embedded[low_quality, 0], X_embedded[low_quality, 1], X_embedded[high_quality, 2], alpha=0.2, color="red")
plt.show()

# Largest principal component
sns.distplot(wine_quality_data[high_quality, 6], bins=12, kde=False, color="green")
sns.distplot(wine_quality_data[low_quality, 6], bins=12, kde=False, color="red")
plt.show()