import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot
sns.set(color_codes=True)

wine_quality_data = np.loadtxt('winequality/winequality-white.csv', skiprows=1, delimiter=";")

# Choose elements
high_quality = np.where(wine_quality_data[:, -1] >= 7)[0]
low_quality = np.where(wine_quality_data[:, -1] < 6)[0]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=100, random_state=5)

X = wine_quality_data[:, :-1]
y = wine_quality_data[:, -1]
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Most important feature
mean00 = np.median(wine_quality_data[high_quality, indices[0]])
mean01 = np.median(wine_quality_data[high_quality, indices[1]])
mean10 = np.median(wine_quality_data[low_quality, indices[0]])
mean11 = np.median(wine_quality_data[low_quality, indices[1]])
fig = pyplot.figure()
plt.scatter(wine_quality_data[high_quality, indices[0]],
            wine_quality_data[high_quality, indices[1]],
            alpha=0.2, color="green")
plt.scatter(wine_quality_data[low_quality, indices[0]],
            wine_quality_data[low_quality, indices[1]],
            alpha=0.2, color="red")
plt.scatter(mean00, mean01, marker='X', s=40, color="black")
plt.scatter(mean10, mean11, marker='X', s=40, color="black")
plt.show()

