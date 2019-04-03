import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
sns.set(color_codes=True)

wine_quality_data = np.loadtxt('winequality/winequality-red.csv', skiprows=1, delimiter=";")

# Choose elements
high_quality = np.where(wine_quality_data[:, -1] >= 6)[0]
low_quality = np.where(wine_quality_data[:, -1] < 6)[0]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=100, random_state=5)

X = wine_quality_data[:, :-1]
y = wine_quality_data[:, -1]
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Most important feature
sns.distplot(wine_quality_data[high_quality, indices[0]], bins=12, kde=False, color="green")
sns.distplot(wine_quality_data[low_quality, indices[0]], bins=12, kde=False, color="red")
plt.show()