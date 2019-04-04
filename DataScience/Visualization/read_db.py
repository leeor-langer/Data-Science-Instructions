import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

sns.set(color_codes=True)

wine_quality_data = np.loadtxt('winequality/winequality-red.csv', skiprows=1, delimiter=";")