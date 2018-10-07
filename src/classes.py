import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sys import stdout, argv, exit
from imageio import imread



class HomeMadeOLS():
	def __init__(self):

		self.beta = 0
		self.pred = 0

	def fit(self, X, zr):
		self.beta = np.linalg.inv( X.T @ X ) @ X.T @ zr

		return self

	def predict(self, X):
		self.pred = X @ self.beta

		return self

	def ConfIntBeta(self, X, zr, pred):
		N = X.shape[0]
		p = X.shape[1]


		variance = 1./(N - p - 1) * np.sum((zr - pred)**2)

		var_beta = [(np.linalg.inv(X.T @ X)*variance)[i, i] for i in range(0, p)]

		self.conf_intervals = [[float(self.beta[i]) - 2*np.sqrt(var_beta[i]), 
							float(self.beta[i]) + 2*np.sqrt(var_beta[i])] for i in range(0, len(var_beta))]

class HomeMadeRidge():
	def __init__(self):

		self.beta = 0
		self.pred = 0

	def fit(self, X, zr, lmb = 1e-3):
		Id_mat = np.eye(X.shape[1])
		self.beta = (np.linalg.inv(X.T @ X + lmb*Id_mat) @ X.T @ zr).flatten()

		return self

	def predict(self, X):
		self.pred = X @ self.beta

		return self

class MapDataImport():
	def __init__(self):
		b = 0

	def ImportData(self, filename = '../data/SRTM_data_Norway_1.tif'):
		self.terrain = imread(filename)

	def PlotTerrain(self):
		plt.figure()
		plt.title('Terrain over Norway 1')
		plt.imshow(self.terrain, cmap='gray')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()


if __name__ == '__main__':
	exit('Please, run main.py')