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
	# Ordinaty least squared regression. Fit fit's the data
	# and predict predicts
	# ConfIntBeta finds the confidence intervar of the beta values
	# and plotter plots them

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

		self.var_beta = [(np.linalg.inv(X.T @ X))[i, i] for i in range(0, p)]

		self.conf_intervals = [[float(self.beta[i]) - 2*np.sqrt(self.var_beta[i]), 
							float(self.beta[i]) + 2*np.sqrt(self.var_beta[i])] for i in range(0, len(self.var_beta))]

		return self

	def plotter(self, fs1 = 20, fs2 = 20, fs3 = 20, task = 'a', method = 'OLS', patch = 'None'):

		twostd_beta = 2*np.sqrt(self.var_beta)

		plt.errorbar(np.linspace(0, len(self.beta), len(self.beta)), self.beta, yerr=twostd_beta)
		plt.title(r'$\beta$ confidence for %s' %method, fontsize = fs1)
		plt.xlabel(r'$\beta_j$', fontsize = fs2)
		plt.savefig('../figures/%s-%s-contour-beta-patch%s.png' %(task, method, patch))
		plt.tight_layout()


class HomeMadeRidge():
	# Ridge regression. Fit fit's the data
	# and predict predicts
	# ConfIntBeta finds the confidence intervar of the beta values
	# and plotter plots them

	def __init__(self):

		self.beta = 0
		self.pred = 0

	def fit(self, X, zr, lmb = 1e-3):
		self.Id_mat = np.eye(X.shape[1])
		self.beta = (np.linalg.inv(X.T @ X + lmb*self.Id_mat) @ X.T @ zr).flatten()

		return self

	def predict(self, X):
		self.pred = X @ self.beta

		return self

	def ConfIntBeta(self, X, zr, pred, lmb = 1e-3):
		N = X.shape[0]
		p = X.shape[1]

		variance = 1./(N - p - 1) * np.sum((zr - pred)**2)

		self.var_beta = [(variance*(np.linalg.inv(X.T @ X * lmb * self.Id_mat)) @ X.T @ X @((np.linalg.inv(X.T @ X * lmb * self.Id_mat)).T))[i, i] for i in range(0, p)]

		self.conf_intervals = [[float(self.beta[i]) - 2*np.sqrt(self.var_beta[i]), 
							float(self.beta[i]) + 2*np.sqrt(self.var_beta[i])] for i in range(0, len(self.var_beta))]

		return self

	def plotter(self, fs1 = 20, fs2 = 20, fs3 = 20, task = 'a', method = 'OLS', patch = 'None'):

		twostd_beta = 2*np.sqrt(self.var_beta)

		plt.errorbar(np.linspace(0, len(self.beta), len(self.beta)), self.beta, yerr=twostd_beta)
		plt.title(r'$\beta$ confidence for %s' %method, fontsize = fs1)
		plt.xlabel(r'$\beta_j$', fontsize = fs2)
		plt.savefig('../figures/%s-%s-contour-beta-patch%s.png' %(task, method, patch))
		plt.tight_layout()


class MapDataImport():
	#imports and maps data
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