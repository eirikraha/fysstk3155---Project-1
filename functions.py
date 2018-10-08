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
from classes import *

def FrankeFunction(x,y):
	#Given from the project1.pdf
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def variance(y, y_hat, p):
	# Based upon the calculation of sigma^2 on page
	# 47 in HTF

	return 1.0/(len(y) - p - 1) * np.sum((y - y_hat)**2)

def MeanSquareError(y, y_hat):
	# Mean Square Error as defined by project 1

	return 1.0/len(y) * np.sum((y - y_hat)**2)

def mean_val(y):
	# Mean value as defined by project 1
	return 1.0/len(y) * np.sum(y)

def R2score(y, y_hat):
	# R^2 score as defined by project 1
	return 1 - np.sum((y - y_hat)**2)/np.sum((y - np.mean(y))**2)

def R2MSEeval(zr, pred, sk = False, method = " ", printer = False):

	print ("Help!")

	zr = zr.flatten()

	MSE = MeanSquareError(zr, pred)
	R2 = R2score(zr, pred)
	MSEsci = 0
	R2sci = 0
	
	if printer and not sk:
		print (method)
		print ("MSE: \n", MSE)
		print ("R2score: \n", R2)

		f = open(filename, 'a+')
		f.write('%1.2f, %1.2f \n' %(MSE, R2))
		f.close()

		return MSE, R2

	elif sk and not printer:
		MSEsci = mean_squared_error(zr, pred)
		R2sci = r2_score(zr, pred)	

		return MSEsci, R2sci

	elif printer and sk:
		MSEsci = mean_squared_error(zr, pred)
		R2sci = r2_score(zr, pred)	

		print ("\n", method)
		print ("MSE: \n", MSE)
		print ("SciKit MSE: \n", MSEsci)
		print ("R2score: \n", R2)
		print ("SciKit R2score: \n", R2sci)

		f = open(filename, 'a+')
		f.write('%1.2f, %1.2f, %1.2f, %1.2f \n' %(MSE, MSEsci, R2, R2sci))
		f.close()

		return MSE, MSEsci, R2, R2sci

	else:
		return MSE, R2


def bootstrap(X, zdata, method, lmb = 1e-3, alpha = 1e-3, nBoots = 1000, 
				sk = True, test_size = 0.40, output = False, printer = True):

	bootVec = np.zeros(int(np.floor(test_size*len(zdata))))
	#bootVec = np.array([np.zeros(len(zdata)+ 1) for i in range(0, nBoots)])

	bootIndexes = np.linspace(0, len(zdata) - 1, len(zdata), dtype = int)

	MSE = np.zeros(nBoots)
	MSE_SK = np.zeros(nBoots)
	R2 = np.zeros(nBoots)
	R2_SK = np.zeros(nBoots)

	progress_tracker = np.floor(nBoots/100.0)

	# Choose random elements from the data-array, one element may be
	# chosen more than once. 
	#bootVec = np.random.choice(bootIndexes, int(np.floor(test_size*len(bootIndexes))), replace = False)

	X_train, X_test, zr_train, zr_test = train_test_split(X, zdata, test_size=test_size)
	
	print (X.shape, zdata.shape)

	# X_train    = X[bootVec]
	# zr_train   = zdata[bootVec]

	# test_index = [j for j in bootIndexes if j not in bootVec]

	# print ("Test Index finder done")

	# X_test = X[test_index]
	# zr_test = (zdata[test_index]).flatten()

	pred_z = np.empty((zr_test.shape[0], nBoots))


	for k in range(0,nBoots):

		if k == progress_tracker:
			print ("Progress = %d%%" %np.floor(k/(nBoots/100.0)), end = '\r')
			progress_tracker += np.floor(nBoots/100.0)

		X_, zr_ = resample(X_train, zr_train)

		if method == "OLS" and not sk:
			pred_z[:, k] = HomeMadeOLS().fit(X_, zr_).predict(X_test).pred.flatten()
			MSE[k], R2[k] = R2MSEeval(zr_test,  pred_z[:, k])

		elif method == "OLS" and sk:
			pred_z[:, k] = (LinearRegression(fit_intercept = False).fit(X_, zr_).predict(X_test)).flatten()
			MSE_SK[k], R2_SK[k] = R2MSEeval(zr_test.flatten(), pred_z[:, k])

		elif method == "Ridge" and not sk:
			pred_z[:, k] = HomeMadeRidge().fit(X_, zr_, lmb = lmb).predict(X_test).pred.flatten()
			MSE[k], R2[k] = R2MSEeval(zr_test, pred_z[:, k])

		elif method == "Ridge" and sk:
			pred_z[:, k] = (Ridge(alpha = lmb, fit_intercept = False).fit(X_, zr_).predict(X_test)).flatten()
			MSE_SK[k], R2_SK[k] = R2MSEeval(zr_test, pred_z[:, k])

		elif method == "Lasso" and not sk:
			print ("Sorry, but you have to use scikit's Lasso")
			sk = True

		elif method == "Lasso" and sk:
			pred_z[:, k] = (Lasso(alpha = alpha, fit_intercept = False).fit(X_, zr_).predict(X_test)).flatten()
			MSE_SK[k], R2_SK[k] = R2MSEeval(zr_test, pred_z[:, k])
		
		else:
			print ("Please use either OLS, Ridge or Lasso as your method")


	bootMSE_avg = np.average(MSE)
	bootR2_avg = np.average(R2)
	bootMSE_SK_avg = np.average(MSE_SK)
	bootR2_SK_avg = np.average(R2_SK)

	bootMSE_var = np.var(MSE)
	bootR2_var = np.var(R2)
	bootMSE_SK_var = np.var(MSE_SK)
	bootR2_SK_var = np.var(R2_SK)

	bootMSE_std = np.std(MSE)
	bootR2_std = np.std(R2)
	bootMSE_SK_std = np.std(MSE_SK)
	bootR2_SK_std = np.std(R2_SK)

	zr_test = zr_test.reshape(-1, 1)

	error = np.mean( np.mean((zr_test - pred_z)**2, axis=1, keepdims=True))
	bias = np.mean((zr_test - np.mean(pred_z, axis=1, keepdims=True))**2)
	variance = np.mean( np.var(pred_z, axis=1, keepdims=True))

	if printer:
		print('Error:', error)
		print('Bias^2:', bias)
		print('Var:', variance)
		print('{} >= {} + {} = {}'.format(error, bias, variance, (bias+variance)))

		if not sk:
			print ('Diff: ', bootMSE_avg - (bias+variance))
			print ('Bootstrap MSE: ', bootMSE_avg)
			print ('Bootstrap MSE std: ', bootMSE_std)
			print ('Bootstrap MSE var: ', bootMSE_var)
			print ('Bootstrap R2:', bootR2_avg)
			print ('Bootstrap R2 std: ', bootR2_std)
			print ('Bootstrap R2 var: ', bootR2_var)


		if sk:
			print ('Diff: ', bootMSE_SK_avg - (bias+variance))
			print ('Bootstrap MSE: ', bootMSE_SK_avg, 'std: ', bootMSE_SK_std, 'var: ', bootMSE_SK_var)
			print ('Bootstrap R2:  ', bootR2_SK_avg, 'std: ', bootR2_SK_std, 'var: ', bootR2_SK_var)


	if output and not sk:
		return bootMSE_avg, bootR2_avg

	if output and sk:
		return bootMSE_SK_avg, bootR2_SK_avg


def X_creator(x, y, n_samples1 = 100, n_samples2 = 100, k=6):

	X = np.c_[np.ones((n_samples1*n_samples2, 1))]

	for i in range(1, k):
		for j in range(0, i+1):
			X = np.c_[X, x**(i - j)*y**(j)]
#			print ("x**", (i-j), "*y**", (j))

	return X

if __name__ == '__main__':
	exit('Please, run main.py')