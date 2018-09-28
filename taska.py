import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sys import stdout


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

class HomeMadeRidge():
	def __init__(self):

		self.beta = 0
		self.pred = 0

	def fit(self, X, zr, lmb = 1e-3):
		Id_mat = np.eye(X.shape[1])
		self.beta = (np.linalg.inv(X.T @ X + lmb*Id_mat) @ X.T @ zr).flatten()

	def predict(self, X):
		self.pred = X @ self.beta

		return self

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
	return 1 - np.sum((y - y_hat)**2)/np.sum((y - mean_val(y))**2)


def HomeMadeRidge(X, zr, lmb_values):

	Id_mat = np.eye(X.shape[1])
	beta_ridge[:, i] = (np.linalg.inv(X.T @ X + lmb_values*Id_mat) @ X.T @ zr).flatten()

	pred_ridge = X @ beta_ridge   #Why not RSS like in the book at equation 3.43?

	return pred_ridge, lmb_values


def R2MSEeval(zr, pred, method = "", printer = False, sk = False):
	MSE = MeanSquareError(zr, pred)
	R2 = R2score(zr, pred)
	
	if printer:
		print (Method)
		print ("MSE: \n", MSE)
		print ("R2score: \n", R2)

		return MSE, R2

	elif sk and not printer:
		MSEsci = mean_squared_error(zr, pred)
		R2sci = r2_score(zr, pred)	

		return MSE, MSEsci, R2, R2sci

	elif printer and sk:
		print (Method)
		print ("MSE: \n", MSE)
		print ("SciKit MSE: \n", MSEsci)
		print ("R2score: \n", R2)
		print ("SciKit R2score: \n", R2sci)

		return MSE, MSEsci, R2, R2sci

	else:
		return MSE, R2


def bootstrap(X, zdata, method, lmb = 1e-3, alpha = 1e-3, nBoots = 100, sk = True):
    bootVec = np.zeros(int(np.floor(0.6*len(zdata))))
    #bootVec = np.array([np.zeros(len(zdata)+ 1) for i in range(0, nBoots)])

    bootIndexes = np.linspace(0, len(zdata) - 1, len(zdata), dtype = int)

    MSE = np.zeros(nBoots)
    MSE_SK = np.zeros(nBoots)
    R2 = np.zeros(nBoots)
    R2_SK = np.zeros(nBoots)

    progress_tracker = np.floor(nBoots/100.0)

    for k in range(0,nBoots):

    	if k == progress_tracker:
    		print ("Progress = %d%%" %np.floor(k/(nBoots/100.0)), end = '\r')
    		progress_tracker += np.floor(nBoots/100.0)

    	# Choose random elements from the data-array, one element may be
    	# chosen more than once. 
    	bootVec = np.random.choice(bootIndexes, int(np.floor(0.6*len(bootIndexes))))
    	X_train    = X[bootVec]
    	zr_train   = zr[bootVec]

    	test_index = [j for j in bootIndexes if j not in bootVec]

    	X_test = X[test_index]
    	zr_test = zr[test_index]

    	if method == "OLS" and not sk:
    		pred_OLS = HomeMadeOLS().fit(X_train, zr_train).predict(X_test).pred
    		MSE[k], R2[k] = R2MSEeval(zr_test, pred_OLS)

    	elif method == "OLS" and sk:
    		pred_OLS_SK = (LinearRegression(fit_intercept = False).fit(X_train, zr_train).predict(X_test)).flatten()
    		MSE_SK[k], R2_SK[k] = R2MSEeval(zr_test, pred_OLS_SK)

    	elif method == "Ridge" and not sk:
    		pred_ridge = HomeMadeOLS().fit(X_train, zr_train, lmb).predict(X_test).pred
    		MSE[k], R2[k] = R2MSEeval(pred_ridge, zr_test)

    	elif method == "Ridge" and sk:
    		pred_ridge_SK = (Ridge(alpha = alpha, fit_intercept = False).fit(X_train, zr_train).predict(X_test)).flatten()
    		MSE_SK[k], R2_SK[k] = R2MSEeval(pred_ridge_SK, zr_test)

    	elif method == "Lasso" and not sk:
    		print ("Sorry, but you have to use scikit's Lasso")
    		sk = True

    	elif method == "Lasso" and sk:
    		pred_lasso_SK = (Lasso(alpha = alpha, fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()
    		MSE_SK[k], R2_SK[k] = R2MSEeval(pred_lasso_SK, zr_test)
    	
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


    if not sk:
    	print (bootMSE_avg, bootMSE_std, bootMSE_var, bootR2_avg, bootR2_std, bootR2_var)
    if sk:
    	print (bootMSE_SK_avg, bootMSE_SK_std, bootMSE_SK_var, bootR2_SK_avg, bootR2_SK_std, bootR2_SK_var)
    # bootAvg = np.average(bootVec)
    # bootVar = np.var(bootVec)
    # bootStd = np.std(bootVec)

    # return [bootVec, bootAvg, bootVar, bootStd]

def X_creator(x, y, k=6):

	X = np.c_[np.ones((n_samples**2, 1))]

	for i in range(1, k):
		for j in range(0, i+1):
			X = np.c_[X, x**(i - j)*y**(j)]
#			print ("x**", (i-j), "*y**", (j))

	return X



n_samples = 100

x_start = np.random.rand(n_samples, 1)
y_start = np.random.rand(n_samples, 1)
x_mesh, y_mesh = np.meshgrid(x_start,y_start)

z_mesh = FrankeFunction(x_mesh, y_mesh)

x = np.ravel(x_mesh)
y = np.ravel(y_mesh)

print (x.shape, y.shape)

zr = z_mesh.reshape(-1, 1)

# Centering x and y. Why do we need to do this?
# This is added to give a common frame of reference, not needed
# in this project, but might be used later
#x_ = x - np.mean(x)
#y_ = y - np.mean(y)
#z_ = z - np.mean(z)

#Creating the vector
#np.c_ sets each element as a column
X = X_creator(x, y)
# X = np.c_[np.ones((n_samples**2,1)), x, x**2, x**3, x**4, x**5,
# 								  y, y**2, y**3, y**4, y**5,
# 								  x*y, x*y**2, x*y**3, x*y**4,
# 								  y*x**2, y*x**3, y*x**4,
# 								  y**2*x**2, y**2*x**3, y**3*x**2]

# Finding the predicted values using different methods
beta_ls = np.linalg.inv( X.T @ X ) @ X.T @ zr
pred_ls = X @ beta_ls     # How do I know which of all the models in X gives me this value?
pred_lsSK = (LinearRegression(fit_intercept = False).fit(X, zr).predict(X)).flatten()
# pred_ridge, pred_ridgeSK,lmb_values = HomeMadeRidge(X, zr)
pred_lasso = (Lasso(alpha = 0.01, fit_intercept = False).fit(X, zr).predict(X)).flatten()  #Which alpha should I use and what does it do?

#Data with noise
#zr_noise = np.array([(0.01*np.random.uniform(low=-0.9999,high = 1) + zr[i]) for i in range(0, len(zr))])
zr_noise = zr + 0.01*np.random.randn(n_samples**2, 1)   #Gives shape (10000, 10000), but shouldn't we want only (10000,)?

print (pred_lasso)

# print (np.mean(zr), np.mean(zr_noise))

# Finding the predicted values using different methods based on noisy data
beta_ls_noise = np.linalg.inv( X.T @ X ) @ X.T @ zr_noise
pred_ls_noise = X @ beta_ls_noise     # How do I know which of all the models in X gives me this value?
pred_lsSK_noise = (LinearRegression(fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()
# pred_ridge_noise, lmb_values_noise = HomeMadeRidge(X, zr_noise)
pred_lasso_noise = (Lasso(alpha = 0.01, fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()  #Which alpha should I use and what does it do?

print (pred_lasso_noise)

#bootstrap(X, zr, method = "OLS")

# #Bootstrap resampling
# bootResults = bootstrap(zr)
# zr_bs = bootResults[0]

# # Finding the predicted values using different methods based on bootstrap data
# beta_ls_bs = np.linalg.inv( X.T @ X ) @ X.T @ zr_bs
# pred_ls_bs = X @ beta_ls_bs     # How do I know which of all the models in X gives me this value?
# pred_lsSK_bs = (LinearRegression(fit_intercept = False).fit(X, zr_bs).predict(X)).flatten()
# pred_ridge_bs, pred_ridgeSK_bs,lmb_values_bs = HomeMadeRidge(X, zr_bs, sk = True)
# pred_lasso_bs = (Lasso(alpha = 0.01, fit_intercept = False).fit(X, zr_bs).predict(X)).flatten()  #Which alpha should I use and what does it do?


# #Getting real terrain data


print (np.average(pred_ls - pred_lsSK))





sigma_sq = variance(zr, pred_ls, X.shape[1] - 1)   #Here I assume X has dim (N * p + 1), and hence take -1 from X.shape[1], but is this correct?
var_beta = np.linalg.inv( X.T @ X) * sigma_sq
MSE_ls = MeanSquareError(zr, pred_ls)
MSEsci_ls = mean_squared_error(zr, pred_ls)
R2_ls = R2score(zr, pred_ls)
R2sci_ls = r2_score(zr, pred_ls)

#print ("Var(beta): \n", var_beta)

print ("OLS:")
print ("MSE: \n", MSE_ls)
print ("SciKit MSE: \n", MSEsci_ls)
print ("R2score: \n", R2_ls)
print ("SciKit R2score: \n", R2sci_ls)

# # for i in range(0, pred_ridge.shape[1]):
# # 	MSE_ri = MeanSquareError(zr, pred_ridge[:, i])
# # 	MSEsci_ri = mean_squared_error(zr, pred_ridge[:, i])
# # 	R2_ri = R2score(zr, pred_ridge[:, i])
# # 	R2sci_ri = r2_score(zr, pred_ridge[:, i])

# # 	print ("Ridge with lambda = %1.2e:" %lmb_values[i])
# # 	print ("MSE: \n", MSE_ri)
# # 	print ("SciKit MSE: \n", MSEsci_ri)
# # 	print ("R2score: \n", R2_ri)
# # 	print ("SciKit R2score: \n", R2sci_ri)
# # 	print (" ")

# MSE_lasso = MeanSquareError(zr, pred_lasso)
# MSEsci_lasso = mean_squared_error(zr, pred_lasso)
# R2_lasso = R2score(zr, pred_lasso)
# R2sci_lasso = r2_score(zr, pred_lasso)

# print ("Lasso:")
# print ("MSE: \n", MSE_lasso)
# print ("SciKit MSE: \n", MSEsci_lasso)
# print ("R2score: \n", R2_lasso)
# print ("SciKit R2score: \n", R2sci_lasso)

#bootResults = bootstrap(x_start, y_start, nBoots = 100)

#print (bootResults)

# print (np.max(pred_ls - pred_lsSK))
# print (np.max(pred_ridge - pred_ridgeSK))