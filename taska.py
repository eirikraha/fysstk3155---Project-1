import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression

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

def HomeMadeRidge(X, zr, sk = False):
	lmb_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
	num_values = len(lmb_values)
	beta_ridge = np.zeros((X.shape[1], num_values))

	Id_mat = np.eye(X.shape[1])

	for i in range(0, num_values):
		beta_ridge[:, i] = (np.linalg.inv(X.T @ X + lmb_values[i]*Id_mat) @ X.T @ zr).flatten()

	pred_ridge = X @ beta_ridge   #Why not RSS like in the book at equation 3.43?

	if sk == True:
		pred_ridgeSK = np.zeros((X.shape[0], num_values))
		for i in range(0, num_values):
			pred_ridgeSK[:, i] = (Ridge(alpha=lmb_values[i], fit_intercept = False).fit(X, zr).predict(X)).flatten()
		
		return pred_ridge, pred_ridgeSK, lmb_values
	else:
		return pred_ridge, lmb_values


def bootstrap(xdata, ydata, nBoots = 1000):
	# bootstrap as found in slide 92 in the lecture notes on Regression
    bootVec = np.zeros(nBoots)
    bootR2 = np.zeros(nBoots)
    bootMSE = np.zeros(nBoots)

    # Choose random elements from the data-array, one element may be
    # chosen more than once. 
    for k in range(0,nBoots):
    	print(xdata, len(xdata))
    	temp_x = np.random.choice(xdata, len(xdata))
    	temp_y = np.random.choice(ydata, len(ydata))

    	temp_x_mesh, temp_y_mesh = np.meshgrid(temp_x, temp_y)
    	temp_z_mesh = FrankeFunction(temp_x_mesh, temp_y_mesh)

    	boot_x = np.ravel(temp_x_mesh)
    	boot_y = np.ravel(temp_y_mesh)
    	boot_z = np.ravel(temp_z_mesh)

    	boot_X = np.c_[np.ones((n_samples**2,1)), boot_x, boot_x**2, boot_x**3, boot_x**4, boot_x**5,
    							  boot_y, boot_y**2, boot_y**3, boot_y**4, boot_y**5,
    							  boot_x*boot_y, boot_x*boot_y**2, boot_x*boot_y**3, boot_x*boot_y**4,
    							  boot_y*boot_x**2, boot_y*boot_x**3, boot_y*boot_x**4,
    							  boot_y**2*boot_x**2, boot_y**2*boot_x**3, boot_y**3*boot_x**2]

		# OLS 
    	boot_beta_ls = np.linalg.inv( boot_X.T @ boot_X ) @ boot_X.T @ boot_z
    	boot_pred_ls = boot_X @ boot_beta_ls 

        #bootVec[k] = np.average(np.random.choice(data, len(data)))
    	bootVec[k] = np.average(boot_pred_ls)
    	bootR2[k]  = R2score(boot_z, boot_pred_ls)
    	bootMSE[k] = MeanSquareError(boot_z, boot_pred_ls)
    
    bootAvg = np.average(bootVec)
    bootVar = np.var(bootVec)
    bootStd = np.std(bootVec)

    bootR2avg  = np.average(bootR2)
    bootMSEavg = np.average(bootMSE)

    return [bootVec, bootAvg, bootVar, bootStd, bootR2avg, bootMSEavg]


fig = plt.figure()
ax = fig.gca(projection='3d')


n_samples = 100

x_start = np.random.rand(n_samples, 1)
y_start = np.random.rand(n_samples, 1)
x_mesh, y_mesh = np.meshgrid(x_start,y_start)

z_mesh = FrankeFunction(x_mesh, y_mesh)

x = np.ravel(x_mesh)
y = np.ravel(y_mesh)

print (x.shape, y.shape)

zr = np.ravel(z_mesh)
z_noise = zr + 0.01*np.random.rand(n_samples, 1)

# Centering x and y. Why do we need to do this?
# This is added to give a common frame of reference, not needed
# in this project, but might be used later
#x_ = x - np.mean(x)
#y_ = y - np.mean(y)
#z_ = z - np.mean(z)

#Creating the vector
#np.c_ sets each element as a column
X = np.c_[np.ones((n_samples**2,1)), x, x**2, x**3, x**4, x**5,
								  y, y**2, y**3, y**4, y**5,
								  x*y, x*y**2, x*y**3, x*y**4,
								  y*x**2, y*x**3, y*x**4,
								  y**2*x**2, y**2*x**3, y**3*x**2]

# OLS 
beta_ls = np.linalg.inv( X.T @ X ) @ X.T @ zr
pred_ls = X @ beta_ls     # How do I know which of all the models in X gives me this value?
pred_lsSK = (LinearRegression(fit_intercept = False).fit(X, zr).predict(X)).flatten()
pred_ridge, pred_ridgeSK,lmb_values = HomeMadeRidge(X, zr, sk = True)
pred_lasso = (Lasso(alpha = 0.01, fit_intercept = False).fit(X, zr).predict(X)).flatten()  #Which alpha should I use and what does it do?


sigma_sq = variance(zr, pred_ls, X.shape[1] - 1)   #Here I assume X has dim (N * p + 1), and hence take -1 from X.shape[1], but is this correct?
var_beta = np.linalg.inv( X.T @ X) * sigma_sq
MSE_ls = MeanSquareError(zr, pred_ls)
MSEsci_ls = mean_squared_error(zr, pred_ls)
R2_ls = R2score(zr, pred_ls)
R2sci_ls = r2_score(zr, pred_ls)

#print ("Var(beta): \n", var_beta)

# print ("OLS:")
# print ("MSE: \n", MSE_ls)
# print ("SciKit MSE: \n", MSEsci_ls)
# print ("R2score: \n", R2_ls)
# print ("SciKit R2score: \n", R2sci_ls)

# for i in range(0, pred_ridge.shape[1]):
# 	MSE_ri = MeanSquareError(zr, pred_ridge[:, i])
# 	MSEsci_ri = mean_squared_error(zr, pred_ridge[:, i])
# 	R2_ri = R2score(zr, pred_ridge[:, i])
# 	R2sci_ri = r2_score(zr, pred_ridge[:, i])

# 	print ("Ridge with lambda = %1.2e:" %lmb_values[i])
# 	print ("MSE: \n", MSE_ri)
# 	print ("SciKit MSE: \n", MSEsci_ri)
# 	print ("R2score: \n", R2_ri)
# 	print ("SciKit R2score: \n", R2sci_ri)
# 	print (" ")

MSE_lasso = MeanSquareError(zr, pred_lasso)
MSEsci_lasso = mean_squared_error(zr, pred_lasso)
R2_lasso = R2score(zr, pred_lasso)
R2sci_lasso = r2_score(zr, pred_lasso)

print ("Lasso:")
print ("MSE: \n", MSE_lasso)
print ("SciKit MSE: \n", MSEsci_lasso)
print ("R2score: \n", R2_lasso)
print ("SciKit R2score: \n", R2sci_lasso)

#bootResults = bootstrap(x_start, y_start, nBoots = 100)

#print (bootResults)

# print (np.max(pred_ls - pred_lsSK))
# print (np.max(pred_ridge - pred_ridgeSK))