import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error

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

def bootstrap(y, y_hat, nBoots = 1000):
	# bootstrap as found in slide 92 in the lecture notes on Regression

    bootVec = np.zeros(nBoots)

    # Choose random elements from the data-array, one element may be
    # chosen more than once. 
    for k in range(0,nBoots):
        bootVec[k] = np.average(np.random.choice(y_hat, len(y_hat)))
    bootAvg = np.average(bootVec)
    bootVar = np.var(bootVec)
    bootStd = np.std(bootVec)

    bootR2  = R2score(y, bootVec)
    bootMSE = MeanSquareError(y, bootVec)

    return bootVec, bootAvg bootVar, bootStd, bootR2, bootMSE


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

sigma_sq = variance(zr, pred_ls, X.shape[1] - 1)   #Here I assume X has dim (N * p + 1), and hence take -1 from X.shape[1], but is this correct?
var_beta = np.linalg.inv( X.T @ X) * sigma_sq
MSE = MeanSquareError(zr, pred_ls)
MSEsci = mean_squared_error(zr, pred_ls)
R2 = R2score(zr, pred_ls)
R2sci = r2_score(zr, pred_ls)

print ("Var(beta): \n", var_beta)
print ("MSE: \n", MSE)
print ("SciKit MSE: \n", MSE)
print ("R2score: \n", R2)
print ("SciKit R2score: \n", R2sci)

