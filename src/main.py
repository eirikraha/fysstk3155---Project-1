import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as mtick
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sys import stdout, argv, exit
from imageio import imread
import time
from classes import *
from functions import *

################################
#####     Initializing     #####
################################

if len(argv) == 1:
	print ("Please write a, b, c, d or e after the filename in terminal")
	print ("You can also write plot after the filename if you wish.")
	print ("Plotting only applies for task b and c")
	exit()
elif len(argv) == 2:
	if argv[1] == "b" or argv[1] == "c":
		print ("If you want to plot, please write plot after task %s" %argv[1])

	argv.append("noplot")

fs1 = 22
fs2 = 18
fs3 = 12
show = True


n_samples = 100

x_start = np.random.rand(n_samples, 1)
y_start = np.random.rand(n_samples, 1)
x_mesh, y_mesh = np.meshgrid(x_start,y_start)


z_mesh = FrankeFunction(x_mesh, y_mesh)

x = np.ravel(x_mesh)
y = np.ravel(y_mesh)

zr = z_mesh.reshape(-1, 1)
zr_noise = zr + 0.03*np.random.randn(n_samples**2, 1)

# Centering x and y. Why do we need to do this?
# This is added to give a common frame of reference, not needed
# in this project, but might be used later
#x_ = x - np.mean(x)
#y_ = y - np.mean(y)
#z_ = z - np.mean(z)

#Creating the vector
#np.c_ sets each element as a column
X = X_creator(x, y)


###########################
#####     Task a)     #####
###########################

if argv[1] == "a" or argv == "all":
	#Without noise
	OLS = HomeMadeOLS().fit(X, zr).predict(X)
	
	pred_LS = (OLS.pred).flatten()
	beta_conf = (OLS.ConfIntBeta(X, zr, pred_LS)).conf_intervals
	OLS.plotter(task = 'a', method = 'OLS')


	pred_LS_SK = (LinearRegression(fit_intercept = False).fit(X, zr).predict(X)).flatten()

	#With noise
	pred_LS_noise = (HomeMadeOLS().fit(X, zr_noise).predict(X).pred).flatten()
	pred_LS_noise_SK = (LinearRegression(fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()

	filename = '../benchmarks/taska.txt'
	f = open(filename, 'w')
	f.write('# Initial line for OLS on FrankeFunction \n')
	f.close()



	#Finding MSE and R2
	R2MSEeval(zr, pred_LS, filename, method = "OLS", printer = True, sk = True)   #Should we compare with zr_noise?"
	R2MSEeval(zr, pred_LS_SK, filename, method = "OLS from SK", printer = True, sk = True)
	R2MSEeval(zr, pred_LS_noise, filename, method = "OLS with noise", printer = True, sk = True)
	R2MSEeval(zr, pred_LS_noise_SK, filename, method = "OLS from SK with noise", printer = True, sk = True)


	print ("Bootstrap using OLS without noise:")
	bootstrap(X, zr, filename = filename, method = "OLS")
	print ("Bootstrap using OLS with noise:")
	bootstrap(X, zr_noise, filename = filename, method = "OLS")

###########################
#####     Task b)     #####
###########################

elif argv[1] == "b" or argv == "all":
	
	if argv[2] == "plot":
		#Finding optimal lambda value for the instance without noise
		lmb_values = np.logspace(-5, -1, 100)
		R2list = []
		MSElist = []

		for lmb in lmb_values:
			MSE, R2 = bootstrap(X, zr, lmb = lmb, method = "Ridge", sk = True, 
								output = True, printer = False)
			R2list.append(R2)
			MSElist.append(MSE)


		fig, ax = plt.subplots()

		ax.semilogx(lmb_values, R2list, label = "R2")
		ax.set_title(r"R2 as a function of $\lambda$", fontsize = fs1)
		ax.set_xlabel(r"$\lambda$", fontsize = fs2)
		ax.set_ylabel(r"R2", fontsize = fs2)
		ax.tick_params(labelsize = fs3)
		ax.legend()

		plt.tight_layout()
		fig.savefig('../figures/RidgeR2.png')

		fig2, ax2 = plt.subplots()
		ax2.semilogx(lmb_values, MSElist, label = "MSE") 
		ax2.set_title(r"MSE as a function of $\lambda$", fontsize = fs1)
		ax2.set_xlabel(r"$\lambda$", fontsize = fs2)
		ax2.set_ylabel(r"MSE", fontsize = fs2)
		ax2.tick_params(labelsize = fs3)
		#ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1e'))
		ax2.legend()

		plt.tight_layout()
		fig2.savefig('../figures/RidgeMSE.png')
		
		if show:
			plt.show()
			
	if len(argv) > 1 and argv[2] == "analyze":
		if show:
			FitAnalyze(x, y, zr, method = "Ridge", show = True)
			NoiseAnalyze(X, zr, method = "Ridge", show = True)
		else:
			FitAnalyze(x, y, zr, method = "Ridge")
			NoiseAnalyze(X, zr, method = "Ridge")
	
	lmb = 1e-4

	#Without noise
	#pred_ridge = (HomeMadeRidge().fit(X, zr, lmb = lmb).predict(X).pred).flatten()

	ridge = HomeMadeRidge().fit(X, zr, lmb = lmb).predict(X)
	
	pred_ridge = (ridge.pred).flatten()
	beta_conf = (ridge.ConfIntBeta(X, zr, pred_ridge, lmb = lmb)).conf_intervals
	ridge.plotter(task = 'b', method = 'Ridge')

	filename = "../benchmarks/taskb_lambda%1.2e.txt" %(lmb)
	f = open(filename, 'w')
	f.write("#Task b")
	f.close()


	pred_ridge_SK = (Ridge(alpha = lmb, fit_intercept = False).fit(X, zr).predict(X)).flatten()

	#With noise
	#OBS! Remeber to vary noise
	pred_ridge_noise = (HomeMadeRidge().fit(X, zr_noise, lmb = lmb).predict(X).pred).flatten()
	pred_ridge_noise_SK = (Ridge(alpha = lmb, fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()

	#Finding MSE and R2
	R2MSEeval(zr, pred_ridge, filename, method = "Ridge", printer = True, sk = True)
	R2MSEeval(zr, pred_ridge_SK, filename, method = "Ridge from SK", printer = True, sk = True)
	R2MSEeval(zr, pred_ridge_noise, filename, method = "Ridge with noise", printer = True, sk = True)
	R2MSEeval(zr, pred_ridge_noise_SK, filename, method = "Ridge from SK with noise", printer = True, sk = True)


	print ("Bootstrap using Ridge without noise:")
	bootstrap(X, zr, filename = filename, lmb = lmb, method = "Ridge")
	print ("Bootstrap using Ridge with noise:")
	bootstrap(X, zr_noise, filename = filename, lmb = lmb, method = "Ridge")

	if argv[2] == "noise":
		noise = np.linspace(0, 10, 100)
		for i in noise:
			zr_noise = zr + i*np.random.randn(n_samples**2, 1)
			pred_ridge_noise_SK = (Ridge(fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()

			MSE_noise, MSE_SK_noise, R2_noise, R2_SK_noise = R2MSEeval(zr, pred_ridge_noise_SK, filename2, method = ("Ridge from SK with noise %1.2f" %i), printer = True, sk = True)


###########################
#####     Task c)     #####
###########################

elif argv[1] == "c" or argv == "all":

	#Plots if neccessary
	if len(argv) > 1 and argv[2] == "plot":
		#Finding optimal alpha value for the instance without noise
		alpha_values = np.logspace(-5, -1, 30)   #Should we use logspace?
		R2list = []
		MSElist = []

		for alpha in alpha_values:
			MSE, R2 = bootstrap(X, zr, alpha = alpha, method = "Lasso", sk = True, 
								output = True, printer = False, nBoots = 100)
			R2list.append(R2)
			MSElist.append(MSE)


		fig, ax = plt.subplots()

		ax.semilogx(alpha_values, R2list, label = "R2")
		ax.set_title(r"R2 as a function of $\alpha$", fontsize = fs1)
		ax.set_xlabel(r"$\alpha$", fontsize = fs2)
		ax.set_ylabel(r"R2", fontsize = fs2)
		ax.legend()
		ax.tick_params(labelsize = fs3)
		plt.tight_layout()

		fig.savefig('../figures/LassoR2')

		fig2, ax2 = plt.subplots()
		ax2.semilogx(alpha_values, MSElist, label = "MSE") 
		ax2.set_title(r"MSE as a function of $\alpha$", fontsize = fs1)
		ax2.set_xlabel(r"$\alpha$", fontsize = fs2)
		ax2.set_ylabel(r"MSE", fontsize = fs2)
		ax2.legend()
		ax2.tick_params(labelsize = fs3)
		plt.tight_layout()

		fig2.savefig('../figures/LassoMSE')

		if show:
			plt.show()
			
	if len(argv) > 1 and argv[2] == "analyze":
		if show:
			FitAnalyze(x, y, zr, method = "Lasso", show = True)
			NoiseAnalyze(X, zr, method = "Lasso", show = True)
		else:
			FitAnalyze(x, y, zr, method = "Lasso")
			NoiseAnalyze(X, zr, method = "Lasso")

		#Found the best aplha to be below 1e-4, but those values gives me errors in the scikit lasso.
		# Did therefore choose a more stable alpha.

	alpha = 1e-4

	filename = "../benchmarks/taskc_alpha%1.2e.txt" %(alpha)

	f = open(filename, 'w')
	f.write("#Task c")
	f.close()

	#Without noise
	pred_lasso_SK = (Lasso(alpha = alpha, fit_intercept = False).fit(X, zr).predict(X)).flatten()

	#With noise
	pred_lasso_noise_SK = (Lasso(alpha = alpha, fit_intercept = False).fit(X, zr_noise).predict(X)).flatten()

	#Finding MSE and R2
	R2MSEeval(zr, pred_lasso_SK, filename, method = "Lasso", printer = True, sk = True)
	R2MSEeval(zr, pred_lasso_noise_SK, filename, method = "Lasso with noise", printer = True, sk = True)


	print ("Bootstrap using Lasso without noise:")
	bootstrap(X, zr, filename = filename, alpha = alpha, method = "Lasso")
	print ("Bootstrap using Lasso with noise:")
	bootstrap(X, zr_noise, filename = filename, alpha = alpha, method = "Lasso")

###########################
#####     Task d)     #####
###########################

elif argv[1] == "d" or argv == "all":

	terrainReader = MapDataImport()
	terrainReader.ImportData()
	terrainReader.PlotTerrain()

	print ("Please, come back later")
	exit()
#Import data


###########################
#####     Task e)     #####
###########################

elif argv[1] == "e" or argv == "all":

	terrainReader = MapDataImport()
	terrainReader.ImportData()

	n_patches = 20

	#z = terrainReader.terrain.reshape(-1, 1)

	map_terrain = terrainReader.terrain
	map_terrain = map_terrain/(float(np.max(map_terrain)))


	[n, m] = terrainReader.terrain.shape

	patch_n = int(np.floor(n/n_patches))
	patch_m = int(np.floor(m/n_patches))

	terrain = []

	for i in range(0, n_patches):
		terrain.append((map_terrain)[i*patch_n:(i+1)*patch_n, i*patch_m:(i+1)*patch_m])
		#Will loose the last row by doing it like this

	rows = np.linspace(0, 1, patch_n)
	cols = np.linspace(0, 1, patch_m)

	[C, R] = np.meshgrid(cols, rows)

	x = C.reshape(-1, 1)
	y = R.reshape(-1, 1)

	num_data = n*m

	filename = "../benchmarks/taske.txt"
	f = open(filename, 'w')
	f.write("#Task e")
	f.close()

	n_samples = patch_n
	X = X_creator(x, y, n_samples1 = patch_n, n_samples2 = patch_m)

	timetracker = [0, 0, 0]


	#This whole if statement activates all of the analysis tools used for b and c
	#It would be wise to skip this part when it's not needed
	if len(argv) > 1 and argv[2] == "analyze":
		zr = terrain[0].flatten()

		#Finding optimal alpha value for the instance without noise
		alpha_values = np.logspace(-5, -1, 10) 
		R2list = []
		MSElist = []

		for alpha in alpha_values:
			MSE, R2 = bootstrap(X, zr, alpha = alpha, method = "Lasso", sk = True, 
								output = True, printer = False, nBoots = 100)
			R2list.append(R2)
			MSElist.append(MSE)


		fig, ax = plt.subplots()

		ax.semilogx(alpha_values, R2list, label = "R2")
		ax.set_title(r"R2 as a function of $\alpha$", fontsize = fs1)
		ax.set_xlabel(r"$\alpha$", fontsize = fs2)
		ax.set_ylabel(r"R2", fontsize = fs2)
		ax.legend()
		ax.tick_params(labelsize = fs3)
		plt.tight_layout()

		fig.savefig('../figures/Taske_LassoR2')

		fig2, ax2 = plt.subplots()
		ax2.semilogx(alpha_values, MSElist, label = "MSE") 
		ax2.set_title(r"MSE as a function of $\alpha$", fontsize = fs1)
		ax2.set_xlabel(r"$\alpha$", fontsize = fs2)
		ax2.set_ylabel(r"MSE", fontsize = fs2)
		ax2.legend()
		ax2.tick_params(labelsize = fs3)
		plt.tight_layout()

		fig2.savefig('../figures/Taske_LassoMSE')

		if show:
			plt.show()
			
		if show:
			FitAnalyze(x, y, zr, method = "Lasso", show = True, taske = True, 
						n_samples1 = patch_n, n_samples2 = patch_m)
			#NoiseAnalyze(X, zr, method = "Lasso", show = True, taske = True)
		else:
			FitAnalyze(x, y, zr, method = "Lasso", taske = True, 
						n_samples1 = patch_n, n_samples2 = patch_m)
			#NoiseAnalyze(X, zr, method = "Lasso", taske = True)


		#Finding optimal lambda value for the instance without noise
		lmb_values = np.logspace(-5, -1, 100)
		R2list = []
		MSElist = []

		for lmb in lmb_values:
			MSE, R2 = bootstrap(X, zr, lmb = lmb, method = "Ridge", sk = True, 
								output = True, printer = False)
			R2list.append(R2)
			MSElist.append(MSE)


		fig, ax = plt.subplots()

		ax.semilogx(lmb_values, R2list, label = "R2")
		ax.set_title(r"R2 as a function of $\lambda$", fontsize = fs1)
		ax.set_xlabel(r"$\lambda$", fontsize = fs2)
		ax.set_ylabel(r"R2", fontsize = fs2)
		ax.tick_params(labelsize = fs3)
		ax.legend()

		plt.tight_layout()
		fig.savefig('../figures/Taske_RidgeR2.png')

		fig2, ax2 = plt.subplots()
		ax2.semilogx(lmb_values, MSElist, label = "MSE") 
		ax2.set_title(r"MSE as a function of $\lambda$", fontsize = fs1)
		ax2.set_xlabel(r"$\lambda$", fontsize = fs2)
		ax2.set_ylabel(r"MSE", fontsize = fs2)
		ax2.tick_params(labelsize = fs3)
		#ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1e'))
		ax2.legend()

		plt.tight_layout()
		fig2.savefig('../figures/Taske_RidgeMSE.png')
		
		if show:
			plt.show()
			
		if show:
			FitAnalyze(x, y, zr, method = "Ridge", show = True, taske = True, 
						n_samples1 = patch_n, n_samples2 = patch_m)
			#NoiseAnalyze(X, zr, method = "Ridge", show = True, taske = True)
		else:
			FitAnalyze(x, y, zr, method = "Ridge", taske = True, 
						n_samples1 = patch_n, n_samples2 = patch_m)
			#NoiseAnalyze(X, zr, method = "Ridge", taske = True)



	#Bootstrap done on each patch with each method.
	
	for i in range(0, len(terrain)):
		z = terrain[i].flatten()

		f = open(filename, 'a+')
		f.write('\n # Patch number (%d/%d) \n' %(i, len(terrain)))
		f.close()

			#Beta confidence interval
		OLS = HomeMadeOLS().fit(X, z).predict(X)

		pred_LS = (OLS.pred).flatten()
		beta_conf_LS = (OLS.ConfIntBeta(X, z, pred_LS)).conf_intervals
		OLS.plotter(task = 'e', method = 'OLS', patch = '%d' %i)

		ridge = HomeMadeRidge().fit(X, z, lmb = 1e-4).predict(X)

		pred_ridge = (ridge.pred).flatten()
		beta_conf_R = (ridge.ConfIntBeta(X, z, pred_ridge, lmb = 1e-4)).conf_intervals
		ridge.plotter(task = 'e', method = 'ridge', patch = '%d' %i)

		olsstart = time.time()
		bootstrap(X, z, filename = filename, method = "OLS")
		olsend = time.time()
		f = open(filename, 'a+')
		f.write('# Time used = %1.2f \n' %(olsend - olsstart))
		f.close()
		timetracker[0] += (olsend - olsstart)

		ridgestart = time.time()
		bootstrap(X, z, filename = filename, lmb = 1e-4, method = "Ridge")	
		ridgeend = time.time()
		f = open(filename, 'a+')
		f.write('# Time used = %1.2f \n' %(ridgeend - ridgestart))
		f.close()
		timetracker[1] += (ridgeend - ridgestart)

		lassostart = time.time()
		bootstrap(X, z, filename = filename, alpha = 1e-4, method = "Lasso")
		lassoend = time.time()
		f = open(filename, 'a+')
		f.write('# Time used = %1.2f \n' %(lassoend - lassostart))
		f.close()
		timetracker[2] += (lassoend - lassostart)

		

		# pred = (LinearRegression(fit_intercept = False).fit(X, z).predict(X)).flatten()

		# print (pred.shape)

	f = open(filename, 'a+')
	f.write('# Time used for each method: \n')
	f.write('OLS: %1.2f' %timetracker[0])
	f.write('Ridge: %1.2f' %timetracker[1])
	f.write('Lasso: %1.2f' %timetracker[2])
	f.close()

	# plt.figure()
	# plt.title('Terrain over Norway 1')
	# plt.imshow(pred.reshape(-1, 1), cmap='gray')
	# plt.xlabel('X')
	# plt.ylabel('Y')
	# plt.show()



# Perform a), b) and c) on data from d).


else:
	print ("Please write a, b, c, d or e after filename in terminal")
	exit()
