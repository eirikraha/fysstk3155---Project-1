import numpy as np
import matplotlib.pyplot as plt

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


n_samples = 100

x = np.random.rand(n_samples, 1)
y = np.random.rand(n_samples, 1)

z = FrankeFunction(x, y)
z_noise = z + 0.01*np.random.rand(n_samples, 1)

# Centering x and y. Why do we need to do this?
x_ = x - np.mean(x)
y_ = y - np.mean(y)
z_ = z - np.mean(z)

#Creating the vector
#np.c_ tells each 
X = np.c_[np.ones((n_samples, 1)), x, x**2] #Should I add x**3 .. x**5, y .. y**5, xy, xy**2 .. xy**4 etc?
X_ = np.c_[x_, x_**2]   

print(X)
print(X_)
