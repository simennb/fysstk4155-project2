import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
import sgd
import time


def test_SGD():
    np.random.seed(100)
    N_epochs = 500
    m = 1000
    x = 2*np.random.rand(m, 1)
    y = 4+3*x+np.random.randn(m, 1)

    X = np.c_[np.ones((m, 1)), x]
    beta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print("Linear inversion")
    print(beta_linreg)
    sgdreg = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1)#, tol=None)#, learning_rate='constant')
#    sgdreg = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1, tol=None)
#    sgdreg = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1, tol=None, learning_rate='constant')
    ts_skl = time.time()
    sgdreg.fit(x, y.ravel())
    te_skl = time.time()
    print("sgdreg from scikit")
    print(sgdreg.intercept_, sgdreg.coef_)

    sgdreg_own = sgd.LinRegSGD(N_epochs, m, eta0=0.1, learning_rate='')
    sgdreg_own.set_step_length(0.1, 10.0)
    sgdreg_own.fit(X, y.ravel())  # First time its run will also compile
    ts_own = time.time()
    sgdreg_own.fit(X, y.ravel())
    te_own = time.time()
    print("LinRegSGD")
    print(sgdreg_own.beta)

    t_skl = te_skl - ts_skl
    t_own = te_own - ts_own
    print('Time SKL: %.3e s' % t_skl)
    print('Time own: %.3e s' % t_own)
    print('Factor own/skl = %.3e' % (t_own/t_skl))


'''
m = 100
x = 2*np.random.rand(m,1)
y = 4+3*x+np.random.randn(m,1)

X = np.c_[np.ones((m,1)), x]
theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)

theta = np.random.randn(2,1)
eta = 0.1
Niterations = 1000


for iter in range(Niterations):
    gradients = 2.0/m*X.T @ ((X @ theta)-y)
    theta -= eta*gradients
print("theta from own gd")
print(theta)

xnew = np.array([[0],[2]])
Xnew = np.c_[np.ones((2,1)), xnew]
ypredict = Xnew.dot(theta)
ypredict2 = Xnew.dot(theta_linreg)


n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ ((xi @ theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
print("theta from own sdg")
print(theta)

plt.plot(xnew, ypredict, "r-")
plt.plot(xnew, ypredict2, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()
'''

if __name__ == '__main__':
    test_SGD()
