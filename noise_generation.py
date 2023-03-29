import matplotlib.pyplot as plt
import numpy as np
import random as ran
from math import*
from scipy.stats import norm
##from scipy.integrate import odeint


N = 100000000

#beta = 0.5

dt =0.05

e = 1.09
M = 1.1

 
###parameters##########

fsqB = 5.184 ## bare parameter
zeta_NB = -0.0908 ## bare parameter 

## the noise is generated with the above values of fsqB and zeta_NB. 

#fsq = fsqB - zeta_NB/(fsqB*dt) + (5*zeta_NB**2)/(6*(fsqB**3)*dt**2) ### fsqB is the bare f^2

c = 0.00001

eta_unnorm = lambda x: np.exp(-dt*(fsqB/2)*x**2 - dt*(zeta_NB/factorial(4))*x**4-dt*(c/factorial(6))*x**6)

g = lambda x: norm.pdf(x, 0, e/sqrt(fsqB*dt))

# Make the plots
fig, ax = plt.subplots(1, 1)

# The x coordinates
x = np.linspace(-15.,15.0,100000)
print(len(x))
#normalise eta distribution

area_eta = np.zeros(len(x))

for i in range(len(x)-1):
    area_eta[i+1] = area_eta[i] + eta_unnorm(x[i])*(x[i+1]-x[i])

eta1 = lambda x: (1/area_eta[99999])*np.exp(-dt*(fsqB/2)*x**2 - dt*(zeta_NB/factorial(4))*x**4-dt*c*x**6)

#def eta1(x):
##################    return (1/area_eta[len(x)-1])*np.exp(-(a/2)*x**2 - b*x**4)

x_samples = M * np.random.normal(0, 1/sqrt(fsqB*dt), (N,))
u = np.random.uniform(0, 1, (N, ))


#eta_plot = [(x_samples[i], u[i] * M * g(x_samples[i])) for i in range(N) if u[i] < eta1(x_samples[i]) / (M * g(x_samples[i]))] #to plot the distribution within the function
eta = [(x_samples[i]) for i in range(N) if u[i] < eta1(x_samples[i]) / (M * g(x_samples[i]))] #to store the non-gausian data

np.savetxt('non_G_noise_new1_1E8.dat',eta)

#print((1/N)*sum(eta))
#print(ran.choice(eta))
'''
#print(eta)
###### The samples found by rejection sampling


plt.title('Probability distribution')
#plt.xlabel('eta')
ax.plot([sample[0] for sample in eta_plot], [sample[1] for sample in eta_plot], 'g.', label='Samples')

#ax.plot([sample[0] for sample in eta], 'g.', label='Samples')

# The target probability function

p1,=ax.plot(x, eta1(x), 'r-', label='$eta$')

# The approximated probability density function

p2,=ax.plot(x, M * g(x), 'b-', label='$M \cdot g(x)$')

plt.legend([p1,p2],['target distribution', 'reference distribution'],loc='best')



plt.show()

'''




