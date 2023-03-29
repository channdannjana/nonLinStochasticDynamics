import matplotlib.pyplot as plt
import numpy as np
import random as ran
from math import*
from scipy.stats import norm
##from scipy.integrate import odeint

##### plot of velocity variance ####################


dt =0.05
T = 8.
n = np.arange(0,T,dt)
beta = 0.5

####################################################

nonG_noise = np.loadtxt('non_G_noise_new1_1E8.dat')

dt =0.05
ensemble = np.arange(0,10000000,1)
no_ensemble = len(ensemble)
T = 5
n = np.arange(0,T,dt)
N = np.zeros((len(n),len(ensemble)))
for i in range(0,len(ensemble)):
    for k in range(0,len(n)-1):
        N[k,i] = ran.choice(nonG_noise)
######## noise two point function
N2 = N[50,:]*N
N2_avg = np.mean(N2,axis=1)

'''
fig1 = plt.figure()
plt.xlabel('time(t)')
#plt.ylabel('<N(t)N(5)>')
plt.title('Noise-Noise two point function')
p1,= plt.plot(n,N2_avg)
#plt.legend([p1,p2],['dt=0.1', 'dt=0.2'],loc='best')
fig1.savefig('Noise_two_pt_fn.jpg')

'''

print(N2_avg[50])

######### Noise three point function

'''
N3 = N*N[50,:]**2
N3_avg = np.mean(N3,axis=1) - 3*N2_avg*np.mean(N,axis=1)


fig2 = plt.figure()
plt.xlabel('time(t)')
plt.ylabel('<N(t)N(5)^2>')
plt.title('Noise three point function')
p2,= plt.plot(n,N3_avg, 'r')
#p4,= plt.plot(n,Npt2four_pt_avg, 'b')
#plt.legend([p3,p4],['dt=0.1', 'dt=0.2'],loc='best')
fig2.savefig('Noise_three_pt_fn.jpg')
'''
######### Noise four point function

N4 = N*N[50,:]**3
N4_avg = np.mean(N4,axis=1) - 3*N2_avg[50]*N2_avg

'''
fig3 = plt.figure()
plt.xlabel('time(t)')
#plt.ylabel('<N(t)N(5)^3>')
plt.title('Noise four point function')
p3,= plt.plot(n,N4_avg, 'r')
#p4,= plt.plot(n,Npt2four_pt_avg, 'b')
#plt.legend([p3,p4],['dt=0.1', 'dt=0.2'],loc='best')
fig3.savefig('Noise_four_pt_fn.jpg')
'''

print(N4_avg[50])

######### Noise six point function

N6 = N*N[50,:]**5
N6_avg = np.mean(N6,axis=1) - 15*N2_avg*N2_avg[50]**2 - 5*N2_avg*N4_avg[50] - 10*N4_avg*N2_avg[50]

print(N6_avg[50]) 
'''
fig4 = plt.figure()
plt.xlabel('time(t)')
#plt.ylabel('<N(t)N(5)^3>')
plt.title('Noise six point function')
p4,= plt.plot(n,N6_avg, 'r')
#p4,= plt.plot(n,Npt2four_pt_avg, 'b')
#plt.legend([p3,p4],['dt=0.1', 'dt=0.2'],loc='best')
fig4.savefig('Noise_six_pt_fn.jpg')


print(N6_avg[50])  

plt.show()
'''


