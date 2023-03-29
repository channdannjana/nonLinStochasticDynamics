import matplotlib.pyplot as plt
import numpy as np
import random as ran
from math import*
from scipy.stats import norm
##from scipy.integrate import odeint

#np.savetxt('non_G_noise.dat',eta)

nonG_noise = np.loadtxt('non_G_noise_new1_1E8.dat')

beta = 0.5

dt =0.05

#### Bare parameters ##########

fsqB = 5.1105 # bare strength of Gaussian noise
zeta_NB = -0.0857 # bare non-gaussian strength
alpha = 2.0   # deviation from non-linear FDR
#### Renormalised parameters ##########

fsq = fsqB - zeta_NB/(2*fsqB*dt) + (zeta_NB**2)/(2*(fsqB**3)*dt**2) # renormalised strength of Gaussian noise
gma = (beta/2)*fsq # renormalised damping constant
zeta_N = zeta_NB - (7*zeta_NB**2)/(2*dt*fsqB**2) + (119./12.)*(zeta_NB**3)/((fsqB**4)*dt**2) # renormalised non-gaussianity
zeta_gma = -alpha*(beta/12)*zeta_N ## renormalised thermal jitter

zeta_gmaB = zeta_gma/(1- 3*zeta_NB/(2*dt*fsqB**2) + (31./12.)*(zeta_NB**2)/((fsqB**4)*dt**2) ) # bare thermal jitter
gmaB = gma - zeta_gmaB/(fsqB*dt) + zeta_gmaB*zeta_NB/(2*(fsqB**3)*dt**2) ## bare damping constant


##############

ensemble = np.arange(0,10000000,1)
no_ensemble = len(ensemble)

T = 8.
n = np.arange(0,T,dt)
N = np.zeros(len(ensemble))
NL = np.zeros(len(ensemble))

q = np.zeros((len(n),len(ensemble)))
qdot = np.zeros((len(n),len(ensemble)))


#qL = np.zeros((len(n),len(ensemble)))
#qdotL = np.zeros((len(n),len(ensemble)))


q0 = 0.
qdot0 = 0.

for i in range(1,len(n)-1):
    for k in range(0,len(ensemble)):
        N[k] = ran.choice(nonG_noise)
        #NL[k] = ran.normalvariate(0,1)
    q[i+1,:] = q[i,:] + dt*qdot[i,:]
    qdot[i+1,:] = qdot[i,:] - dt*gmaB*qdot[i,:] - dt*zeta_gmaB*(N**2)*qdot[i-1,:] + dt*fsqB*N
    #qL[i+1,:] = qL[i,:] + dt*qdotL[i,:]
    #qdotL[i+1,:] = qdotL[i,:] - dt*gma*qdotL[i,:] + np.sqrt(fsq*dt)*NL

########### q v correlator #############

#q_avg = np.mean(q,axis=1)
#qdot_avg = np.mean(qdot,axis=1)


'''
q_avgL = np.mean(qL,axis=1)
qdot_avgL = np.mean(qdotL,axis=1)
'''

'''
qqdot = q*qdot
qqdot_avg = np.mean(qqdot,axis=1) #- q_avg*qdot_avg
'''


############ velocity two point function ###########


qdot2 = (qdot)**2
qdot2_avg = np.mean(qdot2,axis=1) # - (qdot_avg)**2

np.savetxt('v2_alpha2_3.dat',qdot2_avg)

#qdot2L = (qdotL)**2
#qdot2_avgL = np.mean(qdot2L,axis=1)

#np.savetxt('v2_var.dat',qdot2_avg)
#np.savetxt('velocity_var_LBare.dat',qdot2_avgL)


############ qdot^4  ######################


qdot4 = (qdot)**4
qdot4_avg = np.mean(qdot4,axis=1) #- 3*qdot2_avg**2 #- qdot_avg**4

np.savetxt('v4_alpha2_3.dat',qdot4_avg)

#qdot4 = (qdot)**4
#qdot4_avg = np.mean(qdot4,axis=1) - 3*qdot2_avg**2 #- qdot_avg**4

'''
frthroot_qdot4 = (np.abs(qdot4_avg))**(1/4.)
plt.plot(n,frthroot_qdot4)
'''

'''
frthroot_qdot4 = np.zeros(len(n))

for i in range(0,len(n)-1):
    if qdot4_avg[i] < 0:
        frthroot_qdot4[i] = -(np.abs(qdot4_avg[i]))**(1/4.)
    else:
        frthroot_qdot4[i] = (qdot4_avg[i])**(1/4.)

p4,=plt.plot(n,frthroot_qdot4)
#np.savetxt('frth_root_v4_alpha1.dat',frthroot_qdot4)
'''

