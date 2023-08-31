import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson

#defining variables:
#F:  max. number of photons we want to represent
# g is the coupling constant
#t1 and t2 are the timelenghts of the JC and aJC pulses correspondingly
#JC[n] and aJC[n] are the  angle that the pulses are rotating in our 'n' state
#'times' is the number of times we want to make our pulses perform (analogous to the discrete time variable in the Quantum Walk formalism)

f = 200
n = list(range(-1,f-1))
g = 2.0
print(n)
omega_n = [np.sqrt((i+1))*g for i in n]
omega_n.insert(0,0)
print(omega_n)

t_1 = 1
t_2 = 1

JC = [j*t_1/2 for j in omega_n]
aJC = [k*t_2/2 for k in omega_n]
print(aJC)

times = 2000

#defining g and e, with dimension n x times. The row j is the statevector after applying j times our operator
#g for the ground component, and e for the excited component
g = [[0]*len(n) for i in range(times)]
e = [[0]*len(n) for i in range(times)]


#Inserting now the initial conditions: (Remember normalizing)

g[0][2] = np.sqrt(1/2)
e[0][2] = np.sqrt(1/2)



#Defining the probability matrix P 
#We insert in the firs row the initial conditions
P = [[0]*len(n) for i in range(times)]
mean_V = [[0]*len(n) for i in range(times)]
mean_V_sq = [[0]*len(n) for i in range(times)]


for i in range(len(n)):
    P[0][i] = (g[0][i])**2 + (e[0][i])**2


#we simulate now the time evolution after applying the JC/aJC pulses the afore selected amount of times .
#note that the equations we are using are extracted from the JC model

for k in range(0,times-1):
    for l in range(2,f-2):
         e[k+1][l] = np.sin(aJC[l-1])*(-e[k][l-2]*np.sin(JC[l-2])+g[k][l-1]*np.cos(JC[l-2]))+ np.cos(aJC[l-1])*(e[k][l]*np.cos(JC[l])+g[k][l+1]*np.sin(JC[l]))
         g[k+1][l] = -np.sin(aJC[l])*(e[k][l+1]*np.cos(JC[l+1])+g[k][l+2]*np.sin(JC[l+1]))+ np.cos(aJC[l])*(-e[k][l-1]*np.sin(JC[l-1])+g[k][l]*np.cos(JC[l-1]))
         P[k+1][l] = (g[k+1][l])**2 + (e[k+1][l])**2
         mean_V[k][l-2] = P[k][l]*(l)
         mean_V_sq[k][l-2] = P[k][l]*((l)**2)


#computing important parameters in order to plot their time evolutions:
mean = np.zeros(times)
mean_sq = np.zeros(times)
Q = np.zeros(times)
Varianza = np.zeros(times)

sum_e = np.zeros(times)

for j in range(times):
    sum_e[j] = np.sum(np.square(e[j]))


for k in range(times-1):
    mean[k]= np.sum(mean_V[k])
    mean_sq[k] = np.sum(mean_V_sq[k])


for k in range(times-1):
    Varianza[k] = abs(mean_sq[k]- mean[k]**2)


for k in range(len(Varianza)-1):
    Q[k] = (Varianza[k]-mean[k])/mean[k]




#plots:
#Time evolution of the variance
plt.plot(range(times),Varianza)
plt.xlim(0, times-2)
plt.show()
#Time evolution of the mean of the distribution
plt.plot(range(times),mean)
plt.xlim(0, times-2)
plt.show()
#Time evolution of the Mandel parameter 'Q'
plt.plot(range(times),Q)
plt.xlim(0, times-2)
plt.show()


#Distribution of probabilities against number of photons for a certain time value
plt.plot(n, P[100])
plt.xlim(0, f-2)
plt.ylim(0, 1)
plt.show()


#3dplot (k= time, n = photon number, P = probability)

vector_P = np.ravel(P)
n_rep = np.tile(n, times)
row = np.repeat(np.arange(times), f)

print(len(row))
print(len(n_rep))
print(len(vector_P))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = vector_P**(1/2) 

sc = ax.scatter(row, n_rep, vector_P, c=colors, cmap='coolwarm',s = 8)
sc.set_clim(vmin=0, vmax=0.25)

ax.set_xlabel('k')
ax.set_ylabel('n')
ax.set_zlabel('P')
cbar = plt.colorbar(sc)

plt.show()

