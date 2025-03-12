import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that is sin graph:
def T_out(t): 
    return 9.06632 * np.sin( np.pi / 12 * (t - 8.6225) ) + 92.04167

# function that returns dz/dt
def model(T_in,t,constants):
    dTindt = constants * (T_out(t) - T_in)
    return dTindt

# initial condition
T_in = 0

# number of time points
n = 401

# time points
t = np.linspace(0,24,n)

# Initial values
initial_value = 80

# Times:
t_data = np.linspace(start=0, stop=24, num=400)

# solve ODE
z = odeint(model,initial_value, t_data)

# plot results
plt.plot(t_data,z,'g:',label='t_in')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()