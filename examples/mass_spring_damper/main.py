# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:28:00 2020

Author: Mohamed Baioumy
Email: Mohamed@robots.ox.ac.uk
Affiliation: Oxford Robotics Institute

"""


# Standard import Lib
import numpy as np
import matplotlib.pyplot as plt

# Import helper function and parameters
import helpers as hlp
import active_inf_helpers as aic

# For plotting
import seaborn as sns
palette = sns.color_palette()

# =============================================================================
# Setting for changing high-level behaviour
# =============================================================================

MODE = 2
ACTION_TIME = 0
ACTIONS_ALLOWED = True
VAR_UPDATE = 0
ACTION_MODE = 1
VARIANCE_UPDATE_ALLOWED = True


# =============================================================================
# # Real environmental dynamics parameters
# =============================================================================
t_start = 0
t_end = 15
dt = 0.001
t = np.linspace(t_start, t_end, -1 + 1 + int((t_end - t_start)/dt))

# x is positoin, x_p velocity and x_pp acceleration
x = np.zeros(len(t))
x_p = np.zeros(len(t))
x_pp= np.zeros(len(t))

# Initial condition on latent variables
x[0] = -0.5
x_p[0] = -1

# Target variables
x_target = 1
x_p_target = 0

a_max = 50
a_min = -50


# =============================================================================
# Active inference control parameters
# =============================================================================

# Covariances
sigma_mu0_inv = 0.1*np.ones(len(t))
sigma_mu1_inv = 0.5*np.ones(len(t))
sigma_obs0_inv = 1.4*np.ones(len(t))
sigma_obs1_inv = 0.6*np.ones(len(t))


# Learning rates
k_mu = 20
k_action = 500
k_omega = 0.0

# actions
a = np.zeros(len(t))
a[0] = 0

# Beliefs
mu = np.zeros(len(t))
mu_p = np.zeros(len(t))
mu_pp = np.zeros(len(t))

mu[0] = x[0] 
mu_p[0] = x_p[0]  

# Free energy
F = np.zeros(len(t))
F_after = np.zeros(len(t))

# Observations
obs = np.zeros(len(t))
obs_p = np.zeros(len(t))


tau_inv = 1
tau_inv_vec = np.zeros(len(t))
tau_inv_vec[0] = tau_inv
k_tau_inv = 0



# Active inference loop
for i in range(len(x) - 1):

    # Observation
    obs[i] = hlp.add_normal_noise(x[i], 0.0001)
    obs_p[i] = hlp.add_normal_noise(x_p[i], 0.0001)

    # Definition of error terms
    eps_mu = mu_p[i] - aic.f_mu(x_target, mu[i], tau_inv)
    eps_mu_p = mu_pp[i] - aic.par_f_mu(x_target, mu[i], tau_inv) *mu_p[i]
    eps_obs = obs[i] - aic.g_mu(mu[i])
    eps_obs_p = obs_p[i] - aic.par_g_mu(mu[i])*mu_p[i]


    F[i] = 0.5*(-np.log(sigma_mu0_inv[i]*sigma_mu1_inv[i]*sigma_obs0_inv[i]*sigma_obs1_inv[i])
                     + np.float_power(eps_mu, 2)*sigma_mu0_inv[i]
                     + np.float_power(eps_mu_p, 2)*sigma_mu1_inv[i]
                     + np.float_power(eps_obs, 2)*sigma_obs0_inv[i]
                     + np.float_power(eps_obs_p, 2)*sigma_obs1_inv[i]
                     )

    par_f = aic.par_f_mu(x_target, mu[i], tau_inv)
    par_par_f = aic.par_par_f_mu(x_target, mu[i], tau_inv)
    par_g = aic.par_g_mu(mu[i])
    par_par_g = aic.par_par_g_mu(mu[i])
    par_f_mu_p = aic.par_f_mu_p(x_target, mu[i], tau_inv)




    # Belief update
    mu_dot = (mu_p[i]
             + k_mu*eps_obs*sigma_obs0_inv[i] * aic.par_g_mu(mu[i])
             + k_mu*eps_mu*sigma_mu0_inv[i] * aic.par_f_mu(x_target, mu[i], tau_inv)
             )

#    mu_p_dot = (mu_pp[i]
#               + k_mu*eps_obs_p/sigma_obs1_inv[i] * (par_par_g * mu_p[i] + par_g)
#               - k_mu*eps_mu_p/sigma_mu0_inv[i]
#               + k_mu*eps_mu_p/sigma_mu1_inv[i]*(par_par_f * mu_p[i]+ par_f)
#               )

    mu_p_dot = (mu_pp[i]
               + k_mu*eps_obs_p*sigma_obs1_inv[i] * (par_par_g * mu_p[i] + par_g)
               - k_mu*eps_mu_p*sigma_mu0_inv[i]*(par_f_mu_p)
               + k_mu*eps_mu_p*sigma_mu1_inv[i]*(par_par_f * mu_p[i]+ par_f)
               )

    mu_pp_dot = - k_mu * eps_mu_p*sigma_mu1_inv[i]


    mu[i + 1] = mu[i] + dt*mu_dot
    mu_p[i + 1] = mu_p[i] + dt*mu_p_dot
    mu_pp[i + 1] = mu_pp[i] + dt*mu_pp_dot


    if ACTIONS_ALLOWED:
        do_da = aic.do_da(ACTION_MODE,obs[i-1], obs[i], a[i-1], a[i])
        do_p_da = aic.do_da(ACTION_MODE,obs_p[i-1], obs_p[i], a[i-1], a[i])

        a_proposed = (a[i]
                    - dt*k_action
                    * (do_da * eps_obs*sigma_obs0_inv[i]
                    + do_p_da*eps_obs_p*sigma_obs1_inv[i])
                    )

        a[i+1] = max(a_min, min(a_max, a_proposed))


    if VARIANCE_UPDATE_ALLOWED:
        k_omega = 0
    else:
        k_omega = 0

#
    sigma_obs0_inv[i+1] = max (0.01, (sigma_obs0_inv[i]
                        - dt*k_omega*(-1/sigma_obs0_inv[i] + np.float_power(eps_obs, 2))))
    sigma_obs1_inv[i+1] = max (0.01, (sigma_obs1_inv[i]
                        - dt*k_omega*(-1/sigma_obs1_inv[i] + np.float_power(eps_obs_p, 2))))

#    sigma_mu0_inv[i+1] = max (1, (sigma_mu0_inv[i]
#                        - dt*k_omega*(-1/sigma_mu0_inv[i] + np.float_power(eps_mu, 2))))
#    sigma_mu1_inv[i+1] = max (1, (sigma_mu1_inv[i]
#                        - dt*k_omega*(-1/sigma_mu1_inv[i] + np.float_power(eps_mu_p, 2))))
#

    df_dtau_inv = -2*sigma_mu0_inv[i]*(x_target - mu[i]) * (mu_p[i] - tau_inv*(x_target
                             - mu[i])) + 2*sigma_mu1_inv[i]*(mu_pp[i]+tau_inv*mu_p[i])*mu_p[i]


    tau_inv_vec[i+1] = max(0.5, tau_inv_vec[i] - k_tau_inv*df_dtau_inv*dt)
    tau_inv = tau_inv_vec[i+1]

    F_after[i+1] = 0.5*(-np.log(sigma_mu0_inv[i+1]*sigma_mu1_inv[i+1]*sigma_obs0_inv[i+1]*sigma_obs1_inv[i+1])
                     + np.float_power(eps_mu, 2)*sigma_mu0_inv[i+1]
                     + np.float_power(eps_mu_p, 2)*sigma_mu1_inv[i+1]
                     + np.float_power(eps_obs, 2)*sigma_obs0_inv[i+1]
                     + np.float_power(eps_obs_p, 2)*sigma_obs1_inv[i+1]
                     )

    if (F_after[i+1] > F[i]):
        sigma_obs0_inv[i+1] = sigma_obs0_inv[i]
        sigma_obs1_inv[i+1] = sigma_obs1_inv[i]
        sigma_mu0_inv[i+1] = sigma_mu0_inv[i]
        sigma_mu1_inv[i+1] = sigma_mu1_inv[i]


   # if(i*dt) == 0:
   #     precision_update_allowed = False
   #
   # if(i*dt) == 4:
   #     precision_update_allowed = False
   #
   # if(i*dt) == 9:
   #     precision_update_allowed = False



#
#    if (i*dt) == 10:
#        x_target = 0.5
#    if (i*dt) == 15:
#        x_target = 1.2

    # Simulaiton of real dynamics
    x_pp= hlp.sim_dynamics(a[i], x[i], x_p[i])
    x_p[i+1] = hlp.rk4_int(x_p[i], dt, x_pp)
    x[i+1] = hlp.rk4_int(x[i], dt, x_p[i])

# Adjust the last value for Free-energy
F[-1] = F[-2]


plt.figure(0, figsize=[2*6.4, 1.5*4.8])
plt.plot(t, x, label="True position")
plt.plot(t, mu, label=r"Belief $\mu$")
plt.plot(t, x_target*np.ones(len(t)), '--', label="Target")

plt.legend(loc='lower center',
          fancybox=True, shadow=False, ncol=4, fontsize=12)
plt.title(r'Belief ($\mu(t)$) about position (x)', fontsize=18)
plt.ylabel('Position (m)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
