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
k_omega = 0.1

# actions
a = np.zeros(len(t))
a[0] = 0

# Beliefs
mu = np.zeros(len(t))
mu_p = np.zeros(len(t))
mu_pp = np.zeros(len(t))

mu[0] = x[0] + 0.5
mu_p[0] = x_p[0] - 0.5

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
