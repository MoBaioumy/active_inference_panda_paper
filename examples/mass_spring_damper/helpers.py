# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:54:47 2020

Author: Mohamed Baioumy
Email: Mohamed@robots.ox.ac.uk
Affiliation: Oxford Robotics Institute

"""

import numpy as np


# Parameter definition
k1 = 1          # Spring constant
k2 = 0.1        # Damper contant
m = 1           # Mass


def sim_dynamics(a_t, x, v):
     return (a_t - k1*x - k2*v)/m

def rk4_int(prior, time_step, derivative):
    """
        Implements RK4 integration for one time-step
    """
    h = time_step
    f_dot = derivative
    f_t_min = prior

    k1 = h*f_dot
    k2 = h*(f_dot + k1/2)
    k3 = h*(f_dot + k2/2)
    k4 = h*(f_dot + k3)

    f_t = f_t_min + (k1 + 2*k2 + 2*k3 +k4)/6
    return f_t

def add_normal_noise(state, variance):
    # Add gaussian noise to scalars 
    #TODO: adjust for vectors
    
    return(state + np.random.normal(0, variance, 1))
