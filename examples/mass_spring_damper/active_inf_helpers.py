# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:15:03 2020

Author: Mohamed Baioumy
Email: Mohamed@robots.ox.ac.uk
Affiliation: Oxford Robotics Institute

"""



# ToDO: add PDF with general derivation of control and state-estimation
# ToDO: Move the code with Beta to a differnt repo
# ToDO: Move do/da code


def g_mu(mu):
    return mu

def par_g_mu(mu):
    return 1

def par_par_g_mu(mu):
    return 0

def f_mu(target, mu, tau_inv=1, beta=0, mu_p=0):
    if tau_inv > 0:
        return tau_inv*(target - mu) + beta*(-mu_p)
    else:
#        print("tau_inv has to be a positive real number")
        return tau_inv*(target - mu) + beta*(-mu_p)

def par_f_mu(target, mu, tau_inv=1, beta=0):
    if tau_inv > 0:
        return (-tau_inv)
    else:
#        print("tau_inv has to be a positive real number")
        return (-tau_inv)

def par_f_mu_p(target, mu, tau_inv=1, beta=0):
    if tau_inv > 0:
        return (-beta)
    else:
#        print("tau_inv has to be a positive real number")
        return 0

def par_par_f_mu(target, mu, tau_inv=1, beta=0):
    if tau_inv > 0:
        return 0
    else:
#        print("tau_inv has to be a positive real number")
        return 0

def do_da(mode, obs1=0, obs2=0, a1=0, a2=0):
    if mode == 1:
        return 1
    if mode == 2:
        return((obs2 - obs1)/(a2 - a1))

def do_p_da(mode, obs_p1=0, obs_p2=0, a1=0, a2=0):
    if mode == 1:
        return 1
    if mode == 2:
        return((obs_p2 - obs_p1)/(a2 - a1))
