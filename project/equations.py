"""
Notation:
C -> calcium concentration
P -> IP3
V -> membrane potential
bar ->  maximum, limiting, or steady-state value
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# playing around with the model

# === Hill Function ===
def hill (X, K, n):
        
    X = np.maximum(X, 0.0)  # force non-negative
    result = X**n / (X**n + K**n)
    
    return result


# === DEFINITION OF PARAMETES ===
# Cell geometry
R_cell     = 8         # um;           T-cell radius
f_R        = 0.25      # unitless;     Nuclear radius fraction of R_cell
f_V        = 0.01      # unitless;     ER volume fraction
f_A        = 30        # unitless;     ER surface, fold of spherical
C_m        = 28        # fF/um^2;      Membrane capacitance

# Ions and potential
T          = 310       # K;            Temperature
V0         = -60       # mV;           Resting membrane potential
V_ER_0     = -60       # mV;           Resting ER potential
C0         = 0.1       # uM;           Resting calcium
C_ext      = 2000      # uM;           Extracellular calcium
dV_C       = 78        # mV;           Reversal potential shift
dV_C_ER    = 63        # mV;           ER-reversal potential shift

# Calcium buffer
b0         = 100       # µM
Kb         = 0.1       # µM
b_ER_0     = 30000     # µM
K_ER_b     = 100       # µM  

# Second messengers
# P0         = 8.7       # nM;           Resting IP3
# beta_P     = 0.6       # nM/s;         IP3 production rate
# gamma_P    = 0.01149   # /s;           IP3 degradation rate
# C_P        = 0.5       # uM;           Calcium of half IP3 production
# n_P        = 1         # unitless;     IP3 production Hill-coefficient
P0         = 0.0087    # µM   (was 8.7 nM → ÷1000)
beta_P     = 0.0006    # µM/s (was 0.6 nM/s → ÷1000)
gamma_P    = 0.01149   # /s
C_P        = 0.5       # µM
n_P        = 1

# Transmembrane protein densities
rho_IP3R   = 11.35     # /um^2;         ER-IP3R density                 (fixed by steady-state)
rho_SERCA  = 700       # /um^2;         ER-SERCA density                (Variable)
rho_PMCA   = 68.57     # /um^2;         PMCA density                    (Fixed by steady-state)
rho_CRAC_0 = 0.6       # /um^2;         Resting active CRAC density     (Variable)
rho_CRAC_p = 3.9       # /um^2;         Max active CRAC density         (Variable)
rho_CRAC_n = 0.5115    # /um^2;         Min active CRAC density         (fixed by steady-state)

# Other constants
R          = 8.314     # J/K*mol:       Rydberg Constant
F          = 96485     # C/mol;         Faraday constant
zCa        = 2         # unitless;      Valence of Ca ions


# Cell geometry
R_nucleus = f_R * R_cell                                         # Radius of nucleus

A_cell  = 4 * np.pi * R_cell**2                                  # Surface area of cell   (19)
Vol_cyt = (4/3) * np.pi * R_cell**3 * (1-f_V-f_R**3)             # Volume of cytosol      (20)
tilde_Vol_ER  = (4/3) * np.pi * f_V * R_cell**3                  # Volume of ER           (21)
A_ER    = 4 * np.pi * f_A * ((3*tilde_Vol_ER)/(4*np.pi))**(2/3)  # Surface area of ER     (22)

xi     = A_cell / Vol_cyt                                        # SA:V ratio of cell               (16)
xi_ERC = A_ER / Vol_cyt                                          # SA:V ratio of ER to cytosol      (17)
xi_ER  = A_ER / tilde_Vol_ER                                     # SA:V ratio of ER                 (18)


# Reverse Potentials
def bar_V_C(C):
    result = ((R*T)/(zCa*F))*np.log(C_ext/C)-dV_C/1000        # Reverse potential wrt C (9)
    return result                                       

def bar_V_C_ER(C):
    result = ((R*T)/(zCa*F))*np.log(C_ext/C)-dV_C_ER/1000        # Reverse potential wrt ER (9)
    return result

# === CRAC Dynamics ===
# CRAC constants
bar_g_CRAC = 0.002    # pS
tau_CRAC   = 5        # s;        Time scale of CRAC recruitment
C_CRAC     = 400      # uM
n_CRAC     = 4.2

# Open CRAC channel current
def I_CRAC(V, C):
    I_crac = bar_g_CRAC * (V - bar_V_C(C))
    return I_crac

def bar_rho_CRAC(C_ER, C_CRAC, n_CRAC):
    result = rho_CRAC_n + (rho_CRAC_p - rho_CRAC_n) * (1 - hill(C_ER, C_CRAC, n_CRAC))  # (25)
    return result

# note: this is not to be passed into scipy.integrate
def drho_CRAC_dt(t, rho_CRAC, C_ER, C_CRAC, n_CRAC):
    result = (bar_rho_CRAC(C_ER, C_CRAC, n_CRAC) - rho_CRAC) / tau_CRAC     # Density of active CRAC-channels (24)
    return result

# === IP3R Dynamics ===
# IP3R constants
g_IP3R_max = 0.81
bar_g_IP3R = 0.064     # pS
C_IP3R_act = 0.21      # µM
n_IP3R_act = 1.9
n_IP3R_inh = 3.9
bar_C_IP3R_inh = 52    # µM
n_IP3R_C = 4
P_IP3R_C = 0.05        # µM
tau_IP3R = 0.100       # s 
theta_IP3R = 0.300     # s 


def g_IP3R(C):
    activation = g_IP3R_max * hill(C, C_IP3R_act, n_IP3R_act)       # Activation term (27)
    return activation

def C_IP3R_inh(P):  
    conc = bar_C_IP3R_inh * hill(P, P_IP3R_C, n_IP3R_C)             # (27)
    return conc   

def h_IP3R(C, P):
    inactivation = hill(C_IP3R_inh(P), C, n_IP3R_inh)               # Inactivation term (27)
    return inactivation

# note here: V and V_ER are to be kept at a constant V0 = V_ER as defined prior
def I_IP3R(C, P, V, V_ER):
    current = bar_g_IP3R * g_IP3R(C) * h_IP3R(C, P) * (V - V_ER - bar_V_C_ER(C))  # IP3R calcium current (28)
    return current

# Inactivation dynamics
# note: these ODEs are not to be used with scipy.integrate; merely to pass into 
def dgIP3R_dt(t, C):
    derivative = (g_IP3R_max * hill(C, C_IP3R_act, n_IP3R_act) - g_IP3R(C)) / tau_IP3R  # Activation factor (29)
    return derivative

def dhIP3R_dt(t, C, P):
    derivative = (hill(C_IP3R_inh(P), C, n_IP3R_inh) - h_IP3R(C, P)) / theta_IP3R
    return derivative

def T_of_t(t):
    if 10 <= t <= 300:  # stimulated from 10s to 200s
        return 1.6
    else:
        return 0

def dP_dt(t, C, P):
    derivative = beta_P * hill(C, C_P, n_P) * T_of_t(t) - gamma_P * P
    return derivative

# === PMCA Dynamics ===
# PMCA constants
bar_I_PMCA = 1e-5
n_PMCA = 2      # hill coefficient
tau_PMCA = 50   # s
C_PMCA = 0.1    # uM;   half-activation calcium concentration

def dgPMCA_dt(t, C, C_PMCA, g_PMCA):
    derivative = (hill(C, C_PMCA, n_PMCA) - g_PMCA) / tau_PMCA
    return derivative

# todo: implement solving for I_PMCA

# === SERCA dynamics ===
bar_I_SERCA = 3e-6     # pA
n_SERCA = 2
C_SERCA = 0.4          # uM


def I_SERCA (C):
    current = bar_I_SERCA * hill(C, C_SERCA, n_SERCA)
    return current

# === Calcium dynamics in ER ===
# ER constants
C_ER_0 = 400        # uM;   resting ER calcium level

# Calcium buffer in ER (2) (5)
def B_C(ca_conc, b, K):
    BC = (b * K) / (ca_conc + K)**2
    return BC

# Fraction of free Ca in ER (3) (6)
def free_calcium(ca_conc, b, K):
    free_ca = 1 / (1 + b / (ca_conc + K))
    return free_ca

def dCER_dt(t, C, C_ER, P):

    B_C_ER = B_C(C_ER, b_ER_0, K_ER_b)
    dCERdt = (xi_ER * (rho_SERCA * I_SERCA(C) + rho_IP3R * I_IP3R(C, P, V0, V_ER_0))) / (zCa * F * (1 + B_C_ER))

    return dCERdt
