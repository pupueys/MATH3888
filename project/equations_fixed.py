import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Hill function ===
def hill(X, K, n):
    return X**n / (X**n + K**n)
    
# === Constants (IN SI) ===
# Cell geometry
R_cell      = 8e-6          # m;            T-cell radius
f_R         = 0.25          # unitless;     Nuclear radius fraction
f_V         = 0.01          # unitless;     ER volume fraction
f_A         = 30            # unitless;     ER surface multiplier
C_m         = 28e-3         # F/m^2;        Membrane capacitance 

# Ions and potential
T           = 310           # K;            Cell temperature
V0          = -0.06         # V;            Resting membrane potential
V_ER_0      = -0.06         # V;            Resting ER potential
C0          = 1e-4          # mol/m^3;      Resting calcium
C_ext       = 2000e-3       # mol/m^3;      Extracellular calcium

dV_C        = 0.078         # V;            Reverse potential shift
dV_C_ER     = 0.063         # V;            ER-reversal potential shift

# Calcium buffer
b0          = 100e-3        # mol/m^3      
Kb          = 0.1e-3        # mol/m^3
b_ER_0      = 30000e-3      # mol/m^3
K_ER_b      = 100e-3        # mol/m^3

# Second messengers
P0          = 0.0087e-3     # mol/m^3
beta_P      = 0.0006e-3     # mol/m^3/s
gamma_P     = 0.01149       # 1/s
C_P         = 0.5e-3        # mol/m^3
n_P         = 1

# Channel densities (/m^2)
rho_IP3R   = 11.35e12       # /m^2;         ER-IP3R density                 (fixed by steady-state)
rho_SERCA  = 700e12         # /m^2;         ER-SERCA density                (Variable)
rho_PMCA   = 68.57e12       # /m^2;         PMCA density                    (Fixed by steady-state)
rho_CRAC_0 = 0.6e12         # /m^2;         Resting active CRAC density     (Variable)
rho_CRAC_p = 3.9e12         # /m^2;         Max active CRAC density         (Variable)
rho_CRAC_n = 0.5115e12      # /m^2;         Min active CRAC density         (fixed by steady-state)

# Physical constants
R          = 8.314          # J/mol/K;      Rydberg constant
F          = 96485          # C/mol;        Faraday constant
zCa        = 2              # unitless;     Valence of Ca

# === Derived geometry ===
R_nucleus = f_R * R_cell
A_cell  = 4 * np.pi * R_cell**2
Vol_cyt = (4/3) * np.pi * R_cell**3 * (1-f_V-f_R**3)
tilde_Vol_ER = (4/3) * np.pi * f_V * R_cell**3
A_ER    = 4 * np.pi * f_A * ((3*tilde_Vol_ER)/(4*np.pi))**(2/3)

xi     = A_cell / Vol_cyt
xi_ERC = A_ER / Vol_cyt
xi_ER  = A_ER / tilde_Vol_ER

# === Reverse Potentials ===
def bar_V_C(C):
    return (R * T / (zCa * F)) * np.log(C_ext / C) - dV_C


def bar_V_C_ER(C, C_ER):
    return (R * T / (zCa * F)) * np.log(C_ER / C) - dV_C_ER

# === CRAC Dynamics ===
# CRAC constants
bar_g_CRAC = 0.002e-12  # S (0.002 pS)
tau_CRAC   = 5.0        # s
C_CRAC     = 400e-3     # mol/m^3
n_CRAC     = 4.2

# Open CRAC channel current
def I_CRAC(V, C):
    return bar_g_CRAC * (V - bar_V_C(C))

def bar_rho_CRAC(C_ER, C_CRAC, n_CRAC):
    return rho_CRAC_n + (rho_CRAC_p - rho_CRAC_n) * (1 - hill(C_ER, C_CRAC, n_CRAC))    # (25)

def drho_CRAC_dt(t, rho_CRAC, C_ER, C_CRAC, n_CRAC):
    return (bar_rho_CRAC(C_ER, C_CRAC, n_CRAC) - rho_CRAC) / tau_CRAC                   # Density of active CRAC-channels (24)

# === IP3R Dynamics ===
# IP3R constants
g_IP3R_max = 0.81
bar_g_IP3R = 0.064e-12   # S
C_IP3R_act = 0.21e-3     # mol/m^3
n_IP3R_act = 1.9
n_IP3R_inh = 3.9
bar_C_IP3R_inh = 52e-3   # mol/m^3
n_IP3R_C = 4
P_IP3R_C = 0.05e-3       # mol/m^3
tau_IP3R = 0.100         # s
theta_IP3R = 0.300       # s

def g_IP3R(C):
    return g_IP3R_max * hill(C, C_IP3R_act, n_IP3R_act)                                 # Activation term (27)

def C_IP3R_inh(P):
    return bar_C_IP3R_inh * hill(P, P_IP3R_C, n_IP3R_C)                                 # (27)

def h_IP3R(C, P):
    return hill(C_IP3R_inh(P), C, n_IP3R_inh)                                           # Inactivation term (27)

# note here: V and V_ER are to be kept at a constant V0 = V_ER as defined prior
def I_IP3R(C, C_ER, P, V, V_ER):
    return bar_g_IP3R * g_IP3R(C) * h_IP3R(C, P) * (V - V_ER - bar_V_C_ER(C, C_ER))     # IP3R calcium current (28)

def dgIP3R_dt(C, gIP3R):
    return  (g_IP3R_max * hill(C, C_IP3R_act, n_IP3R_act) - gIP3R) / tau_IP3R 

def dhIP3R_dt(C, P, hIP3R):
    return (hill(C_IP3R_inh(P), C, n_IP3R_inh) - hIP3R) / theta_IP3R

# === IP3 dynamics ===
# Stimulation time
def T_of_t(t):
    return 1.6 if 10 <= t <= 300 else 1                           

def dP_dt(t, C, P):
    return beta_P * hill(C, C_P, n_P) * T_of_t(t) - gamma_P * P

# === PMCA Dynamics ===
# PMCA constants
bar_I_PMCA = 1e-5 * 1e-12   # A
tau_PMCA = 50.0             # s
n_PMCA = 2                  # hill coefficient
C_PMCA = 0.1e-3             # mol/m^3

def dgPMCA_dt(t, C, C_PMCA, g_PMCA):
    return (hill(C, C_PMCA, n_PMCA) - g_PMCA) / tau_PMCA

# === SERCA Dynamics ===
# SERCA constants
bar_I_SERCA = 6e-6 * 1e-12  # A (6e-6 pA)n_SERCA = 2
n_SERCA = 2             
C_SERCA = 0.4e-3            # mol/m^3

def I_SERCA(C):
    return bar_I_SERCA * hill(C, C_SERCA, n_SERCA)

# === Calcium dynamics in ER ===
# ER constants
C_ER_0 = 400e-3                         # mol/m^3

# Calcium buffer in ER (2) (5)
def B_C(ca_conc, b, K):
    return (b * K) / (ca_conc + K)**2

# Fraction of free Ca in ER (3) (6)
def free_calcium(ca_conc, b, K):
    return 1 / (1 + b / (ca_conc + K))