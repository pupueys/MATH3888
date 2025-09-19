from equations import *

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Simplifying flux terms for ODE ===
def current_to_flux(rho, I_channel_A, area_to_vol_ratio=xi):
    # returns the respective current to the respective dCdt term
    return (area_to_vol_ratio * rho * I_channel_A) / (zCa * F)

# === ODE for calcium in the ER ===
def dCER_dt(t, C, C_ER, V, P):
    I_SERCA_chan = I_SERCA(C)                      # current per SERCA (A)
    I_IP3R_chan  = I_IP3R(C, C_ER, P, V, V)      # IP3R current (A)

    B_C_ER = B_C(C_ER, b_ER_0, K_ER_b)
    dCERdt = ( current_to_flux(rho_SERCA, I_SERCA_chan, xi_ER)
             + current_to_flux(rho_IP3R, I_IP3R_chan, xi_ER) ) / (1 + B_C_ER)
    
    return dCERdt

def calcium_ode(t, y):

    C, C_ER, V, P, rho_CRAC, g_PMCA = y          

    B_cyt = (b0 * Kb) / (C + Kb)**2

    # IP3 dynamics
    dPdt = dP_dt(t, C, P)

    # PMCA activation dynamics
    dgPMCAdt = dgPMCA_dt(t, C, C_PMCA, g_PMCA)

    # rho CRAC dynamics
    drhoCRACdt = drho_CRAC_dt(t, rho_CRAC, C_ER, C_CRAC, n_CRAC)

    # channel currents (in A)
    I_PMCA_chan  = bar_I_PMCA * g_PMCA
    I_CRAC_chan  = I_CRAC(V0, C)
    I_SERCA_chan = I_SERCA(C)
    I_IP3R_chan  = I_IP3R(C, C_ER, P, V0, V0)

    # membrane potential dynamics
    dVdt = -(I_CRAC_chan + I_PMCA_chan + I_SERCA_chan + I_IP3R_chan) / (C_m * A_cell)   

    # flux contributions
    term_PMCA = current_to_flux(rho_PMCA, I_PMCA_chan, xi)
    term_CRAC = current_to_flux(rho_CRAC, I_CRAC_chan, xi)
    term_SERCA = current_to_flux(rho_SERCA, I_SERCA_chan, xi_ERC)
    term_IP3R = current_to_flux(rho_IP3R, I_IP3R_chan, xi_ERC)

    # ER dynamics
    dCERdt = dCER_dt(t, C, C_ER, V0, P)

    # cytosolic calcium dynamics
    dCdt = -(term_PMCA + term_CRAC + term_SERCA + term_IP3R) / (1 + B_cyt)

    return [dCdt, dCERdt, dVdt, dPdt, drhoCRACdt, dgPMCAdt]

# initial conditions
g_PMCA_0 = 0.5
y0 = [C0, C_ER_0, V0, P0, rho_CRAC_0, g_PMCA_0]

# # Solving
t_span = (0, 300)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(calcium_ode, t_span, y0, t_eval=t_eval)

# plotting
plt.plot(sol.t, sol.y[0] * 1e6, label='Cytosolic Calcium [C]')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (µM)')
plt.legend()
plt.title('Calcium Dynamics Simulation')
plt.grid(True)
plt.show()

# === EXTRAS ===
# Phase trajectory: cytosolic Ca vs ER Ca
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(sol.y[0] * 1e6, sol.y[1] * 1e6, 'k-', lw=2, label='Trajectory')  # µM
ax.plot(y0[0] * 1e6, y0[1] * 1e6, 'go', ms=10, label='Start')

ax.set_title('Phase Trajectory: Cytosol vs ER Calcium', fontsize=16)
ax.set_xlabel('Cytosolic Ca [C] (µM)', fontsize=12)
ax.set_ylabel('ER Ca [C_ER] (µM)', fontsize=12)

ax.legend()
ax.grid(True)
plt.show()
