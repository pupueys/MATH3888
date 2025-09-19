from equations_fixed import *

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

    C, C_ER, V, P, rho_CRAC, g_PMCA, g_IP3R, h_IP3R = y          

    B_cyt = (b0 * Kb) / (C + Kb)**2

    # IP3 dynamics
    dPdt = dP_dt(t, C, P)

    dgIP3Rdt = dgIP3R_dt(C, g_IP3R)
    dhIP3Rdt = dhIP3R_dt(C, P, h_IP3R)

    # PMCA activation dynamics
    dgPMCAdt = dgPMCA_dt(t, C, C_PMCA, g_PMCA)

    # rho CRAC dynamics
    drhoCRACdt = drho_CRAC_dt(t, rho_CRAC, C_ER, C_CRAC, n_CRAC)

    # channel currents (in A)
    I_PMCA_chan  = bar_I_PMCA * g_PMCA
    I_CRAC_chan  = I_CRAC(V0, C)
    I_SERCA_chan = I_SERCA(C)
    #I_IP3R_chan  = I_IP3R(C, C_ER, P, V0, V0)
    I_IP3R_chan = bar_g_IP3R * g_IP3R * h_IP3R * (V0 - V0 - bar_V_C_ER(C, C_ER))

    # membrane potential dynamics
    dVdt = (I_CRAC_chan + I_PMCA_chan + I_SERCA_chan + I_IP3R_chan) / (C_m * A_cell)   

    # flux contributions
    term_PMCA = current_to_flux(rho_PMCA, I_PMCA_chan, xi)
    term_CRAC = current_to_flux(rho_CRAC, I_CRAC_chan, xi)
    term_SERCA = current_to_flux(rho_SERCA, I_SERCA_chan, xi_ERC)
    term_IP3R = current_to_flux(rho_IP3R, I_IP3R_chan, xi_ERC)

    # ER dynamics
    dCERdt = dCER_dt(t, C, C_ER, V0, P)

    # cytosolic calcium dynamics
    dCdt = -(term_PMCA + term_CRAC + term_SERCA + term_IP3R) / (1 + B_cyt)

    return [dCdt, dCERdt, dVdt, dPdt, drhoCRACdt, dgPMCAdt, dgIP3Rdt, dhIP3Rdt]

# initial conditions
g_PMCA_0 = 0.5
g_IP3R_0 = 0.05
h_IP3R_0 = 0.9
y0 = [C0, C_ER_0, V0, P0, rho_CRAC_0, g_PMCA_0, g_IP3R_0, h_IP3R_0]

# # Solving
t_span = (0, 300)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(calcium_ode, t_span, y0, t_eval=t_eval)

# plotting
plt.plot(sol.t, sol.y[0] * 1e3, label='Cytosolic Calcium [C]')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (µM)')
plt.legend()
plt.title('Calcium Dynamics Simulation')
plt.grid(True)
plt.show()

# --- Unpack solution ---
t = sol.t
C, C_ER, V, P, rho_CRAC, g_PMCA, gIP3R, hIP3R = sol.y

# --- Compute currents along trajectory ---
I_PMCA_plot = bar_I_PMCA * g_PMCA
I_CRAC_plot = np.array([I_CRAC(V0, c) for c in C])
I_SERCA_plot = np.array([I_SERCA(c) for c in C])
I_IP3R_plot = np.array([I_IP3R(c, cer, p, V0, V0) for c, cer, p in zip(C, C_ER, P)])

# --- Compute fluxes (cytosolic sign convention) ---
term_PMCA = np.array([current_to_flux(rho_PMCA, I, xi) for I in I_PMCA_plot])
term_CRAC = np.array([current_to_flux(rho_CRAC[i], I, xi) for i, I in enumerate(I_CRAC_plot)])
term_SERCA = np.array([current_to_flux(rho_SERCA, I, xi_ERC) for I in I_SERCA_plot])
term_IP3R = np.array([current_to_flux(rho_IP3R, I, xi_ERC) for I in I_IP3R_plot])

# Net cytosolic Ca flux
I_net_cyt = term_PMCA + term_CRAC + term_SERCA + term_IP3R
# Net ER Ca flux (SERCA in, IP3R out)
I_net_ER = term_SERCA + term_IP3R

# --- Plotting ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: channel currents
axes[0].plot(t, I_PMCA_plot, label="PMCA current")
axes[0].plot(t, I_CRAC_plot, label="CRAC current")
axes[0].plot(t, I_SERCA_plot, label="SERCA current")
axes[0].plot(t, I_IP3R_plot, label="IP3R current")
axes[0].set_ylabel("Current (A)")
axes[0].set_title("Channel Currents")
axes[0].legend()

# Panel 2: cytosolic fluxes
axes[1].plot(t, term_PMCA, label="PMCA flux (cyt)")
axes[1].plot(t, term_CRAC, label="CRAC flux (cyt)")
axes[1].plot(t, term_SERCA, label="SERCA flux (cyt→ER)")
axes[1].plot(t, term_IP3R, label="IP3R flux (ER→cyt)")
axes[1].set_ylabel("Flux (M/s)")
axes[1].set_title("Cytosolic Calcium Fluxes")
axes[1].legend()

# Panel 3: net fluxes
axes[2].plot(t, I_net_cyt, "--", label="Net cytosolic Ca flux")
axes[2].plot(t, I_net_ER, "--", label="Net ER Ca flux")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Flux (M/s)")
axes[2].set_title("Net Calcium Fluxes")
axes[2].legend()

plt.tight_layout()
plt.show()

