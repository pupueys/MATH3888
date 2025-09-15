from equations import *

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def current_to_dCdt_term(rho, I_channel_A, area_to_vol_ratio=xi):
    # returns the respective current to the respective dCdt term
    return (area_to_vol_ratio * rho * I_channel_A) / (zCa * F) * 1e6

def dCER_dt(t, C, C_ER, P):

    I_SERCA_chan = I_SERCA(C)                    # current per SERCA (pA)
    I_IP3R_chan  = I_IP3R(C, P, V0, V_ER_0)      # IP3R current (per channel)

    B_C_ER = B_C(C_ER, b_ER_0, K_ER_b)
    dCERdt = ( current_to_dCdt_term(rho_SERCA, I_SERCA_chan, xi_ER)
              + current_to_dCdt_term(rho_IP3R, I_IP3R_chan, xi_ER) ) / (1 + B_C_ER)
    return dCERdt

def calcium_ode (t, y):
    
    C, C_ER, V, P, rho_CRAC, g_PMCA = y          # unpacking variables

    B_C = (b0 * Kb) / (C + Kb)**2

    # IP3 dynamcis
    dPdt = dP_dt(t, C, P)

    # PMCA activation dynamics
    dgPMCAdt = dgPMCA_dt(t, C, C_PMCA, g_PMCA)

    # rho CRAC dynamics
    drhoCRACdt = drho_CRAC_dt(t, rho_CRAC, C_ER, C_CRAC, n_CRAC)

    # ER dynamics
    dCERdt = dCER_dt(t, C, C_ER, P)


    # channel currents (ensure units: A per channel or pA per channel)
    I_PMCA_chan = bar_I_PMCA * g_PMCA            # current per PMCA channel (units must be consistent)
    I_CRAC_chan  = I_CRAC(V, C)                 # current per CRAC channel
    I_SERCA_chan = I_SERCA(C)                    # current per SERCA (pA)
    I_IP3R_chan  = I_IP3R(C, P, V, V_ER_0)      # IP3R current (per channel)

    dVdt = -(I_CRAC_chan + I_PMCA_chan + I_SERCA_chan + I_IP3R_chan) / C_m

    # PMCA: removes Ca from cytosol -> negative contribution to cytosol
    term_PMCA = current_to_dCdt_term(rho_PMCA, I_PMCA_chan, xi)

    # CRAC: influx into cytosol -> positive
    term_CRAC  = current_to_dCdt_term(rho_CRAC, I_CRAC_chan, xi)

    # SERCA: pumps cytosol -> ER (removes from cytosol), so negative in cytosol
    term_SERCA = current_to_dCdt_term(rho_SERCA, I_SERCA_chan, xi_ERC)

    # IP3R: releases from ER -> cytosol: positive for cytosol
    term_IP3R  = current_to_dCdt_term(rho_IP3R, I_IP3R_chan, xi_ERC)

    dCdt = -(term_PMCA +
            term_CRAC +
            term_SERCA +
            term_IP3R) / (1 + B_C)

# Only print every 10 seconds
    if int(t) % 10 == 0:
        print(f"t={t:.1f}s | C={C:.3f} µM | C_ER={C_ER:.1f} µM | P={P:.3f} µM | "
          f"rho_CRAC={rho_CRAC:.3f} | g_PMCA={g_PMCA:.3f} | "
          f"I_CRAC={I_CRAC_chan:.3e} | I_IP3R={I_IP3R_chan:.3e} | "
          f"I_SERCA={I_SERCA_chan:.3e} | I_PMCA={I_PMCA_chan:.3e}")


    return [dCdt, dCERdt, dVdt, dPdt, drhoCRACdt, dgPMCAdt]



# Defining initial conditions
g_PMCA_0 = 0.5
y0 = [C0, C_ER_0, V0, P0, rho_CRAC_0, g_PMCA_0]

t_span = (0, 300)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(calcium_ode, t_span, y0, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label='Cytosolic Calcium [C]')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (uM)')
plt.legend()
plt.title('Calcium Dynamics Simulation')
plt.show()
