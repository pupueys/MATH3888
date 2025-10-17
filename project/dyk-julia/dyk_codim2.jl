using BifurcationKit
using Parameters
using Plots
using DifferentialEquations
const BK = BifurcationKit

function calcium_model!(z, p, t=0)

    C, P, x000, x001, x010, x011, x100, x101, x110 = z

    #(;P, c0, c1, nu1, nu2, nu3, nu4, k3, k4, a1, a2, a3, a4, a5, d1, d2, d3, d4, d5, Ir, alpha) = p
    (;c0, c1, nu1, nu2, nu3, nu4, k3, k4, a1, a2, a3, a4, a5, d1, d2, d3, d4, d5, Ir, alpha) = p


    x111 = 1.0 - x000 - x001 - x010 - x011 - x100 - x101 - x110
    
    V1 = a4 * (C * x000 - d4 * x001)
    V2 = a4 * (C * x010 - d4 * x011)
    V3 = a5 * (C * x000 - d5 * x010)
    V4 = a5 * (C * x001 - d5 * x011)

    V5 = a1 * (P * x000 - d1 * x100)
    V6 = a1 * (P * x010 - d1 * x110)
    V7 = a3 * (P * x001 - d3 * x101)
    V8 = a3 * (P * x011 - d3 * x111)


    V9 = a2 * (C * x100 - d2 * x101)
    V10 = a2 * (C * x110 - d2 * x111)
    V11 = a5 * (C * x100 - d5 * x110)
    V12 = a5 * (C * x101 - d5 * x111)


    CER = (c0 - C) / c1

    J1 = c1 * (nu1 * x110^3 + nu2) * (CER - C)

    J2 = nu3 * C^2 / (C^2 + k3^2)


    dC = J1 - J2          
    dx000 = -V1 - V3 - V5    
    dx001 = V1 - V4 - V7    
    dx010 = V3 - V2 - V6    
    dx011 = V2 + V4 - V8     
    dx100 = V5 - V9 - V11   
    dx101 = V7 + V9 - V12   
    dx110 = V6 - V10 + V11  

    dP = nu4 * (C + (1-alpha) * k4) / (C + k4) - Ir * P        # extension

    return [dC, dP, dx000, dx001, dx010, dx011, dx100, dx101, dx110]
end

par = (
 #   P = 0.24,     # [IP3] (uM)
    c0 = 2.0,     # Total [Ca2+] (uM)
    c1 = 0.185,   # Ratio of ER volume to cytosol volume
    nu1 = 6.0,    # Maximum Ca2+ flux through IP3R channel (s⁻¹)
    nu2 = 0.11,   # Ca2+ leak flux constant (s⁻¹)
    nu3 = 0.9,    # Maximum Ca2+ intake by SERCA pump (uM⁻¹/s)
    nu4 = 1.5,
    k3 = 0.1,     # Activation constant for SERCA pump (uM)
    k4 = 1.1,
    a1 = 400.0,   # Receptor binding constants (uM⁻¹/s)
    a2 = 0.2,
    a3 = 400.0,
    a4 = 0.2,
    a5 = 20.0,
    d1 = 0.13,    # Receptor dissociation constants (uM)
    d2 = 1.049,
    d3 = 943.4e-3,
    d4 = 144.5e-3,
    d5 = 82.34e-3,
    Ir = 1,
    alpha = 0.97
)


z0 = [
    0.4995370481987472,
    1.1985546884100169,
    0.007832754088921427,
    0.027069172333116037,
    0.04751943646941965,
    0.16422216252067076,
    0.072215267381315,
    0.03439037816142146,
    0.43811267039995466
]


t_span = (0.0, 100.0)
prob = ODEProblem(calcium_model!, z0, t_span, par)

sol = solve(prob)

plot(sol.t, sol[1,:])


record_calcium(z, p; k...) = (C=z[1], P=z[2], x000=z[3], x001=z[4], x010=z[5], x011=z[6], x100=z[7], x101=z[8], x110=z[9])

prob = BifurcationProblem(calcium_model!, z0, par, (@optic _.nu4);
    record_from_solution = record_calcium
)


opts_br = ContinuationPar(p_min = 1.0, p_max = 4.0, ds = 1e-4, dsmax = 0.01,
    detect_bifurcation = 3, n_inversion = 6)


br = continuation(prob, PALC(), opts_br; normC = norminf)



scene = plot(br)
