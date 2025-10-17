using BifurcationKit
using Parameters
using Plots
const BK = BifurcationKit

function calcium_model!(z, p, t=0)

    C, x000, x001, x010, x011, x100, x101, x110 = z

    (;P, c0, c1, nu1, nu2, nu3, k3, a1, a2, a3, a4, a5, d1, d2, d3, d4, d5) = p


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

    return [dC, dx000, dx001, dx010, dx011, dx100, dx101, dx110]
end

par = (
    P = 0.24,     # [IP3] (uM)
    c0 = 2.0,     # Total [Ca2+] (uM)
    c1 = 0.185,   # Ratio of ER volume to cytosol volume
    nu1 = 6.0,    # Maximum Ca2+ flux through IP3R channel (s⁻¹)
    nu2 = 0.11,   # Ca2+ leak flux constant (s⁻¹)
    nu3 = 0.9,    # Maximum Ca2+ intake by SERCA pump (uM⁻¹/s)
    k3 = 0.1,     # Activation constant for SERCA pump (uM)
    a1 = 400.0,   # Receptor binding constants (uM⁻¹/s)
    a2 = 0.2,
    a3 = 400.0,
    a4 = 0.2,
    a5 = 20.0,
    d1 = 0.13,    # Receptor dissociation constants (uM)
    d2 = 1.049,
    d3 = 943.4e-3,
    d4 = 144.5e-3,
    d5 = 82.34e-3
)


z0 = [
    0.06180045, # C
    0.25477404, # x000
    0.10894608, # x001
    0.19122117, # x010
    0.08176968, # x011
    0.19598004, # x100
    0.01154823, # x101
    0.14709321  # x110
]


record_calcium(z, p; k...) = (C=z[1], x000=z[2], x001=z[3], x010=z[4], x011=z[5], x100=z[6], x101=z[7], x110=z[8])

prob = BifurcationProblem(calcium_model!, z0, par, (@optic _.P);
    record_from_solution = record_calcium
)


opts_br = ContinuationPar(p_min = 0.0, p_max = 1.5, ds = 0.001, dsmax = 0.01,
    detect_bifurcation = 3, n_inversion = 6)


br = continuation(prob, PALC(), opts_br; normC = norminf)


function recordPO(x, p; k...)
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    period = BK.getperiod(p.prob, x, p.p)
    return (;max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period, )
end
function plotPO(x, p; k...)
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    plot!(xtt.t, xtt[1,:]; markersize = 2, k...)
    plot!(xtt.t, xtt[6,:]; k...)
    scatter!(xtt.t, xtt[1,:]; markersize = 1, legend = false, k...)
end


optnewton = NewtonPar(tol = 1e-14, verbose = true, max_iterations = 25)

oopts_br_po = ContinuationPar(plot_every_step = 2, p_min =0., p_max = 60., dsmax=1e-2, ds=1e-3,max_steps = 60, newton_options = optnewton, detect_bifurcation = 0)

hopf_po = continuation(br, 1, oopts_br_po,
        PeriodicOrbitTrapProblem(M = 100);
        normC = norminf,
        #δp = 0.02,
        #bothside = true,
        # jacobian_ma = BK.MinAug(),
        verbosity = 3, plot = true,
        alg = PALC(),
        callback_newton = BK.cbMaxNorm(1e2),
        record_from_solution = recordPO,
        plot_solution = (x, p; k...) -> begin
        plotPO(x, p; k...)
        plot!(br,  subplot=1, putbifptlegend = false)
        end,
        finalise_solution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
            return z.u[end] < 150
            true
        end,
    )

scene = plot(br)
plot(scene, hopf_po)