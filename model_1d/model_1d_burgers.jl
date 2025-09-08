# model_1d_burgers.jl
# A simple 1D Burgers equation model with periodic boundary conditions

# 1D Burgers equation on a periodic domain

# The [Burgers' equation](https://en.wikipedia.org/wiki/Burgers%27_equation) is a Partial Differential Equation (PDE) that describes 
# convection and diffusion. It can also be viewed as a simplified version of the 
# [Navier-Stokes equation](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations). Here, we start with the equation in the following form:
# $\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}=\nu \frac{\partial^2 u}{\partial x^2}$
# on the domain $x \in [0, L]$ and $0 \le t \le T$. The domain is considered periodic in $x$. And the initial condition is given by:
# $u(x,t=0) = 1.0 + 0.5 \cos(2\pi/L x)$ 


"""
    Wave1DPeriodic_cpu

    A 1D wave equation with periodic boundary conditions.
    The state is represented as a ComponentArray with two fields:
    - h: height
    - u: velocity
"""
struct burgers_equation
    Δx::Float64 # grid spacing
    ν::Float64 # viscosity
    du_dx::Vector{Float64} # temporary storage for du/dx
    d2u_dx2::Vector{Float64} # temporary storage for d2u/dx2
end

function dx!(du,u,Δx) #upwind du_dx for u*du/dx term -> requires u>=0
    n=length(u)
    for i=2:n
        du[i]=(u[i]-u[i-1])/Δx
    end
    du[1]=(u[1]-u[end])/Δx
end

function dx2!(du,u,Δx) #central second difference for ν d2u/dx2 term
    n=length(u)
    for i=2:n-1
        du[i]=(u[i+1]-2*u[i]+u[i-1])/(Δx^2)
    end
    du[1]=(u[end]-2*u[1]+u[2])/(Δx^2)
    du[end]=(u[end-1]-2*u[end]+u[1])/(Δx^2)
end

"""
# Define the function that computes the time derivative of the state
 du/dt = -u * du/dx + ν * d2u/dx2
"""
function (f::burgers_equation)(du_dt, u, p, t)
    Δx=f.Δx
    ν=f.ν
    du_dx=f.du_dx
    d2u_dx2=f.d2u_dx2
    dx!(du_dx,u,Δx)
    dx2!(d2u_dx2,u,Δx)
    @. du_dt = -u*du_dx #+  ν*d2u_dx2 
    # The diffusion term is switched off because the first order upwind scheme is numerically diffusive
end

function burgers_initial_condition(x,k,u_mean=1.0,u_amplitude=0.5)
    return u_mean .+ u_amplitude.*cos.(k*x)
end

