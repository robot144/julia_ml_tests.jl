# model_1d_wave_periodic.jl
# 1D wave equation with periodic boundary conditions
# 1D Wave-equation / linearized shallow-water equation on a periodic domain
#
# The one-dimensional shallow-water equations can be linearized and simplified to the 1D wave-equation:
# $\partial h/\partial t + D \partial u / \partial x = 0$
# $\partial u/\partial t + g \partial h / \partial x = 0$
# where $h$ denotes the water-height above the reference, $u$ the velocity, $D$ the depth below the reference and $t,x$ time and space.

"""
    Wave1DPeriodic
    container for the parameters of the 1D wave equation
    We use a 'functor', i.e. a struct that behaves like a function.
    So this function can contain data, eg parameters.
"""
struct Wave1DPeriodic_cpu
    g::Float64 # gravity
    D::Float64 # depth
    L::Float64 # length
    dx::Float64 # spatial step
    nx::Int64 # number of spatial points
end

"""
    initial state for the 1D wave equation
    We use ComponentArrays to store the state variables
    x.h : height
    x.u : velocity
    The data is stored on a regular but staggered grid
    h at: 0.0, dx, 2dx, ...
    u at: dx/2, 3dx/2, 5dx/2, ...

    The initial state is a Gaussian bump in the height field
    of height h0, centered at x_center with a width of w times the length
    and located at c times the length.
"""
function initial_state_bump(f::Wave1DPeriodic_cpu, h0=1.0, w=0.05, c=0.5)
    x_center=f.L*c
    width=f.L*w
    x_h = 0.0:f.dx:(f.L-f.dx/2)
    h = h0.*exp.(-((x_h .- x_center).^2) ./ (2 * width^2))
    u = zeros(f.nx) .+ 0.0
    x = ComponentVector(h=h,u=u)
    return x
end

"""
   Compute the spatial derivative of h
   function dh_dx!(∂h∂x,h,dx)
"""
function dh_dx!(∂h∂x,h,dx)
    nx=length(h)
    for i in 1:(nx-1)
        ∂h∂x[i] = (h[i+1]-h[i])/(dx)
    end
    ∂h∂x[end] = (h[1]-h[end])/dx
end

"""
   Compute the spatial derivative of u
   function du_dx!(∂u∂x,u,dx)
"""
function du_dx!(∂u∂x,u,dx)
    nx=length(u)
    for i in 2:nx
        ∂u∂x[i] = (u[i]-u[i-1])/(dx)
    end
    ∂u∂x[1] = (u[1]-u[end])/dx
end

"""
    Wave1DPeriodic
    Compute time derivative of the state
"""
function (f::Wave1DPeriodic_cpu)(dx_dt,x,p,t)
    dx=f.dx
    g=f.g
    D=f.D
    # temporary variables
    ∂h∂x = similar(x.h) # allocating version, is not optimal for performance
    ∂u∂x = similar(x.u)
    # compute spatial derivatives
    dh_dx!(∂h∂x,x.h,dx)
    du_dx!(∂u∂x,x.u,dx)
    # compute time derivatives
    @. dx_dt.u = -g * ∂h∂x
    @. dx_dt.h = -D * ∂u∂x
end

