# include("diffusion_unet_model.jl")
include("unet_large.jl")
using Plots
using Random
using ProgressMeter: Progress, next!, @showprogress
using CUDA

# Helper function to convert model output to (set of) images

function ConvertToImage(x, y_size)
    Gray.(permutedims(vcat(reshape.(Flux.chunk(x,y_size), 28, :)...), (2,1)))
end

# Function to initialize sampling parameters

function setup_sampler(device, num_images, num_steps, ϵ=1.0f-3)
    t = ones(Float32, num_images)
    init_x = (
        randn(Float32, (28,28,1,num_images)).*
        expand_dims(marginal_prob_std(t),3)
    ) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

# Function for std at time t
diffusion_coeff(t, σ=convert(eltype(t), 25.0f0)) = σ.^t

# Euler-Maruyama sampler for reverse diffusion process

function EM_sampler(model, init_x, time_steps, Δt)
    x = mean_x = init_x
    progress = Progress(length(time_steps))
    for (index,time_step) in enumerate(time_steps)
        batch_time = fill!(similar(init_x,size(init_x)[end]),1) .* time_step
        g = diffusion_coeff(batch_time)
        mean_x = x .+ expand_dims(g,3).^2 .* model(x,batch_time).*time_step
        x = mean_x .+ sqrt(Δt).*expand_dims(g,3).*randn!(similar(x))
        # x = mean_x .+ sqrt(Δt).*expand_dims(g,3).*randn(Float32,size(x))
        next!(progress, showvalues=[(:time_step, index)])
    end
    return mean_x
end

function EM_inpainter(model, x_true, mask, init_x, time_steps, Δt)
    x = mean_x = init_x
    for time_step in time_steps
        batch_time = fill!(similar(init_x, size(init_x)[end]),1).*time_step
        g = diffusion_coeff(batch_time)
        mean_x = x .+ expand_dims(g,3).^2 .* model(x,batch_time).*time_step
        x = mean_x .+ sqrt(Δt) .* expand_dims(g,3) .* randn!(similar(x))
        data = x_true + randn!(similar(x_true)).*expand_dims(batch_time,3)
        mean_x = mean_x.*mask + x_true.*(1 .- mask)
        x = x.*mask + data.*(1 .- mask)
    end
    return mean_x
end
# MCMC correction to EM sampling

function MCMC_sampler(model, init_x, time_steps, Δt, snr=0.16f0)
    x = mean_x = init_x
    progress = Progress(length(time_steps))
    for (index,time_step) in enumerate(time_steps)
        batch_time = fill!(similar(init_x,size(init_x)[end]),1).*time_step
        # Correction step
        grad = model(x,batch_time)
        num_px = prod(size(grad)[1:end-1])
        grad_batch = reshape(grad, (size(grad)[end], num_px))
        grad_norm = mean(sqrt, sum(abs2, grad_batch, dims=2))
        noise_norm = Float32(sqrt(num_px))
        langevin_step_size = 2*(snr*noise_norm/grad_norm)^2
        x += (
            langevin_step_size .* grad .+
            sqrt(2*langevin_step_size).*randn!(similar(x))
        )
        # Predictor (EM sampling)
        g = diffusion_coeff(batch_time)
        mean_x = x .+ expand_dims((g.^2),3) .* model(x,batch_time).*Δt
        x = mean_x + sqrt.(expand_dims((g.^2),3).*Δt).*randn!(similar(x))
        
        next!(progress, showvalues=[(:time_step, index)])
        
    end
    return mean_x
end

function MCMC_inpainter(model, x_true, masks, init_x, time_steps, Δt, snr=0.16f0)
    x = mean_x = init_x
    for time_step in time_steps
        batch_time = fill!(similar(init_x, size(init_x)[end]),1).*time_step
        # Correction step
        grad = model(x, batch_time)
        num_px = prod(size(grad)[1:end-1])
        grad_batch = reshape(grad, (size(grad)[end], num_px))
        grad_norm = mean(sqrt, sum(abs2, grad_batch, dims=2))
        noise_norm = Float32(sqrt(num_px))
        langevin_step_size = 2*(snr*noise_norm/grad_norm)^2
        x += (
            langevin_step_size .* grad .+
            sqrt(2*langevin_step_size).*randn!(similar(x))
        )
        # Predictor step
        g = diffusion_coeff(batch_time)
        mean_x = x .+ expand_dims((g.^2),3).*model(x,batch_time).*Δt
        x = mean_x + sqrt.(expand_dims((g.^2),3).*Δt).*randn!(similar(x))
        data = x_true + randn!(similar(x_true)).*expand_dims(marginal_prob_std(batch_time), 3)
        mean_x = mean_x.*masks + x_true.*(1 .- masks)
        x = x.*masks + data.*(1 .- masks)    
    end
    return mean_x
end