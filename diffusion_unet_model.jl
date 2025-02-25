using Random
using Flux
using Statistics



# Function to project time axis on (random but fixed) freqeuncy components

struct GaussFourierProjection{T}
    W::T
end

function GaussFourierProjection(embed_dim, scale)
    # W .= randn(Float32, embed_dim÷2) .* scale
    # return(GaussFourierProjection(W))
    return(GaussFourierProjection(randn(Float32, embed_dim÷2) .* scale))
end


function (gfp::GaussFourierProjection)(t)
    t_proj = t' .* gfp.W * Float32(2π)
    [sin.(t_proj); cos.(t_proj)]
end

Flux.@functor GaussFourierProjection
Flux.trainable(gfp::GaussFourierProjection) = ()
# Flux.gpu(gfp::GaussFourierProjection) = GaussFourierProjection(gpu(gfp.W))
# Flux.cpu(gfp::GaussFourierProjection) = GaussFourierProjection(cpu(gfp.W))

# function GaussFourierProjection(embed_dim, scale)
#     W = randn(Float32, embed_dim÷2) .* scale
#     function gaussfourierproj(t)
#         t_proj = t' .* W * Float32(2π)
#         [sin.(t_proj); cos.(t_proj)]
#     end
# end

# Helper function for commonly occuring standard deviation
# Used for scaling network outputs and sampling process

marginal_prob_std(t, σ=25.0f0) = sqrt.((σ.^(2t).-1.0f0)./2.0f0./log(σ))

# Helper function for adding dims to image

expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x,(ntuple(i->1,dims)...,size(x)...))

# UNet definition

struct UNET
    layers::NamedTuple
end

function UNET(channels, embed_dim, scale)
    return UNET((
        fourier_proj = GaussFourierProjection(embed_dim, scale),
        linear = Dense(embed_dim, embed_dim, swish),
        # Encoder
        conv1 = Conv((3,3), 1=>channels[1], stride=1, bias=false),
        dense1 = Dense(embed_dim, channels[1]),
        gnorm1 = GroupNorm(channels[1], 4, swish),
        conv2 = Conv((3,3), channels[1]=>channels[2], stride=1, bias=false),
        dense2 = Dense(embed_dim, channels[2]),
        gnorm2 = GroupNorm(channels[2], 32, swish),
        # Decoder
        tconv2 = ConvTranspose((3,3), channels[2]=>channels[1], stride=1, bias=false),
        tdense1 = Dense(embed_dim, channels[1]),
        tgnorm1 = GroupNorm(channels[1], 32, swish),
        tconv1 = ConvTranspose((3,3), channels[1]+channels[1]=>1, stride=1, bias=false)
    ))    
end

Flux.@functor UNET

function (unet::UNET)(x,t)
    # Embedding layers
    embed = unet.layers.fourier_proj(t)
    embed = unet.layers.linear(embed)
    # Encoder
    h1 = unet.layers.conv1(x)
    h1 = h1 .+ expand_dims(unet.layers.dense1(embed),2)
    h1 = unet.layers.gnorm1(h1)
    
    h2 = unet.layers.conv2(h1)
    h2 = h2 .+ expand_dims(unet.layers.dense2(embed),2)
    h2 = unet.layers.gnorm2(h2)
    # Decoder
    h = unet.layers.tconv2(h2)
    h = h .+ expand_dims(unet.layers.tdense1(embed),2)
    h = unet.layers.tgnorm1(h)

    h = unet.layers.tconv1(cat(h,h1,dims=3))
    # Normalize output
    h ./ expand_dims(marginal_prob_std(t), 3)
end

# Loss function for score based diffusion

function score_loss(model, x, ϵ=1.0f-5)
    batch_size = size(x)[end]
    # random times for expectation values over time steps
    random_t = rand!(similar(x,batch_size)).*(1.0f0-ϵ).+ϵ
    # pertubations for expectation value of pdf at time t
    z = randn!(similar(x))
    std = expand_dims(marginal_prob_std(random_t),3)
    perturbed_x = x + z.*std

    score = model(perturbed_x, random_t)
    mean(
        sum((score.*std + z).^2; dims=1:(ndims(x)-1))
    )
end