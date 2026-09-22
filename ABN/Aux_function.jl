using LinearAlgebra, Statistics, Plots
using Parameters, LaTeXStrings
using NLsolve
using QuantEcon
using Roots
using StatsBase
using CSV, DataFrames
using Random
using JuMP, OSQP
#ENV["GKSwstype"] = "100" 
Random.seed!(1925)

function stoch_matrix(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β)
    S_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).state_values
    Π_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).p
    S_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).state_values
    Π_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).p
    S_β = QuantEcon.rouwenhorst(N, ρ_β, σ_β, μ_β).state_values
    Π_β = QuantEcon.rouwenhorst(N, ρ_β, σ_β, μ_β).p
    Π = kron(Π_g, Π_θ)
    Π = kron(Π, Π_β)
    S_I = Iterators.product(S_β, S_θ, S_g) |> collect
    S_I = reshape(S_I, (N * N * N))
    S = transpose([tup[4-k] for k in 1:3, tup in S_I])
    return (S=S, Π=Π)
end

function stoch_matrix_alt(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β)
    S=zeros(N,3)
    Π=zeros(N,N)
    S_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).state_values
    Π_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).p
    S_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).state_values
    S_β = QuantEcon.rouwenhorst(N, ρ_β, σ_β, μ_β).state_values
    S[:,1] .= S_g
    S[:,2] .= S_θ
    S[:,3] .= S_β
    return (S=S, Π=Π_g)
end


function stoch_matrix_flipped(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β)
    S=zeros(N,3)
    Π=zeros(N,N)
    S_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).state_values
    S_g = reverse(S_g)
    Π_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).p
    S_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).state_values
    S_β = QuantEcon.rouwenhorst(N, ρ_β, σ_β, μ_β).state_values
    S[:,1] .= S_g
    S[:,2] .= S_θ
    S[:,3] .= S_β
    return (S=S, Π=Π_g)
end


function stoch_matrix_onlybeta(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β)
    S=zeros(N,3)
    Π=zeros(N,N)
    S_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).state_values
    Π_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).p
    S_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).state_values
    S_β = QuantEcon.rouwenhorst(N, ρ_β, σ_β, μ_β).state_values
    S[:,1] .= μ_g*ones(N)
    S[:,2] .= S_θ
    S[:,3] .= S_β
    return (S=S, Π=Π_g)
end


function define_param(; M_β=log(0.99),
    ρ_θ=0.95,
    ρ_g=0.95,
    ρ_β=0.95,
    M_θ=log(1),
    M_g=log(15 / 100),
    S_θ=2 / 400,
    GHH="no",
    S_g=1. / 15,
    S_β=1. / 15,
    σ=2.0,
    γ=1.0,
    debt_ratio=4.0,
    T=500,
    N=5,
    μ_g=M_g * (1 - ρ_g),
    μ_θ=M_θ * (1 - ρ_θ),
    μ_β=M_β * (1 - ρ_β),
    σ_θ=S_θ * sqrt(1 - ρ_θ^2),
    σ_g=S_g * sqrt(1 - ρ_g^2),
    σ_β=S_β * sqrt(1 - ρ_β^2),
    S=stoch_matrix_alt(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β).S,
    Π=stoch_matrix_alt(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β).Π,
    horizon=5000)
    β = exp.(S[:, 3])
    params = @with_kw (β=β,
        ρ_θ=ρ_θ,
        GHH=GHH,
        ρ_g=ρ_g,
        ρ_β=ρ_β,
        μ_θ=μ_θ,
        μ_g=μ_g,
        μ_β=μ_β,
        σ_θ=σ_θ,
        σ_g=σ_g,
        σ_β=σ_β,
        σ=σ,
        γ=γ,
        N=N,
        S=S,
        debt_ratio=debt_ratio,
        T=T,
        Π=Π,
        horizon=horizon)
    return params
end


function define_param_flipped(; M_β=log(0.99),
    ρ_θ=0.95,
    ρ_g=0.95,
    ρ_β=0.95,
    M_θ=log(1),
    M_g=log(15 / 100),
    S_θ=2 / 400,
    GHH="no",
    S_g=1. / 15,
    S_β=1. / 15,
    σ=2.0,
    γ=1.0,
    debt_ratio=4.0,
    T=500,
    N=5,
    μ_g=M_g * (1 - ρ_g),
    μ_θ=M_θ * (1 - ρ_θ),
    μ_β=M_β * (1 - ρ_β),
    σ_θ=S_θ * sqrt(1 - ρ_θ^2),
    σ_g=S_g * sqrt(1 - ρ_g^2),
    σ_β=S_β * sqrt(1 - ρ_β^2),    
    S=stoch_matrix_flipped(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β).S,
    Π=stoch_matrix_flipped(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ, σ_β, μ_β, ρ_β).Π,
    horizon=5000)
    β = exp.(S[:, 3])
   
    params = @with_kw (β=β,
        ρ_θ=ρ_θ,
        GHH=GHH,
        ρ_g=ρ_g,
        ρ_β=ρ_β,
        μ_θ=μ_θ,
        μ_g=μ_g,
        μ_β=μ_β,
        σ_θ=σ_θ,
        σ_g=σ_g,
        σ_β=σ_β,
        σ=σ,
        γ=γ,
        N=N,
        S=S,
        debt_ratio=debt_ratio,
        T=T,
        Π=Π,
        horizon=horizon)
    return params
end
