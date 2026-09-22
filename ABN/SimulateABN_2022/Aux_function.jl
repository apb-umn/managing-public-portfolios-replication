using LinearAlgebra, Statistics, Plots
BLAS.set_num_threads(1)
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

function stoch_matrix(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ; prod="no")
    S_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).state_values
    Π_g = QuantEcon.rouwenhorst(N, ρ_g, σ_g, μ_g).p
    if prod == "no"
        S = [S_g[k] for k in 1:N]
        S = hcat(S, zeros(N))
        return (S=S, Π=Π_g)
    else
        S_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).state_values
        Π_θ = QuantEcon.rouwenhorst(N, ρ_θ, σ_θ, μ_θ).p
        Π = kron(Π_g, Π_θ)
        S = zeros(N * N, 2)
        for k in 1:N
            for l in 1:N
                S[(k-1)*N+l, :] = [S_g[k], S_θ[l]]
            end
        end
        return (S=S, Π=Π)
    end
end

function define_param(; β=0.99,
    ρ_θ=0.95,
    ρ_g=0.95,
    M_θ=log(1),
    M_g=log(15 / 100),
    S_θ=2 / 400,
    S_g=1.2 / 15,
    σ=2.0,
    γ=1.0,
    debt_ratio=4.0,
    T=500,
    N=5,
    μ_g=M_g * (1 - ρ_g),
    μ_θ=M_θ * (1 - ρ_θ),
    σ_θ=S_θ * sqrt(1 - ρ_θ^2),
    σ_g=S_g * sqrt(1 - ρ_g^2),
    prod="no",
    S=stoch_matrix(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ; prod=prod).S,
    Π=stoch_matrix(N, ρ_g, σ_g, μ_g, ρ_θ, σ_θ, μ_θ; prod=prod).Π,
    horizon=120,
    GHH="no")
    params = @with_kw (β=β,
        ρ_θ=ρ_θ,
        ρ_g=ρ_g,
        μ_θ=μ_θ,
        μ_g=μ_g,
        σ_θ=σ_θ,
        σ_g=σ_g,
        σ=σ,
        γ=γ,
        N=N,
        S=S,
        debt_ratio=debt_ratio,
        T=T,
        Π=Π,
        horizon=horizon,
        GHH=GHH)
    return params
end
