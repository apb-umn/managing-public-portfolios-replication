

#= include("Aux_function.jl")
params=define_param(N=20,prod="no",T=1)

include("Utility.jl")

include("Ramsey_complete.jl")
ss  = simulation()

 =#
"""
get_approximate(debt_gdp, ABN portfolio, BEGS portfolio, π_Q, π_X, Σ, Σ_Q, Σ_X, β̂, maturities, weights,State; tol=10^(-5))
"""
 function get_approximate(B, ω_abn, ω_BEGS, π_Q, π_X, Σ, Σ_Q, Σ_X, β̃, mat, weights,State,tol)
    
    cc = (π_Q * Σ_Q + π_X * Σ_X) * β̃
    π_Q * Σ_Q * β̃
    π_X * Σ_X * β̃
    (π_X * Σ_X * β̃) ./ (π_Q * Σ_Q * β̃)
 

 
    println("constraint error at initial = ", abs.(Σ * ω_abn - cc))

    error_omega_target=norm(Σ*ω_BEGS-cc)
    error_true=norm(Σ*ω_abn-cc)
    println("error_omega_target = ",error_omega_target)
    println("error_true = ",error_true)

    verbose = true
    ω_BEGS_approx = zeros(length(ω_abn)+1)
    B̃_BEGS_approx = zeros(length(ω_abn)+1)
    # Create JuMP model
    #model = Model(OSQP.Optimizer)

    model =Model(Ipopt.Optimizer)
    #set_silent(model)
    @variable(model, x[1:length(ω_abn)])
    set_start_value.(x, ω_BEGS)
    @objective(model, Min, (x - ω_abn)' * (x - ω_abn))
    @constraint(model, -tol .<= Σ * x - cc .<= tol)
    set_attribute(model, "constr_viol_tol", tol)
    #set_attribute(model, "eps_abs", 1e-7)
    set_optimizer_attribute(model, "max_iter", 100)
    optimize!(model)
    print(model)
    println("Objective value: ", objective_value(model))
    println("x = ", value.(x))
    #println("omega = ", ω_abn)
    println("constraint error = ", abs.(Σ * value.(x) - cc))
    ω_BEGS_approx = vcat(1 - sum(value.(x)), value.(x))
    
    


    full_mat = vcat([1], mat)
    full_weights = vcat([1], weights)


    pp_State = ([[sum(ss.P.(full_mat[n], State) .* full_weights[n] ./ sum(full_weights[n])) for n in 1:length(full_mat)] for State in unique(ss.s)])[1]
    B̃_BEGS_approx=(ω_BEGS_approx * B) ./ pp_State
    return (ω_BEGS_approx, B̃_BEGS_approx, π_X * Σ_X * β̃ ./ (π_Q * Σ_Q * β̃), State)
end

#Mat from 1 to N  
#Bucket 10 quarters(?) if not 20 quarters
#Console + risk free
##All pictures do Angeletos + Target portfolio
##Check π_X*Σ_X*w./(π_Q*Σ_Q*w)