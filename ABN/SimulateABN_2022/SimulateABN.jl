#=File generating the Tauchen discretization and the paramters=#
include("Aux_function.jl")

params = define_param(S_θ=10^(-5), prod="no", GHH="no",N=50)

#BASELINE
#= params=define_param(T=500,S_θ =10^(-5),ρ_g=0.90,ρ_θ=0.95)
                    #=By default the parameters are (;β=.99,
                    ρ_θ =   .95,
                    ρ_g =   .95,
                    M_θ =   log(1),
                    M_g =   log(15/100),
                    S_θ =   2/400,
                    S_g =   1.2/15,
                    σ   =   2.0,   
                    γ   =   1.0,
            debt_ratio  =   4.0,
                    T   =   100,
                    N   =   5)
 =#You can modify any of them by putting parameter_name=parameter value =#


#=File defining all the partial derivatives for the utility function=#
include("Utility.jl")

#=File generating the allocation at time 0, 
                                at time t>0, 
        the simulation_Ramsey given s_0, 
        the simulation_Ramsey where s_0 is drawn from the stationnary distribution =#
include("Ramsey_complete.jl")
ss = simulation_Ramsey()

#=Function drawing s_0 from the stationnary distribution and simulating for T periods

    Returns (s the time series (ts) of states
            c, the ts of consumption
            y, the ts of Output
            θ, the ts of productivity (in levels)
            g, the ts of government expenditure (in levels)
            τ, the ts of taxes 
            Primary_def, the ts of primary deficits
            R_f, the ts of risk free rate
            q, the ts of A-D prices, for every state
            R, 1/q, for every state
            SDF, A function which gives the ts of SDF for k period ahead 
                (write ss.SDF(k)) for every state
            price_bond, A function which gives the ts of the price of a bond of maturity k
                                    (write ss.price_bond(k)) 
            mat_p_bond, A function which returns a matrix which stacks 
                                    the ts of the price of every maturity bonds
                                    (write ss.mat_p_bond()) 
            returns_bond, A function which gives the ts of the return of a bond of maturity k
                                    (write ss.returns_bond(k)) 
            mat_r_bond, A function which returns a matrix which stacks 
                                    the ts of the return of every maturity bonds
                                    (write ss.mat_r_bond()) 
            =#

#= To get any of the ts of interests, just type ss.c (for consumption)
Except for SDF,price_bond,mat_p_bond,returns_bond,mat_r_bond which are functions
    So ss.price_bond returns a functions
    Hence to get the ts of a price of a bond of maturity k
    type ss.price_bond(k)=#

## Simulated data

### Returns
mat_list = vcat(1, [k for k in 2:2:20], 30, 60)

Sim_data = DataFrame()
for k in mat_list
    Sim_data[!, "$(k)"] = ss.returns_bond(k)[2:265] .- 1
end
CSV.write("Sim_data.csv", Sim_data)

### Other time series
Sim_other_series = DataFrame()
Sim_other_series[!, "g"] = ss.g[3:266]
Sim_other_series[!, "Revenue"] = ss.τ[3:266] .* ss.y[3:266]
Sim_other_series[!, "Primary_def"] = ss.Primary_def[3:266]
Sim_other_series[!, "GDP"] = ss.y[3:266]
Sim_other_series[!, "tax"] = ss.τ[3:266]

CSV.write("Sim_other_series.csv", Sim_other_series)
