# Understanding  Angeletos-Buerea-Nicolini (ABN) model
## data
#= 0.01	0.1	0.17	0.26	0.31	0.4	0.45	0.5	0.56	0.62	0.75	
0.0918	0.4928	1.1282	1.815	2.8235	3.6471	4.4247	5.3525	6.09	7.7846	10.2632	3.992127273
6	6	6	6	6	6	6	6	6	6	60	120
0.05	0.05	0.05	0.05	0.05	0.05	0.05	0.05	0.05	0.05	0.5	1
0.00459	0.02464	0.05641	0.09075	0.141175	0.182355	0.221235	0.267625	0.3045	0.38923	5.1316	6.81411
											2.610385029
0.0005	0.005	0.0085	0.013	0.0155	0.02	0.0225	0.025	0.028	0.031	0.375	0.544
											0.079834344 =#

include("Aux_function.jl")
params = define_param(N=2, S_θ=0, T=5, horizon=2000, S_β=0.0,M_β=log(0.987), GHH="no")
#params = define_param(N=2, S_θ=0, T=5, horizon=2000, S_β=0.01/2.,M_β=log(0.99), GHH="no")
#params = define_param(N=2, S_θ=0, T=5, horizon=2000, S_β=0.01/1.69,M_β=log(0.99), GHH="no")
#include("Aux_function_onlybeta.jl")
#params = define_param(N=2, S_θ=0, T=5, horizon=2000, S_β=0.01/1.69,M_β=log(0.99), S_g=0, GHH="no")

@unpack N, T, debt_ratio = params

include("Utility.jl")
include("Ramsey_complete.jl")
s₀=Int(ceil(N/2))
seed=1234
ss=simulation_full(s₀; seed=seed)
include("approximate_target.jl")
size_bin = 1
size_bin = size_bin
mat_list = [[k for k in (t-1)*size_bin+2:t*size_bin+1] for t in 1:(N-1)]
weights = [ones(size_bin) / size_bin for t in 1:(N-1)]
B_Yt,QB_Yt1T, ω_abnt1T,B̃_abn,B_abnt1T = ss.market_value_B(mat_list, weights)
Bt=B_Yt.*ss.y[2:end]
B_Y_BEGSt0T, B̃_BEGSt1T, ω_BEGSt1T, π_Q, π_X, ΣΣ, Σ_QΣ_Q, Σ_XΣ_X, β̂,Σ_PDΣ_PD,ω_BEGS_pinvt1T = ss.debt_path(mat_list, weights)
States_realized=unique(ss.s)
tt=2
State = ss.s[tt]
# find the index where State is equal to the realized state
k = findfirst(isequal(State), ss.s)

ω_abn = ω_abnt1T[tt, 2:end]
ω_BEGS = ω_BEGSt1T[tt, 2:end]
ω_BEGS_pinv = ω_BEGS_pinvt1T[tt, 2:end]


T_consol = 2000
mat_list = [[k for k in 2:T_consol]]
weights = [ones(T_consol - 1)]
size_bin = T_consol - 1

B_Yt,QB_Yt1T, ω_abnt1T,B̃_abn,B_abnt1T = ss.market_value_B(mat_list, weights)
Bt=B_Yt.*ss.y[2:end]
B_Y_BEGSt0T, B̃_BEGSt1T, ω_BEGSt1T, π_Q, π_X, ΣΣ, Σ_QΣ_Q, Σ_XΣ_X, β̂,Σ_PDΣ_PD,ω_BEGS_pinvt1T = ss.debt_path(mat_list, weights)
States_realized=unique(ss.s)
tt=2
State = ss.s[tt]
# find the index where State is equal to the realized state
k = findfirst(isequal(State), ss.s)

ω_consol_abn = ω_abnt1T[tt, 2:end]
ω_consol_BEGS = ω_BEGSt1T[tt, 2:end]
ω_consol_BEGS_pinv = ω_BEGS_pinvt1T[tt, 2:end]

Matrices = DataFrame()
Matrices[!, :Σ] = [ΣΣ[1][1]]
Matrices[!, Symbol("std(r)")] = [sqrt(ΣΣ[1][1])]
Matrices[!, Symbol("Σ_Q*β")] = Σ_QΣ_Q[k] * β̂
Matrices[!, Symbol("Σ_X*β")] = Σ_XΣ_X[k] * β̂
Matrices[!, Symbol("π_Q")] = [π_Q]
Matrices[!, Symbol("π_X")] = [π_X(tt)]
Matrices[!, Symbol("π_Q*Σ_Q*β")] = π_Q * Σ_QΣ_Q[k] * β̂
Matrices[!, Symbol("π_X*Σ_X*β")] = [π_X(tt)] * Σ_XΣ_X[k] * β̂
Matrices[!, Symbol("(1/Σ)")] = inv(ΣΣ[k])[:]
Matrices[!, Symbol("(1/Σ)*π_Q*Σ_Q*β")] = inv(ΣΣ[k]) * π_Q * Σ_QΣ_Q[k] * β̂
Matrices[!, Symbol("(1/Σ)*π_X*Σ_X*β")] = inv(ΣΣ[k]) * [π_X(tt)] * Σ_XΣ_X[k] * β̂
Matrices[!, Symbol("ω_console")] = [ω_BEGSt1T[tt, 2]]
Matrices[!, Symbol("N_states")] = [params.N]
Matrices[!, Symbol("State")] = [State]
Matrices[!, Symbol("ω_abn")] = [ω_abnt1T[tt, 2]]
Matrices[!, Symbol("cov(surplus,r)")] = [-Σ_PDΣ_PD[1][1]]
Matrices[!, Symbol("cov(surplus,r)/var(r)")] = [-Σ_PDΣ_PD[1][1]/ΣΣ[1][1]]
Matrices[!, Symbol("cov(surplus,r)/var(r) data")] = [-0.079]
Matrices[!, Symbol("std(r) data")] = [0.026]
Matrices = round.(Matrices; sigdigits=2)



# define a grid on S_β
S_β_grid = [0,0.0063,0.0075]
# find omega,omega_pass for each S_β
omega_abn = zeros(length(S_β_grid))
omega_formula = zeros(length(S_β_grid))
omega_formula_Q = zeros(length(S_β_grid))
omega_formula_X = zeros(length(S_β_grid))
std_rr= zeros(length(S_β_grid))
cov_rx_covr= zeros(length(S_β_grid))
std_rr_data=ones(length(S_β_grid))*0.0353
cov_rx_covr_data=-ones(length(S_β_grid))*0.0614
cov_rx=zeros(length(S_β_grid))
cov_rx_data=-ones(length(S_β_grid))*0.76

for (i, S_β) in enumerate(S_β_grid)
    println("S_β = $S_β")
    params = define_param(N=2, S_θ=0, T=5, horizon=2000, S_β=S_β, M_β=log(0.99) ,GHH="no")
    include("Ramsey_complete.jl")
    s₀=Int(ceil(N/2))
    seed=1234
    ss=simulation_full(s₀; seed=seed)
    mat_list = [[k for k in 2:T_consol]]
    weights = [ones(T_consol - 1)]

    B_Yt,QB_Yt1T, ω_abnt1T,B̃_abn,B_abnt1T = ss.market_value_B(mat_list, weights)
    Bt=B_Yt.*ss.y[2:end]
    B_Y_BEGSt0T, B̃_BEGSt1T, ω_BEGSt1T, π_Q, π_X, ΣΣ, Σ_QΣ_Q, Σ_XΣ_X, β̂,Σ_PDΣ_PD,ω_BEGS_pinvt1T = ss.debt_path(mat_list, weights)
    States_realized=unique(ss.s)
    tt=2
    State = ss.s[tt]
    # find the index where State is equal to the realized state
    k = findfirst(isequal(State), ss.s)
    omega_abn[i] = ω_abnt1T[tt, 2]
    omega_formula_Q[i]=(inv(ΣΣ[k]) * π_Q * Σ_QΣ_Q[k] * β̂)[1]
    omega_formula_X[i]=(inv(ΣΣ[k]) * π_X(tt) * Σ_XΣ_X[k] * β̂)[1]
    omega_formula[i] = ω_BEGSt1T[tt, 2]
    std_rr[i] = sqrt(ΣΣ[1][1])
    cov_rx_covr[i] = -Σ_PDΣ_PD[1][1]/ΣΣ[1][1]
    cov_rx[i] = -Σ_PDΣ_PD[1][1]
end

# make a table with the results
ABN_table = DataFrame()
ABN_table[!, :VarBeta] = S_β_grid
ABN_table[!, :ConsolShareABN] = omega_abn
ABN_table[!, :ConsolShareFormula] = omega_formula
ABN_table[!, :ConsolShareFormulaQ] = omega_formula_Q
ABN_table[!, :ConsolShareFormulaX] = omega_formula_X
ABN_table[!, :StdR] = std_rr
ABN_table[!, :CovXR] = cov_rx
ABN_table[!, :CovXRData] = cov_rx_data
ABN_table[!, :RatioCovXRCovRR] = cov_rx_covr
ABN_table[!, :StdRData] = std_rr_data
ABN_table[!, :RatioCovXRCovRRData] = cov_rx_covr_data
ABN_table = round.(ABN_table; sigdigits=3)

# flip the correlation
S_β=0.0075
params = define_param_flipped(N=2, S_θ=0, T=5, horizon=2000, S_β=S_β, M_β=log(0.9870687452490858) ,GHH="no")

include("Ramsey_complete.jl")
s₀=Int(ceil(N/2))
seed=1234
ss=simulation_full(s₀; seed=seed)
mat_list = [[k for k in 2:T_consol]]
weights = [ones(T_consol - 1)]
B_Yt,QB_Yt1T, ω_abnt1T,B̃_abn,B_abnt1T = ss.market_value_B(mat_list, weights)
Bt=B_Yt.*ss.y[2:end]
B_Y_BEGSt0T, B̃_BEGSt1T, ω_BEGSt1T, π_Q, π_X, ΣΣ, Σ_QΣ_Q, Σ_XΣ_X, β̂,Σ_PDΣ_PD,ω_BEGS_pinvt1T = ss.debt_path(mat_list, weights)
States_realized=unique(ss.s)
tt=2
State = ss.s[tt]
# find the index where State is equal to the realized state
k = findfirst(isequal(State), ss.s)
omega_abn_flipped = ω_abnt1T[tt, 2]
omega_formula_Q_flipped=(inv(ΣΣ[k]) * π_Q * Σ_QΣ_Q[k] * β̂)[1]
omega_formula_X_flipped=(inv(ΣΣ[k]) * π_X(tt) * Σ_XΣ_X[k] * β̂)[1]
omega_formula_flipped = ω_BEGSt1T[tt, 2]
std_rr_flipped = sqrt(ΣΣ[1][1])
cov_rx_covr_flipped = -Σ_PDΣ_PD[1][1]/ΣΣ[1][1]
cov_rx_flipped = -Σ_PDΣ_PD[1][1]
omega_abn_flipped
omega_formula_Q_flipped
omega_formula_X_flipped
omega_formula_flipped

println(cov_rx_covr_flipped)
println(std_rr_flipped)