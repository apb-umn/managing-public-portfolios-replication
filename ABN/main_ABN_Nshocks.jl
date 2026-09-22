
draft_path = ""

## run simulation 
include("Aux_function.jl")
params = define_param(N=20, S_θ=0, T=4, horizon=200, S_β=0.0,M_β=log(0.99), GHH="no",debt_ratio=4.)
tol_val=10^(-8)

@unpack N, T, debt_ratio = params
include("Utility.jl")
include("Ramsey_complete.jl")
#set s_0 to nearest integer to N/2


s₀=Int(ceil(N/2))
seed=1234
ss=simulation_full(s₀; seed=seed)
import Ipopt

include("approximate_target.jl")

# define market structure
size_bin = 1
size_bin = size_bin
mat_list = [[k for k in (t-1)*size_bin+2:t*size_bin+1] for t in 1:(N-1)]
weights = [ones(size_bin) / size_bin for t in 1:(N-1)]

# solve for ABN portfolio
## B_Yt is the time series for the market value of the total debt/gdp
## QB_Yt is the time series for the market value of the each security/gdp
## ω_abnt is the time series for the portfolio weights
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


# solve for the closet portfolio to ω_abn that satisfies the BEGS formula

ω_BEGS_approx, B̃_BEGS_approx, check, State=get_approximate(Bt[tt],ω_abn,ω_BEGS,π_Q,π_X(tt),ΣΣ[k],Σ_QΣ_Q[k],Σ_XΣ_X[k],β̂,mat_list,weights, State,tol_val)
cc = (π_Q * Σ_QΣ_Q[k] + π_X(tt) * Σ_XΣ_X[k]) * β̂
println("constraint error at soluition = ", ΣΣ[k]*ω_BEGS_approx[2:end]-cc)
println("constraint error at initial = ", ΣΣ[k]*ω_abn-cc)

# for plotting portfolios when securities have pooled maturities create a step functions of the portfolio weights
## For ABN portfolio
@unpack β=params
#State=unique(ss.s)[1]
#B̃_abn=ss.Angeletos_B(vcat(1,mat_list),vcat(1,weights))
B_abn=B_abnt1T[tt]
ω_abn_step = zeros(maximum(maximum(mat_list[:])))
ω_abn_step[1] = B̃_abn[1]*(ss.P(1,State))/(B_abn)
for l in 1:length(mat_list)
    for k in 2:size_bin+1
        ω_abn_step[(l-1)*size_bin+k]=(B̃_abn[l+1]/size_bin)*ss.P((l-1)*size_bin+k,State)/(B_abn)
    end
end

## For BEGS portfolio

ω_BEGS_approx_step = zeros(maximum(maximum(mat_list[:])))
ω_BEGS_approx_step[1] = B̃_BEGS_approx[1]*(ss.P(1,State))/(Bt[tt])
for l in 1:length(mat_list)
    for k in 2:size_bin+1
        ω_BEGS_approx_step[(l-1)*size_bin+k]=(B̃_BEGS_approx[l+1]/size_bin)*ss.P((l-1)*size_bin+k,State)/(B_abn)
    end
end

function plot_mat_struct(ω_abn_step,ω_BEGS_approx_step,State)
    p1 = plot([ω_BEGS_approx_step .* 100, ω_abn_step .* 100], 
    legend=:outerbottom,
    xlabel="Maturity, quarters", 
    ylabel="Portfolio share, percent",
    label=hcat(["portfolio satisfying eq (42)"], ["optimal portfolio (complete market)"]),
    lw=2.5,
    linestyle=[:solid :dash],
    palette=palette([:darkred, :darkseagreen], 2), 
    grid=false,
    alpha=0.8
)

savefig(p1, draft_path*"ABN_BEGGS_$(params.N)_states_bin_$(size_bin)_State_$(State).pdf")
display(p1)

#=         p2=plot([ω_abn[2:end],ω_target_nearest[2:end]],legend=:outerbottom,
            xlabel="Maturities",ylabel=L"\omega_{t}^{i}",label=hcat(["Optimal Portfolio (exact)"],["Target portfolio"]),
            palette=:seaborn_colorblind)
    display(p2)
    savefig(p2,"Full_debt_structure_without_risk_$(params.N)_states_freebin_$(size_bin).pdf")
=#    end

plot_mat_struct(ω_abn_step,ω_BEGS_approx_step,State)


#plot_mat_struct([1-sum(ω_abn);ω_abn], [1-sum(ω_BEGS); ω_BEGS], State)

#plot_mat_struct([1-sum(ω_abn);ω_abn], [1-sum(ω_BEGS_pinv); ω_BEGS_pinv], State)

#plot_mat_struct([1-sum(ω_abn);ω_abn],  ω_BEGS_approx, State)
