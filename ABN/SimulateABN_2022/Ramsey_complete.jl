function FB()
    @unpack S = params
    n_S = size(S)[1]
    θ = exp.(S[:, 2])
    g = exp.(S[:, 1])
    x_0 = vcat(θ, θ)
    function obj(x)
        c = x[1:n_S]
        y = x[n_S+1:2*n_S]
        objective = zeros(n_S, 2)
        objective[:, 1] = U_c.(c, y, θ) + U_y.(c, y, θ)
        objective[:, 2] = y .- c .- g
        return objective[:]
    end
    results = nlsolve(obj, x_0)
    if results.f_converged == false
        error("Didn't converge FB")
    end
    c = results.zero[1:n_S]
    y = results.zero[n_S+1:2*n_S]
    z = U_c.(c, y, θ) .* c + U_y.(c, y, θ) .* y
    return (c=c, y=y, z=z)
end


function time1(Φ)
    @unpack S = params
    n_S = size(S)[1]
    θ = exp.(S[:, 2])
    g = exp.(S[:, 1])
    x_0 = vcat(FB().c, FB().y)
    function obj(x)
        c = x[1:n_S]
        y = x[n_S+1:2*n_S]
        objective = zeros(n_S, 2)
        objective[:, 1] = (1 + Φ) .* (U_c.(c, y, θ) + U_y.(c, y, θ)) .+
                          Φ .* ((U_cc.(c, y, θ) + U_yc.(c, y, θ)) .* c + (U_yy.(c, y, θ) + U_yc.(c, y, θ)) .* y)
        objective[:, 2] = y .- c .- g
        return objective[:]
    end
    results = nlsolve(obj, x_0)
    if results.f_converged == false
        error("Didn't converge time1")
    end
    c = results.zero[1:n_S]
    y = results.zero[n_S+1:2*n_S]
    z = U_c.(c, y, θ) .* c + U_y.(c, y, θ) .* y
    return (c=c, y=y, z=z)
end

function time0(Φ, s₀; debt_ratio=params.debt_ratio)
    @unpack S = params
    n_S = size(S)[1]
    θ = exp.(S[s₀, 2])
    g = exp.(S[s₀, 1])
    x_0 = vcat(FB().c[s₀], FB().y[s₀])
    function obj(x)
        c = x[1]
        y = x[2]
        b₀ = y * debt_ratio
        objective = zeros(2)
        objective[1] = (1 + Φ) * (U_c(c, y, θ) + U_y(c, y, θ)) +
                       Φ * ((U_cc(c, y, θ) + U_yc(c, y, θ)) * c + (U_yy(c, y, θ) + U_yc(c, y, θ)) * y)
        objective[1] = objective[1] - Φ * (U_cc(c, y, θ) + U_yc(c, y, θ)) * b₀
        objective[2] = y - c - g
        return objective
    end
    results = nlsolve(obj, x_0)
    if results.f_converged == false
        error("Didn't converge time0")
    end
    c = results.zero[1]
    y = results.zero[2]
    z = U_c(c, y, θ) * c + U_y(c, y, θ) * y
    return (c=c, y=y, z=z)
end

function price_gap(Φ; s₀)
    @unpack Π, β, S, debt_ratio = params
    θ = exp.(S[s₀, 2])
    z(Φ) = time1(Φ).z
    z₀(Φ) = time0(Φ, s₀).z
    y₀(Φ) = time0(Φ, s₀).y
    c₀(Φ) = time0(Φ, s₀).c
    b₀(Φ) = debt_ratio * y₀(Φ)
    f(Φ) = U_c(c₀(Φ), y₀(Φ), θ) * b₀(Φ) - z₀(Φ) - (β * Π[s₀, :]' / (I - β * Π)) * z(Φ)
    return (f(Φ[1]))
end


function find_fixed_point(s; Φ₀=0.0)
    Φ_new = Φ₀
    obj(Φ) = price_gap(Φ; s₀=s)
    results = find_zero(obj, Φ₀)
    return (Φ = results)
end


function simulation_full(s₀; seed)
    Random.seed!(seed)
    @unpack Π, β, S, γ, T = params
    n_S = size(S)[1]

    Φ = find_fixed_point(s₀)

    s = ones(T + 1)
    s[1] = s₀

    c = similar(s)
    y = similar(s)
    alloc0 = time0(Φ, s₀)

    c[1] = alloc0.c
    y[1] = alloc0.y

    items = [k for k in 1:length(Π[1, :])]

    alloc1 = time1(Φ)
    c1 = alloc1.c
    y1 = alloc1.y
    θ1 = exp.(S[:, 2])
    g1 = exp.(S[:, 1])
    τ1 = 1.0 .+ U_y.(c1, y1, θ1) ./ U_c.(c1, y1, θ1)
    Primary_def1 = g1 .- τ1 .* y1

    for t in 2:T+1
        wv = pweights(Π[Int(s[t-1]), :])
        s[t] = Int(sample(items, wv))
    end
    s = Int.(s)
    s_1 = s[2:T+1]
    s_0 = s[1:T]

    θ = exp.(S[s, 2])
    g = exp.(S[s, 1])
    c[2:T+1] = c1[s_1]
    y[2:T+1] = y1[s_1]
    τ = 1.0 .+ U_y.(c, y, θ) ./ U_c.(c, y, θ)
    Primary_def = g .- τ .* y


    q = zeros(T + 1, n_S)
    sdf = zeros(T + 1, n_S)
    for t in 1:T+1
        q[t, :] = β * U_c.(c1, y1, θ1) .* Π[s[t], :] ./ U_c.(c[t], y[t], θ[t])
        sdf[t, :] = U_c.(c1, y1, θ1) ./ U_c.(c[t], y[t], θ[t])
    end
    R = 1.0 ./ q

    function SDF(k)
        if k == 0
            SS = zeros(T + 1, n_S)
            for t in 1:T+1
                SS[t, s[t]] = 1.0
            end
            return SS
        end

        return β^(k) .* sdf[1:T+1-k, :]
    end

    function P(K, ss)
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        p(K, ss) = sum(β^(K) * sdf_func(ss) .* (Π^K)[ss, :])
        return p(K, ss)
    end

    function P_consol(ss)
        uc_p_consol = (I - β * Π) \ U_c.(c1, y1, θ1)
        p_consol = uc_p_consol ./ U_c.(c1, y1, θ1)
        return p_consol(ss)
    end

    function r_consol_next_vec(ss, ss′)
    end


    function price_bond(k; s=s)
        s_k = s[1:T+1-k]
        p = fill(NaN, T + 1)
        p[1:T+1-k] = sum(SDF(k) .* (Π^k)[s_k, :], dims=2)[:]
        p = p[:]
        return p
    end

    function mat_p_bond()
        pbond = similar(fill(NaN, T + 1, T + 1))
        for k in 1:T+1
            pbond[:, k] = price_bond(k - 1)
        end
        return pbond
    end

    function returns_bond(k)
        RR = zeros(T)
        for t in 2:T+1
            RR[t-1] = (price_bond(k - 1))[t] / (price_bond(k))[t-1]
        end
        return RR
    end

    R_f = returns_bond(1)

    function mat_r_bond()
        rbond = similar(fill(NaN, T, T))
        for k in 1:T
            rbond[:, k] = returns_bond(k)
        end
        return rbond
    end

    #=     function r_next(K,s_T)
            @unpack S=params
            n_S=size(S)[1]
            sdf_func(ss) = U_c.(c1,y1,θ1)./U_c.(c1[ss],y1[ss],θ1[ss])
            p(K,ss) =  sum(β^(K)*sdf_func(ss).*(Π^K)[ss,:])
            p(K) = [p(K,ss) for ss in 1:n_S]
            function RR(K)
                    return  p(K-1)./p(K,s_T)
            end
            r(K) = RR(K) .- RR(1)
            return r(K)
        end
     =#
    function r_next_vec(mat_list, weights, s_T)
        @unpack S, β = params
        n_S = size(S)[1]
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        p(K, ss) = sum(β^(K) * sdf_func(ss) .* (Π^K)[ss, :])

        function sum_return_mat_list(mat_list, weights, s, lag)
            Sum = 0
            for k in 1:length(mat_list)
                Sum = Sum + dot(p.(mat_list[k] .- lag, s), weights[k]) / sum(weights[k])
            end
            return Sum
        end
        function RR(mat_list, weights)
            R = [sum_return_mat_list(mat_list, weights, s, 1) for s in 1:n_S]
            return R ./ sum_return_mat_list(mat_list, weights, s_T, 0)
        end
        r(mat_list, weights) = RR(mat_list, weights) .- RR([1], [1])
        return r(mat_list, weights)
    end

    function Σ_PD(s_T, mat_list, weights)
        @unpack horizon = params
        N_max = length(mat_list)
        mat = zeros(N_max, horizon)

        @unpack Π = params
        val, vec = eigen(Π')
        π_stat = real.(abs.(vec[:, end]) ./ sum(abs.(vec[:, end])))

        Threads.@threads for j in 1:N_max
            Y = r_next_vec(mat_list[j], weights[j], s_T)
            for tt in 1:horizon
                Π_tt = Π^(tt)
                X = Primary_def1 / (sum(y1 .* π_stat))
                E_X_tt = [sum(X .* (Π^(tt-1))[ss, :]) for ss in 1:n_S]
                mat[j, tt] = -sum(Y .* E_X_tt .* Π[s_T, :]) + sum(E_X_tt .* Π[s_T, :]) * sum(Y .* Π[s_T, :])
            end
        end
        return mat
    end



    function Σ_X(s_T, mat_list, weights)
        @unpack horizon = params
        N_max = length(mat_list)
        mat = zeros(N_max, horizon)

        function X_orth(tt)
            ξ = 1 .- γ .* τ1 ./ (1 .- τ1)
            Π_tt = Π^(tt)
            X = -Primary_def1 .- τ1 .* sum(ξ .* Π_tt[s_T, :]) * sum(y1 .* Π_tt[s_T, :])
            return X
        end

        @unpack Π = params
        val, vec = eigen(Π')
        π_stat = real.(abs.(vec[:, end]) ./ sum(abs.(vec[:, end])))

        Threads.@threads for j in 1:N_max
            Y = r_next_vec(mat_list[j], weights[j], s_T)
            for tt in 1:horizon
                Π_tt = Π^(tt)
                X = X_orth(tt) ./ (sum(y1 .* π_stat))
                E_X_tt = [sum(X .* (Π^(tt-1))[ss, :]) for ss in 1:n_S]
                mat[j, tt] = sum(Y .* E_X_tt .* Π[s_T, :]) - sum(E_X_tt .* Π[s_T, :]) * sum(Y .* Π[s_T, :])
            end
        end
        return mat
    end


    function Σ(s_T, mat_list, weights)
        @unpack horizon = params
        N_max = length(mat_list)
        mat = zeros((N_max, N_max))

        for j in 1:N_max
            X = r_next_vec(mat_list[j], weights[j], s_T)
            Threads.@threads for i in j:N_max
                Y = r_next_vec(mat_list[i], weights[i], s_T)
                mat[j, i] = sum(X .* Y .* Π[s_T, :]) - sum(X .* Π[s_T, :]) * sum(Y .* Π[s_T, :])
                mat[i, j] = sum(X .* Y .* Π[s_T, :]) - sum(X .* Π[s_T, :]) * sum(Y .* Π[s_T, :])
            end
        end
        return mat
    end

    function Σ_Q(s_T, mat_list, weights)
        @unpack horizon = params
        N_max = length(mat_list)

        @unpack S = params
        n_S = size(S)[1]
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])

        mat = zeros(N_max, horizon)

        Threads.@threads for j in 1:N_max
            R = r_next_vec(mat_list[j], weights[j], s_T)
            for tt in 1:horizon
                Q = log.([dot(β^(tt) * sdf_func(ss), (Π^(tt))[ss, :]) for ss in 1:n_S])
                mat[j, tt] = sum(R .* Q .* Π[s_T, :]) - sum(R .* Π[s_T, :]) * sum(Q .* Π[s_T, :])
            end
        end
        return mat
    end

    function Angeletos_B(mat, weights)
        @unpack S, T, β, Π = params
        n_S = size(S)[1]
        N = length(mat)
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        p(K, ss) = dot(β^(K) * sdf_func(ss), (Π^K)[ss, :])


        ##Prices of debt components
        ## q[s,n] = next period price of bond of the nth maturity, if the state s_t+1=s

        Q = zeros(n_S, N)
        Threads.@threads for n in 1:N
            for s in 1:n_S
                Q[s, n] = sum(p.(mat[n] .- 1, s) .* weights[n] ./ sum(weights[n]))
            end
        end

        ##Present-value of surpluses tomorrow
        ## V[s] =Present-value of surpluses in s_t+1=s
        V = zeros(n_S)

        Threads.@threads for s in 1:n_S
            V[s] = -Primary_def1[s] - dot(β * (Π/(I-β*Π))[s, :], sdf_func(s) .* Primary_def1)
        end

        #B = quantities of each bond
        B = pinv(Q)*V
        return B
    end

    function Angeletos__market_value(mat, weights)
        @unpack S, T, β, Π = params
        n_S = size(S)[1]
        N = length(mat)
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        p(K, ss) = dot(β^(K) * sdf_func(ss), (Π^K)[ss, :])


        ##Prices of debt components
        ## q[s,n] = next period price of bond of the nth maturity, if the state s_t+1=s

        Q = zeros(n_S, N)
        Threads.@threads for n in 1:N
            for s in 1:n_S
                # Q[s, n] = dot(p.(mat[n] .- 1, s), weights[n] ./ sum(weights[n])) / dot(p.(mat[n], s), weights[n] ./ sum(weights[n]))
                Q[s, n] = dot(p.(mat[n] .- 1, s), weights[n] ./ sum(weights[n]))
            end
        end

        ##Present-value of surpluses tomorrow
        ## V[s] =Present-value of surpluses in s_t+1=s
        V = zeros(n_S)

        Threads.@threads for s in 1:n_S
            V[s] = -Primary_def1[s] + dot(β * (Π/(I-β*Π))[s, :], sdf_func(s) .* -Primary_def1)
        end

        #B = quantities of each bond
        B = pinv(Q)* V
        return B
    end

    function market_value_B(mat, weights)
        @unpack T = params
        mat = vcat([1], mat)
        weights = vcat([1], weights)
        B = Angeletos_B(mat, weights)
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        p(K, ss) = dot(β^(K) * sdf_func(ss), (Π^K)[ss, :])
        portfolio_weights = zeros(T, length(mat))
        market_value = zeros(T)
        Threads.@threads for k in 1:length(mat)
            Price = zeros(T)
            for t in 1:T
                Price[t] = sum(p.(mat[k], s[t+1]) .* weights[k] ./ sum(weights[k]))
            end
            portfolio_weights[:, k] = B[k] .* Price
        end
        market_value = sum(portfolio_weights, dims=2)[:]
        return (market_value ./ y[2:T+1], portfolio_weights ./ y[2:T+1], portfolio_weights ./ market_value)
    end

    function Γ(T)
        n_S = size(S)[1]
        EY = [sum(y1 .* Π[s, :]) for s in 1:n_S]
        𝚪 = dot(EY ./ y1, Π[s[T], :])
        return 𝚪
    end
    function Eq(T)
        n_S = size(S)[1]
        sdf = [β * dot(U_c.(c1, y1, θ1), Π[s, :]) for s in 1:n_S] ./ U_c.(c1, y1, θ1)
        return dot(sdf, Π[s[T], :])
    end

    function debt_path(mat, weights)
        @unpack T, debt_ratio, S, horizon = params
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        p(K, ss) = dot(β^(K) * sdf_func(ss), (Π^K)[ss, :])

        full_mat = vcat([1], mat)
        full_weights = vcat([1], weights)
        W = zeros(T, length(full_mat))
        B = zeros(T + 1)

        BB, PP, SS = market_value_B(mat, weights)
        BB = BB[:]
        B[2:end] = BB .* y[2:T+1]
        B[1] = debt_ratio * y[1]


        market = zeros(T, length(mat) + 1)
        Gamma = Γ(2)
        q = Eq(2)
        π_Q = (1 - q * Gamma) / q

        ζ(t) = (1 - (1 - γ) * τ[t])^(2) / γ
        π_A(t) = ζ(t) * y[t] / B[t]

        π_X(t) = y[t] / (q * B[t])
        ΣΣ = [Σ(ss, mat, weights) for ss in unique(s)]
        Σ_QΣ_Q = [Σ_Q(ss, mat, weights) for ss in unique(s)]
        Σ_XΣ_X = [Σ_X(ss, mat, weights) for ss in unique(s)]
        w = [(q * Gamma)^(h) for h in 1:horizon]
        pp = [[sum(p.(full_mat[n], ss) .* full_weights[n] ./ sum(full_weights[n])) for n in 1:length(full_mat)] for ss in unique(s)]
        Threads.@threads for t in 1:T
            ss = findfirst(isequal(s[t]), unique(s))
            market[t, 2:end] = (ΣΣ[ss]) \ ((π_Q * Σ_QΣ_Q[ss] + π_X(t) * Σ_XΣ_X[ss]) * w)
            market[t, 1] = 1 - sum(market[t, 2:end])

            W[t, :] = market[t, :] * B[t] ./ pp[ss]
        end
        return (B ./ y, W, market, π_Q, π_X, ΣΣ, Σ_QΣ_Q, Σ_XΣ_X, w)
    end

    function Excess_Returns_ts(mat, weights)
        @unpack S, T, β, Π = params
        n_S = size(S)[1]
        N = length(mat)
        sdf_func(ss) = U_c.(c1, y1, θ1) ./ U_c.(c1[ss], y1[ss], θ1[ss])
        sdf_func_0() = U_c.(c1, y1, θ1) ./ U_c.(c[1], y[1], θ[1])
        p(K, ss) = dot(β^(K) * sdf_func(ss), (Π^K)[ss, :])
        p_0(K, ss) = dot(β^(K) * sdf_func_0(), (Π^K)[ss, :])
        R = zeros(T, length(mat))
        for n in 1:length(mat)
            R[2, n] = dot(p.(mat[n] .- 1, s[2]), weights[n] ./ sum(weights[n])) / dot(p_0.(mat[n], s[1]), weights[n] ./ sum(weights[n]))
        end
        Threads.@threads for t in 3:T+1
            for n in 1:length(mat)
                R[t-1, n] = dot(p.(mat[n] .- 1, s[t]), weights[n] ./ sum(weights[n])) / dot(p.(mat[n], s[t-1]), weights[n] ./ sum(weights[n]))
            end
        end
        return (R .- R_f)
    end

    return (s=s, c=c, y=y,
        θ=θ, g=g,
        τ=τ,
        Primary_def=Primary_def,
        R_f=R_f,
        q=q,
        R=R,
        SDF=SDF,
        debt_path=debt_path,
        price_bond=price_bond,
        mat_p_bond=mat_p_bond,
        returns_bond=returns_bond,
        mat_r_bond=mat_r_bond,
        Σ_X=Σ_X,
        Σ=Σ,
        Σ_PD=Σ_PD,
        Σ_Q=Σ_Q,
        Angeletos_B=Angeletos_B,
        market_value_B=market_value_B,
        Γ=Γ,
        Eq=Eq,
        Angeletos__market_value=Angeletos__market_value,
        P=P,
        Excess_Returns_ts=Excess_Returns_ts)
end
#= 
function simulation_time_1(s₀)
@unpack s,c,y,R_f,τ,θ,q,Primary_def,R,price_bond,mat_p_bond,returns_bonds,mat_r_bond=simulation_full(s₀)
return (s=s[2:T+1],c=c[2:T+1],y=y[2:T+1],R_f=R_f,τ=τ[2:T+1],θ=θ[2:T+1],q=q,R=R,
        Primary_def=Primary_def[2:T+1])
end
=#

function simulation_Ramsey(; seed=1234)
    Random.seed!(seed)
    @unpack Π = params
    val, vec = eigen(Π')
    π_stat = real.(abs.(vec[:, end]) ./ sum(abs.(vec[:, end])))
    items = [k for k in 1:length(Π[1, :])]

    wv = pweights(π_stat)
    s₀ = Int(sample(items, wv))
    return (simulation_full(s₀; seed=seed))
end

function get_csv(d; k=-1)
    ss = simulation_Ramsey()
    if k >= 0
        vec = ss[Symbol(d)](k)
        df = DataFrame(t=[k for k in 1:length(vec)])
        name = string(d, "_", k)
        df[!, Symbol(name)] = vec
        CSV.write("$(name).csv", df)
    else
        vec = ss[Symbol(d)]
        df = DataFrame(t=[k for k in 1:length(vec)])
        df[!, Symbol(d)] = vec
        CSV.write("$(d).csv", df)
    end
end


