@unpack γ, σ, GHH = params

u(c) = c^(1 - σ) / (1 - σ)
u_c(c) = c^(-σ)
u_cc(c) = -σ * c^(-σ - 1)

v(l) = l^(1 + 1 / γ) / (1 + 1 / γ)
v_l(l) = l^(1 / γ)
v_ll(l) = 1 / γ * l^(1 / γ - 1)


if GHH == "yes"
    U(c, y, θ) = u(c - v(y / θ))
    U_c(c, y, θ) = u_c(c - v(y / θ))
    U_cc(c, y, θ) = u_cc(c - v(y / θ))
    U_y(c, y, θ) = -u_c(c - v(y / θ)) / θ * v_l(y / θ)
    U_yy(c, y, θ) = (u_cc(c - v(y / θ)) * (v_l(y / θ))^2 - u_c(c - v(y / θ)) * v_ll(y / θ)) / θ^2
    U_yc(c, y, θ) = -u_cc(c - v(y / θ)) / θ * v_l(y / θ)
else
    U(c, y, θ) = u(c) - v(y / θ)
    U_c(c, y, θ) = u_c(c)
    U_cc(c, y, θ) = u_cc(c)
    U_y(c, y, θ) = -1 / θ * v_l(y / θ)
    U_yy(c, y, θ) = -v_ll(y / θ) / θ^2
    U_yc(c, y, θ) = 0.0
end