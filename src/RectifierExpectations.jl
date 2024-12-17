module RectifierExpectations

    using StatsFuns

    Φ(x) = StatsFuns.normcdf(x)

    ϕ(x) = StatsFuns.normpdf(x)

    include("rectifier_expectations.jl")

    export M, V, Vslow, B, M_V, M_V_B

end
