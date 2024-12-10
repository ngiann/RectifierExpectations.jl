module RectifierExpectations

    using StatsFuns

    Φ = StatsFuns.normcdf

    ϕ = StatsFuns.normpdf

    include("rectifier_expectations.jl")

    export M, V, Vslow, B, M_V, M_V_B

end
