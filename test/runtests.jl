using RectifierExpectations
using Test
using QuadGK
using Distributions
using Printf
using ThreadsX

@testset "RectifierExpectations.jl" begin
    
    TOL = 1e-5

    LIMIT = 80.0

    REPEATS = 10

    dx = 1e-5

    finegrid = -LIMIT:dx:LIMIT


    r(x) = x >= 0 ? x : zero(eltype(x)) 
    
    let

        #--------------------------------------------------
        # Test expectation of plain rectifier
        #--------------------------------------------------

        @printf("\n Testing expectation of rectifier\n")

        for _ in 1:REPEATS

            μ, σ = randn()*3, 0.1 + rand()*1.2

            d = Normal(μ,σ)

            numresult = ThreadsX.mapreduce(x -> pdf(d,x)*r(x), +, finegrid) * dx

            @test abs(numresult - M(μ, σ)) < TOL

        end

        #--------------------------------------------------
        # Test expectation of scaled and shifted rectifier
        #--------------------------------------------------

        @printf("\n Testing expectation of scaled and shifted rectifier\n")
        
        for _ in 1:REPEATS

            a, b, μ, σ = 1*rand()+0.01, randn()*3, randn()*3, 0.1 + rand()*1.2

            d = Normal(μ,σ)

            numresult = ThreadsX.mapreduce(x -> pdf(d,x)*r(a*x+b), +, finegrid) * dx

            @test abs(numresult - M(a, b, μ, σ)) < TOL

        end


        #--------------------------------------------------
        # Test variance of scaled and shifted rectifier
        #--------------------------------------------------

        @printf("\n Testing variance of scaled and shifted rectifier\n")

        for _ in 1:REPEATS

            a, b, μ, σ = 3*rand()+0.01, randn()*3, randn()*3, 0.1 + rand()*1.2

            d = Normal(μ,σ)

            numresult = ThreadsX.mapreduce(x -> pdf(d,x)*(r(a*x+b) - M(a,b,μ,σ))^2, +, finegrid) * dx

            @test abs(numresult - V(a, b, μ, σ)) < TOL

        end


        #--------------------------------------------------
        # Test expectation of squared rectifier
        #--------------------------------------------------

        @printf("\n Testing expectation of scaled and shifted rectifier squared\n")

        for _ in 1:REPEATS

            a, b, μ, σ = 3*rand()+0.01, randn()*3, randn()*3, 0.1 + rand()*1.2

            d = Normal(μ,σ)

            numresult = ThreadsX.mapreduce(x -> pdf(d,x)*(r(a*x+b))^2, +, finegrid) * dx

            @test abs(numresult - B(a, b, μ, σ)) < TOL

        end


    end


    @printf("\n consistency check between V and Vslow\n")

    # Check consistency of V and Vslow
    let
        
        for _ in 1:REPEATS

            μ, σ = randn()*3, 0.1 + rand()*1.2

            @test abs(Vslow(μ, σ) - V(μ, σ)) < 1e-9
            
        end

    end
    
    @printf("\n consistency check between M_V and M, V\n")

    # Check consistency between functions
    let
        # Check consistency between M_V and M, V
        for _ in 1:REPEATS

            a, b, μ, σ = 3*rand()+0.01, randn()*3, randn()*3, 0.1 + rand()*1.2

            M1, V1 = M_V(a, b, μ, σ)

            @test abs(M1-M(a, b, μ, σ)) < 1e-9

            @test abs(V1-V(a, b, μ, σ)) < 1e-9
            
        end

        @printf("\n consistency check between M_V_B and M, V, B and M_V\n")

        # Check consistency between M_V_B and M, V, B and M_V
        for _ in 1:REPEATS

            a, b, μ, σ = 3*rand()+0.01, randn()*3, randn()*3, 0.1 + rand()*1.2

            M1, V1 = M_V(a, b, μ, σ)

            M2, V2, B2 = M_V_B(a, b, μ, σ)

            @test abs(M1-M(a, b, μ, σ)) < 1e-9
            @test abs(V1-V(a, b, μ, σ)) < 1e-9

            @test abs(M2-M(a, b, μ, σ)) < 1e-9
            @test abs(V2-V(a, b, μ, σ)) < 1e-9
            @test abs(B2-B(a, b, μ, σ)) < 1e-9

            @test abs(M1-M2) < 1e-9
            @test abs(V1-V2) < 1e-9

        end
    end
            
end
