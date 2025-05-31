using DFOTools
using Test

@testset "structs" begin
    bb1 = Blackbox(2, x -> x[1])
    @test nb_ineqs(bb1) == 0
    @test nb_eqs(bb1) == 0
    add_ineq!(bb1, x -> x[1])
    @test nb_ineqs(bb1) == 1
    add_eq!(bb1, x -> x[1])
    @test nb_eqs(bb1) == 1
    x = [0.0]
    @test eval_ineq(bb1, 1, x) == 0.0
    @test eval_eq(bb1, 1, x) == 0.0
    @test satisfies_ineq(bb1, 1, x)
    @test satisfies_eq(bb1, 1, x)
end
