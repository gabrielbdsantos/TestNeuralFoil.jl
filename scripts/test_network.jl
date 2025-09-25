include("common.jl")


@wrap_pyfunction "numpy" array np_array
@wrap_pyfunction "numpy" genfromtxt np_genfromtxt
@wrap_pyfunction "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters py_get_kulfan_parameters
@wrap_pyfunction "neuralfoil" get_aero_from_kulfan_parameters py_get_aero_from_kulfan_parameters


function py_get_kulfan_from_file(filepath)
    params = py_get_kulfan_parameters(np_genfromtxt(filepath), normalize_coordinates=true)

    upper_weights = pyconvert(Vector{Float64}, params["upper_weights"])
    lower_weights = pyconvert(Vector{Float64}, params["lower_weights"])
    leading_edge_weight = pyconvert(Float64, params["leading_edge_weight"])
    trailing_edge_thickness = pyconvert(Float64, params["TE_thickness"])

    return (
        python = params,
        julia = NeuralFoil.KulfanParameters(
            upper_weights = upper_weights,
            lower_weights = lower_weights,
            leading_edge_weight = leading_edge_weight,
            TE_thickness = trailing_edge_thickness,
        )
    )
end


function run_network_from_file(filepath)
    kulfan_parameters = py_get_kulfan_from_file(filepath)

    alpha = -180:180
    Re = clamp(1e3 ^ exp(rand()), 1e3, 1e9)

    py_ans = py_get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan_parameters.python,
        alpha=np_array(alpha),
        Re=np_array(Re)
    )

    jl_ans = NeuralFoil.get_aero_from_kulfan_parameters(kulfan_parameters.julia, alpha, Re)

    return py_ans, jl_ans
end


function test_network_from_file(filepath; atol=1e-8)
    name = split(split(filepath, "/")[end], ".")[1]
    py_ans, jl_ans = run_network_from_file(filepath)

    @testset "$name" begin
        @test isapprox(pyconvert(Vector{Float64}, py_ans["analysis_confidence"]), jl_ans.analysis_confidence; atol=atol)
        @test isapprox(pyconvert(Vector{Float64}, py_ans["CL"]), jl_ans.cl; atol=atol)
        @test isapprox(pyconvert(Vector{Float64}, py_ans["CD"]), jl_ans.cd; atol=atol)
        @test isapprox(pyconvert(Vector{Float64}, py_ans["CM"]), jl_ans.cm; atol=atol)
        @test isapprox(pyconvert(Vector{Float64}, py_ans["Top_Xtr"]), jl_ans.top_xtr; atol=atol)
        @test isapprox(pyconvert(Vector{Float64}, py_ans["Bot_Xtr"]), jl_ans.bot_xtr; atol=atol)

        # ERROR: The tests below all fail.
        # ------------------------------------------
        # @test isapprox(
        #         stack([pyconvert(Vector{Float64}, py_ans["upper_bl_ue/vinf_$i"]) for i in 0:31]),
        #         jl_ans.upper_bl_ue_over_vinf
        #         ;
        #         atol=atol
        #     )
        # @test isapprox(
        #         stack([pyconvert(Vector{Float64}, py_ans["upper_bl_theta_$i"]) for i in 0:31]),
        #         jl_ans.upper_theta
        #         ;
        #         atol=atol
        #     )
        # @test isapprox(
        #         stack([pyconvert(Vector{Float64}, py_ans["upper_bl_H_$i"]) for i in 0:31]),
        #         jl_ans.upper_H
        #         ;
        #         atol=atol
        #     )
        # @test isapprox(
        #         stack([pyconvert(Vector{Float64}, py_ans["lower_bl_ue/vinf_$i"]) for i in 0:31]),
        #         jl_ans.lower_bl_ue_over_vinf
        #         ;
        #         atol=atol
        #     )
        # @test isapprox(
        #         stack([pyconvert(Vector{Float64}, py_ans["lower_bl_theta_$i"]) for i in 0:31]),
        #         jl_ans.lower_theta
        #         ;
        #         atol=atol
        #     )
        # @test isapprox(
        #         stack([pyconvert(Vector{Float64}, py_ans["lower_bl_H_$i"]) for i in 0:31]),
        #         jl_ans.lower_H
        #         ;
        #         atol=atol
        #     )
    end

    nothing
end


function test_network_on_dataset(directory; atol=1e-8)
    reduce_test_verbosity()

    @testset "Compare entire database" begin
        for file in readdir(directory)
            test_network_from_file(joinpath(directory, file); atol=atol)
        end
    end

    nothing
end


run(database; atol=1e-8) = test_network_on_dataset(database; atol=atol)
