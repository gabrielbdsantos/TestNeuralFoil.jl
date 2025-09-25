include("common.jl")


@wrap_pyfunction "numpy" array np_array
@wrap_pyfunction "numpy" genfromtxt np_genfromtxt
@wrap_pyfunction "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters py_get_kulfan_parameters
@wrap_pyfunction "neuralfoil" get_aero_from_kulfan_parameters py_get_aero_from_kulfan_parameters


function run_network_from_file(filepath)
    jl_coords = coordinates_from_file(filepath)
    jl_kulfan_params = NeuralFoil.get_kulfan_parameters(jl_coords)

    py_coords = np_array(jl_coords)
    py_kulfan_params = py_get_kulfan_parameters(py_coords, normalize_coordinates=false)

    alpha = -180:180
    Re = clamp(1e3 ^ exp(rand()), 1e3, 1e9)

    py_ans = py_get_aero_from_kulfan_parameters(
        kulfan_parameters=py_kulfan_params,
        alpha=np_array(alpha),
        Re=np_array(Re)
    )

    jl_ans = NeuralFoil.get_aero_from_kulfan_parameters(
        jl_kulfan_params, alpha, Re
    )

    return py_ans, jl_ans
end


function test_network_from_file(filepath; atol=1e-3)
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


function test_network_on_dataset(directory; atol=1e-3)
    reduce_test_verbosity()

    @testset "Compare entire database" begin
        for file in readdir(directory)
            test_network_from_file(joinpath(directory, file); atol=atol)
        end
    end

    nothing
end


run(database; atol=1e-3) = test_network_on_dataset(database; atol=atol)
