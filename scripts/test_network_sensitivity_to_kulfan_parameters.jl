ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(pwd(), ".venv/bin/python")

using Test
using PythonCall
using NeuralFoil


# Supress the stacktrace. Better to keep this for the sake of simplicity in case many tests fail.
Test.eval(quote
	function record(ts::DefaultTestSet, t::Union{Fail, Error})
		push!(ts.results, t)
	end
end)


macro wrap_pyfunction(mod, fname, jname)
    quote
        const pymod = pyimport($mod)

        # Define functions with the same names as the input symbols
        $(:(
            function $(esc(jname))(args...; kwargs...)
                pyf = @pyconst pymod.$(fname)
                return pyf(args...; kwargs...)
            end
        ))
    end
end


@wrap_pyfunction "numpy" array np_array
@wrap_pyfunction "numpy" genfromtxt np_genfromtxt
@wrap_pyfunction "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters py_get_kulfan_parameters
@wrap_pyfunction "neuralfoil" get_aero_from_kulfan_parameters py_get_aero_from_kulfan_parameters


function add_perturbations_to_params(params, epsilon)
    num_params = size(params.upper_weights, 1)

    return NeuralFoil.KulfanParameters(
        upper_weights = params.upper_weights .+ rand(num_params) * epsilon,
        lower_weights = params.lower_weights .+ rand(num_params) * epsilon,
        leading_edge_weight = params.leading_edge_weight .+ rand() * epsilon,
        TE_thickness = params.TE_thickness .+ rand() * epsilon
    )
end


function py_get_kulfan_from_file(filepath)
    params = py_get_kulfan_parameters(np_genfromtxt(filepath))

    upper_weights = pyconvert(Vector{Float64}, params["upper_weights"])
    lower_weights = pyconvert(Vector{Float64}, params["lower_weights"])
    leading_edge_weight = pyconvert(Float64, params["leading_edge_weight"])
    trailing_edge_thickness = pyconvert(Float64, params["TE_thickness"])

    return NeuralFoil.KulfanParameters(
        upper_weights = upper_weights,
        lower_weights = lower_weights,
        leading_edge_weight = leading_edge_weight,
        TE_thickness = trailing_edge_thickness,
    )
end


function run_network_from_file(filepath, epsilon)
    params_orig = py_get_kulfan_from_file(filepath)
    params_modified = add_perturbations_to_params(params_orig, epsilon)

    alpha = -180:180
    Re = clamp(1e3 ^ exp(rand()), 1e3, 1e9)

    ans_orig = NeuralFoil.get_aero_from_kulfan_parameters(params_orig, alpha, Re)
    ans_modified = NeuralFoil.get_aero_from_kulfan_parameters(params_modified, alpha, Re)

    return ans_orig, ans_modified
end


function test_network_from_file(filepath, epsilon; atol=1e-3)
    name = split(split(filepath, "/")[end], ".")[1]
    ans_orig, ans_modified = run_network_from_file(filepath, epsilon)

    @testset "$name" begin
        @test isapprox(ans_orig.analysis_confidence, ans_modified.analysis_confidence; atol=atol)
        @test isapprox(ans_orig.cl, ans_modified.cl; atol=atol)
        @test isapprox(ans_orig.cd, ans_modified.cd; atol=atol)
        @test isapprox(ans_orig.cm, ans_modified.cm; atol=atol)
        # @test isapprox(ans_orig.top_xtr, ans_modified.top_xtr; atol=atol)
        # @test isapprox(ans_orig.bot_xtr, ans_modified.bot_xtr; atol=atol)
        #
        # @test isapprox(ans_orig.upper_bl_ue_over_vinf, ans_modified.upper_bl_ue_over_vinf; atol=atol)
        # @test isapprox(ans_orig.upper_H, ans_modified.upper_H; atol=atol)
        # @test isapprox(ans_orig.upper_theta, ans_modified.upper_theta; atol=atol)
        #
        # @test isapprox(ans_orig.lower_bl_ue_over_vinf, ans_modified.lower_bl_ue_over_vinf; atol=atol)
        # @test isapprox(ans_orig.lower_H, ans_modified.lower_H; atol=atol)
        # @test isapprox(ans_orig.lower_theta, ans_modified.lower_theta; atol=atol)
    end

    nothing
end


function test_network_on_dataset(directory, epsilon; atol=1e-3)
    @testset "Compare entire database" begin
        for file in readdir(directory)
            test_network_from_file(joinpath(directory, file), epsilon; atol=atol)
        end
    end

    nothing
end

function run(; epsilon=1e-3, atol=1e-3)
    test_network_on_dataset(joinpath(@__DIR__, "../airfoils"), epsilon; atol=atol)
end
