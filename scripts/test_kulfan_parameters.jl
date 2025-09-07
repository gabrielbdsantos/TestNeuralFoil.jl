ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(pwd(), ".venv/bin/python")

using Test
using PythonCall
using NeuralFoil
using DelimitedFiles

import Base.isapprox


# Supress the stacktrace. Better to keep this for the sake of simplicity in case many tests fail.
Test.eval(quote
	function record(ts::DefaultTestSet, t::Union{Fail, Error})
		push!(ts.results, t)
	end
end)


function isapprox(a::T, b::T; kwargs...) where {T <: KulfanParameters}
    all([
        isapprox(a.lower_weights, b.lower_weights; kwargs...),
        isapprox(a.upper_weights, b.upper_weights; kwargs...),
        isapprox(a.leading_edge_weight, b.leading_edge_weight; kwargs...),
        isapprox(a.TE_thickness, b.TE_thickness; kwargs...),
    ])
end


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


function coordinates_from_file(filepath)
    return readdlm(filepath)
end


function py_get_kulfan_from_file(filepath)
    coords = np_array(coordinates_from_file(filepath))
    params = py_get_kulfan_parameters(coords, normalize_coordinates=true)

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


function jl_get_kulfan_from_file(filepath)
    coords = coordinates_from_file(filepath)
    return NeuralFoil.get_kulfan_parameters(coords)
end


function test_kulfan_from_file(filepath; atol=1e-6)
    name = split(split(filepath, "/")[end], ".")[1]
    py_ans = py_get_kulfan_from_file(filepath)
    jl_ans = jl_get_kulfan_from_file(filepath)

    @testset "$name" begin
        @test isapprox(py_ans, jl_ans, atol=atol)
    end
end


function test_kulfan_on_dataset(directory; atol=1e-6)
    @testset "Compare entire database" begin
        for file in readdir(directory)
            test_kulfan_from_file(joinpath(directory, file); atol=atol)
        end
    end

    nothing
end


run(; atol=1e-6) = test_kulfan_on_dataset(joinpath(@__DIR__, "../airfoils"); atol=atol)
