try
    # Try to reuse the local installed conda environment
    readdir(".CondaPkg")

    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"] = joinpath(pwd(), split("/.CondaPkg/.pixi/envs/default/bin/python", "/")...)
catch IOError
    # Otherwise create a new Python environment and install the necessary packages
    using CondaPkg
    CondaPkg.add_pip("aerosandbox")
end

using Test
using PythonCall
using NeuralFoil
using DelimitedFiles

import Base.isapprox
import Base.-


function reduce_test_verbosity()
    # Supress the stacktrace. Better to keep this for the sake of simplicity in case many tests fail.
    Test.eval(quote
	    function record(ts::DefaultTestSet, t::Union{Fail, Error})
		    push!(ts.results, t)
	    end
    end)
end


function -(a::T, b::T) where {T <: KulfanParameters}
    return KulfanParameters(
        a.upper_weights .- b.upper_weights,
        a.lower_weights .- a.lower_weights,
        a.leading_edge_weight .- b.leading_edge_weight,
        a.TE_thickness .- b.TE_thickness
    )
end


# function isapprox(a::T, b::T; kwargs...) where {T <: KulfanParameters}
#     all([
#         isapprox(a.lower_weights, b.lower_weights; kwargs...),
#         isapprox(a.upper_weights, b.upper_weights; kwargs...),
#         isapprox(a.leading_edge_weight, b.leading_edge_weight; kwargs...),
#         isapprox(a.TE_thickness, b.TE_thickness; kwargs...),
#     ])
# end
function isapprox(a::KulfanParameters, b::KulfanParameters; kwargs...)
    all(
        stack([
            isapprox.(a.upper_weights, b.upper_weights; kwargs...);
            isapprox.(a.lower_weights, b.lower_weights; kwargs...);
            isapprox.(a.leading_edge_weight, b.leading_edge_weight; kwargs...);
            isapprox.(a.TE_thickness, b.TE_thickness; kwargs...)
        ])
    )
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


@inline function normalize_coordinates!(coords)
    coords[:, 1] .-= minimum(coords[:, 1])
    coords ./= maximum(coords[:, 1])
end


function coordinates_from_file(filepath)
    coords = readdlm(filepath)

    normalize_coordinates!(coords)

    # NeuralFoil.jl requires the number of coordinates to compute the CST parameters.
    if iseven(size(coords, 1))
        id = argmin(coords[:, 1])

        return [coords[1:id, :]; coords[id:end, :]]
    else
        return coords
    end
end
