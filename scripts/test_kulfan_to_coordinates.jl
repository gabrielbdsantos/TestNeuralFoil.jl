include("common.jl")


@wrap_pyfunction "numpy" array np_array
@wrap_pyfunction "numpy" genfromtxt np_genfromtxt
@wrap_pyfunction "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters py_get_kulfan_parameters
@wrap_pyfunction "aerosandbox.geometry.airfoil" Airfoil py_Airfoil
@wrap_pyfunction "aerosandbox.geometry.airfoil" KulfanAirfoil py_KulfanAirfoil



function py_get_kulfan_from_coords(coords)
    params = py_get_kulfan_parameters(np_array(coords), normalize_coordinates=false)

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


function py_get_kulfan_from_file(filepath)
    coords = coordinates_from_file(filepath)
    return py_get_kulfan_from_coords(coords)
end


function jl_get_kulfan_from_coords(coords)
    return NeuralFoil.get_kulfan_parameters(coords)
end


function jl_get_kulfan_from_file(filepath)
    coords = coordinates_from_file(filepath)
    return jl_get_kulfan_from_coords(coords)
end


function py_kulfan_to_coordinates(params; npoints=100)
    airfoil = py_KulfanAirfoil(
        upper_weights=params.upper_weights,
        lower_weights=params.lower_weights,
        leading_edge_weight=params.leading_edge_weight,
        TE_thickness=params.TE_thickness
    )

    return pyconvert(
        Matrix{Float64},
        airfoil.to_airfoil(n_coordinates_per_side=npoints).coordinates
    )
end


function compare_coordinates(filepath)
    coords = coordinates_from_file(filepath)

    py_coords = py_kulfan_to_coordinates(py_get_kulfan_from_file(filepath))
    jl_coords = py_kulfan_to_coordinates(jl_get_kulfan_from_file(filepath))

    return coords, py_coords, jl_coords
end


function test_coordinates(filepath; atol=1e-6)
    name = split(split(filepath, "/")[end], ".")[1]
    _, py_coords, jl_coords = compare_coordinates(filepath)

    @testset "$name" begin
        @test isapprox(py_coords, jl_coords, atol=atol)
    end
end


function test_coordinates_on_dataset(directory; atol=1e-6)
    reduce_test_verbosity()

    @testset "Compare entire database" begin
        for file in readdir(directory)
            test_coordinates(joinpath(directory, file); atol=atol)
        end
    end

    nothing
end


run(database; atol=1e-2) = test_coordinates_on_dataset(database; atol=atol)
