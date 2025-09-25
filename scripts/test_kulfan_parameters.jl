include("common.jl")


@wrap_pyfunction "numpy" array np_array
@wrap_pyfunction "numpy" genfromtxt np_genfromtxt
@wrap_pyfunction "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters py_get_kulfan_parameters


function py_get_kulfan_from_file(filepath)
    coords = np_array(coordinates_from_file(filepath))
    params = py_get_kulfan_parameters(coords, normalize_coordinates=false)

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


function test_kulfan_from_file(filepath; atol=1e-2)
    name = split(split(filepath, "/")[end], ".")[1]
    py_ans = py_get_kulfan_from_file(filepath)
    jl_ans = jl_get_kulfan_from_file(filepath)

    @testset "$name" begin
        @test isapprox(py_ans, jl_ans; atol=atol)
    end
end


function test_kulfan_on_dataset(directory; atol=1e-2)
    reduce_test_verbosity()

    @testset "Compare entire database" begin
        for file in readdir(directory)
            test_kulfan_from_file(joinpath(directory, file); atol=atol)
        end
    end

    nothing
end


run(dataset; atol=1e-2) = test_kulfan_on_dataset(dataset; atol=atol)
