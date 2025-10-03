# TestNeuralFoil.jl

Small playground to call and test
[NeuralFoil.jl](https://github.com/byuflowlab/NeuralFoil.jl) against the
original Python package.


## Tests description (and rationale)

The airfoil database was created using
[airfoildb](https://github.com/gabrielbdsantos/airfoildb), which downloads the
airfoil coordinates from the UIUC airfoil database and standardizes them using
B-Splines. The folder `airfoils/raw` contains the raw coordinates and the
folder `airfoils/uniform` contains the standardized representation obtained
with B-Splines.

> [!IMPORTANT]
> The tests evaluating the neural network have a hint of randomness. This is
> because NeuralFoil.jl differs from the Python package in how it evaluate a
> range of angles of attack and Reynolds numbers. In simple terms,
> NeuralFoil.jl combines all angles with all Reynolds numbers, whereas the
> original Python package `zip`s them. Hence, as the inputs have different
> dimensions, so do the outputs.
>
> As a workaround, the tests take in a single Reynolds number. However,
> instead of fixing a value, it randomly evaluates a number between 1e^3
> and 1e^9. Thus, some tests may present slightly different results when
> evaluated consecutively.

1. `scripts/test_kulfan_parameters.jl`

    It tests whether NeuralFoil.jl produces the same Kulfan parameters as the
    Python package when evaluated on a large dataset.

    Current status (airfoils/raw): 896 passed, 740 failed, 0 errored, 0 broken.\
    Current status (airfoils/uniform): 421 passed, 1215 failed, 0 errored, 0 broken.

1. `scripts/test_network.jl`

    It tests whether NeuralFoil.jl and the Python package produce the same
    output given the same set of inputs---i.e., Kulfan parameters, angles of
    attack, and Reynolds numbers.

    Current status (airfoils/raw): 9816 passed, 0 failed, 0 errored, 0 broken.\
    Current status (airfoils/uniform): 9816 passed, 0 failed, 0 errored, 0 broken.

    > Some tests are disabled because the results for boundary layer do not
    > pass the tests at all. This deserves further investigation. The output
    > matrices are probably transposed.

1. `scripts/test_entire_pipeline.jl`

    It tests whether NeuralFoil.jl produces the same results as the Python
    package given the airfoil coordinates, angles of attack, and Reynolds
    numbers.

    Current status (airfoils/raw): 5323 passed, 4493 failed, 0 errored, 0 broken.\
    Current status (airfoils/uniform): 2537 passed, 7279 failed, 0 errored, 0 broken.

1. `scripts/test_network_sensitivity_to_kulfan_parameters.jl`

    Given that NeuralFoil.jl passes all tests in item #2 and fails basically
    all tests in item #1, the differences between the Julia and Python
    packages are likely caused by feeding the neural network with different
    Kulfan parameters. However, how sensitive is the neural network to
    perturbations in the Kulfan parameters?

    So, this test compares the network predictions for various airfoils against
    the predictions for slightly different Kulfan parameters (the epsilon
    parameter). Also, the absolute tolerance for the tests was slightly higher
    than the usual.

    Current status (airfoils/raw):
    - (epsilon=1e-3, atol=1e-3): 27 passed, 9789 failed, 0 errored, 0 broken.
    - (epsilon=1e-4, atol=1e-3): 1285 passed, 8531 failed, 0 errored, 0 broken.
    - (epsilon=1e-5, atol=1e-3): 4968 passed, 4848 failed, 0 errored, 0 broken.

    Current status (airfoils/uniform):
    - (epsilon=1e-3, atol=1e-3): 27 passed, 9789 failed, 0 errored, 0 broken.
    - (epsilon=1e-4, atol=1e-3): 1418 passed, 8398 failed, 0 errored, 0 broken.
    - (epsilon=1e-5, atol=1e-3): 5636 passed, 4180 failed, 0 errored, 0 broken.

1. `scripts/test_kulfan_to_coordinates.jl`

    Test whether the Kulfan parameters obtained using NeuralFoil.jl and the
    Python package generate the same set of coordinates.

    Current status (airfoils/raw): 976 passed, 660 failed, 0 errored, 0 broken.\
    Current status (airfoils/uniform): 840 passed, 796 failed, 0 errored, 0 broken.


## Setup

1. Prerequisites

    - Julia >= 1.10

1. Install Julia dependencies.

    ```bash
    $ julia --project=@. --eval 'using Pkg; Pkg.instantiate()'
    ```

## License

This repository is licensed under the terms of the MIT license. For further
information, see [LICENSE](LICENSE.md).
