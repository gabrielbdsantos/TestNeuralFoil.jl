# TestNeuralFoil.jl

Small playground to call and test
[NeuralFoil.jl](https://github.com/byuflowlab/NeuralFoil.jl) against the
original Python package.


## Tests description (and rationale)

The airfoil database was created using
[airfoildb](https://github.com/gabrielbdsantos/airfoildb), which downloads
the airfoil coordinates from the UIUC airfoil database and standardizes the
airfoil coordinates using B-Splines.

> [!IMPORTANT]
> The tests evaluating the neural network have a hint of randomness in them.
> That's because the NeuralFoil.jl differs from the Python package when it
> comes to evaluating a range of angles of attack and Reynolds numbers. In
> simple terms, NeuralFoil.jl combines all angles with all Reynolds numbers,
> whereas the original Python package `zip`s them. So, as the inputs have
> different dimensions, so do the output.
>
> As a workaround, the tests take in a single Reynolds number. However,
> instead of fixing a value, it randomly evaluates a number between 1e^3
> and 1e^9. Thus, some tests may present slightly different results when
> evaluated consecutively.

1. `scripts/test_kulfan_parameters.jl`

    It tests whether NeuralFoil.jl produces the same CST parameters as the
    Python package when evaluated on a large dataset.

    Current status: 0 passed, 1638 failed, 0 errored, 0 broken.

1. `scripts/test_network.jl`

    It tests whether NeuralFoil.jl and the Python package produce the same
    output given the same set of inputs---i.e., Kulfan parameters, angles of
    attack, and Reynolds numbers.

    Current status: 9828 passed, 0 failed, 0 errored, 0 broken.

    > [!NOTE]
    > In this case, some tests are disabled. That is because the results for
    > boundary layer do not pass the tests. Some preliminary tests I've done
    > show that the numerical results are similar, but the output matrices
    > are different. That is likely due to the way that NeuralFoil.jl
    > reshapes the output matrices. Nonetheless, it deserves further
    > investigation.

1. `scripts/test_entire_pipeline.jl`

    It tests whether NeuralFoil.jl produces the same results as the Python
    package given the airfoil coordinates, angles of attack, and Reynolds
    numbers.

    Current status: 7 passed, 9821 failed, 0 errored, 0 broken.

1. `scripts/test_network_sensitivity_to_kulfan_parameters.jl`

    Given that NeuralFoil.jl passes all tests in item #2 and fails basically
    all tests in item #1, the differences between the Julia and Python
    packages are likely caused by feeding the neural network with different
    Kulfan parameters. However, how sensitive is the neural network to
    perturbations in the Kulfan parameters?

    So, this test compares the outputs produced by the Kulfan parameters
    obtained for various airfoils with the outputs obtained using slightly
    different Kulfan parameters (in the order of ~1e-3). Also,
    the absolute tolerance for the tests was slightly higher than the usual.

    Current status:
    - (epsilon=1e-3, atol=1e-3): 44 passed, 19612 failed, 0 errored, 0 broken.
    - (epsilon=1e-4, atol=1e-3): 1825 passed, 17831 failed, 0 errored, 0 broken.
    - (epsilon=1e-5, atol=1e-3): 7601 passed, 12055 failed, 0 errored, 0 broken.

## Setup

1. Prerequisites

    - Python >= 3.10 and `uv`
    - Julia >= 1.10

1. Install Python dependencies using `uv`.

    ```bash
    $ uv sync
    ```

1. Install Julia dependencies.

    ```bash
    $ julia --project=@. --eval 'using Pkg; Pkg.instantiate()'
    ```

## License

This repository is licensed under the terms of the MIT license. For further
information, see (LICENSE)[LICENSE.md].
