# TestNeuralFoil.jl

Small playground to call and test [NeuralFoil.jl](https://github.com/byuflowlab/NeuralFoil.jl)
against the original Python package.


## Prerequisites

- Python >= 3.11 and `uv`.
- Julia >= 1.10

## Setup

1. Install Python dependencies using `uv`.

    ```bash
    $ uv sync
    ```

2. Install Julia dependencies.

    ```bash
    $ julia --project=@. --eval 'using Pkg; Pkg.instantiate()'
    ```
