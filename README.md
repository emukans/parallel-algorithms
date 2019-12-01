# Matrix multiplication on MPI

## Prerequisites
1. Open MPI. For MacOS can be installed via brew
    ```shell script
    brew install openmpi
    ```
2. Python dependencies. Can be installed via pipenv or pip
    ```shell script
    pipenv install
    ```

## Getting started
1. Populate `Input.txt` file with an initial matrix
2. Multiply the matrix with transposed matrix using command
    ```shell script
    mpirun -np 4 python -m multiply_matrix
    ```
   Where 4 is number of processes
