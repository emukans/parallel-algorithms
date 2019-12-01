import os
import numpy as np
import time
from mpi4py import MPI


def read_matrix():
    """
    Read the initial matrix from `Input.txt`
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'Input.txt'), 'r') as file:
        size = int(file.readline())
        matrix = np.empty((0, size))
        for _ in range(size):
            row = np.array([int(string_value) for string_value in file.readline().split(' ')])
            matrix = np.append(matrix, [row], axis=0)

        return matrix


def save_matrix(matrix):
    """
    Save a result to `Output.txt`

    :param matrix:
    :return:
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'Output.txt'), 'w') as file:
        for row in matrix:
            output = ' '.join([f'{i:.0f}' for i in row])
            file.write(f'{output}\n')


def master_operation():
    """
    Split the initial matrix into sub-matrices and send them for calculation to workers
    """
    workers = MPI.COMM_WORLD.Get_size() - 1
    matrix = read_matrix()
    transpose_matrix = np.ndarray.transpose(matrix)
    start_time = time.time()

    array_chunks = [chunk for chunk in np.array_split(matrix, workers) if len(chunk)]
    pid = 1

    for chunk in np.nditer(array_chunks):
        MPI.COMM_WORLD.send(chunk, dest=pid, tag=1)
        MPI.COMM_WORLD.send(transpose_matrix, dest=pid, tag=2)
        pid = pid + 1

    result_matrix = np.empty((0, matrix.shape[1]))

    pid = 1
    for n in range(len(array_chunks)):
        chunk = MPI.COMM_WORLD.recv(source=pid, tag=pid)
        result_matrix = np.append(result_matrix, [chunk], axis=0)
        pid = pid + 1

    end_time = time.time()

    save_matrix(result_matrix)

    print(f'Time taken in seconds {end_time - start_time}')


def slave_operation():
    """
    Perform matrix multiplication on received matrices
    """

    x = MPI.COMM_WORLD.recv(source=0, tag=1)
    y = MPI.COMM_WORLD.recv(source=0, tag=2)

    result = np.matmul(y, x)

    rank = MPI.COMM_WORLD.Get_rank()
    MPI.COMM_WORLD.send(result, dest=0, tag=rank)
