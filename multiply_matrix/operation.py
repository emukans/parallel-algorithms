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


def serial_multiply_with_transpose():
    """
    Serial matrix multiplication
    """
    matrix = read_matrix()
    start_time = time.time()
    transposed_matrix = np.transpose(matrix)

    result_matrix = np.matmul(transposed_matrix, matrix)

    end_time = time.time()
    print(f'Time taken in seconds {end_time - start_time}')
    save_matrix(result_matrix)


def multiply_with_transpose():
    """
    Split the initial matrix into sub-matrices and send them for calculation to workers
    """
    rank = MPI.COMM_WORLD.Get_rank()
    workers = MPI.COMM_WORLD.Get_size()
    sub_matrix = []
    if rank == 0:
        matrix = read_matrix()
        start_time = time.time()
        sub_matrix = np.array_split(matrix, workers, 1)
    MPI.COMM_WORLD.barrier()
    sub_matrix = MPI.COMM_WORLD.scatter(sub_matrix, root=0)
    transpose_part = np.ndarray.transpose(sub_matrix)

    part_list = MPI.COMM_WORLD.allgather(transpose_part)
    MPI.COMM_WORLD.barrier()
    transpose_matrix = np.row_stack(part_list)

    multipy_matrix = np.matmul(transpose_matrix, sub_matrix)
    result_matrix = MPI.COMM_WORLD.gather(multipy_matrix, root=0)

    if rank == 0:
        result_matrix = np.column_stack(result_matrix)
        end_time = time.time()
        print(f'Time taken in seconds {end_time - start_time}')
        save_matrix(result_matrix)
