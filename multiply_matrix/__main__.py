from mpi4py import MPI

from .operation import multiply_with_transpose, serial_multiply_with_transpose

workers = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

if workers == 1:
    serial_multiply_with_transpose()
else:
    multiply_with_transpose()
