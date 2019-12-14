from .operation import slave_operation, master_operation

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

if rank > 0:
    slave_operation()
else:
    master_operation()
