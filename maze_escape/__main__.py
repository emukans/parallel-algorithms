from mpi4py import MPI

from maze_escape.operation import breadth_first_search_escape, parallel_escape

workers = MPI.COMM_WORLD.Get_size()

if workers == 1:
    breadth_first_search_escape()
else:
    parallel_escape()
