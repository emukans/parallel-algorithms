import os
import numpy as np
from mpi4py import MPI


def read_matrix():
    """
    Read the initial matrix from `Input.txt`
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'Input.txt'), 'r') as file:
        shape = [int(i) for i in file.readline().split(' ')]
        matrix = np.empty((0, shape[1]))
        for _ in range(shape[0]):
            row = np.array([int(string_value) for string_value in file.readline().split(' ')])
            matrix = np.append(matrix, [row], axis=0)

        return matrix


def save_result(result):
    """
    Save a result to `Output.txt`

    :param result:
    :return:
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'Output.txt'), 'w') as file:
        if len(result):
            file.write(f'{len(result)}\n')
            output = ' '.join([f'{col},{row}' for col, row in result])
            file.write(f'{output}\n')
        else:
            file.write('-1\n')


def master_operation():
    """
    Split the initial matrix into tasks and distribute them among slave operations
    """
    workers = MPI.COMM_WORLD.Get_size()
    matrix = read_matrix()
    accumulator = []
    task_queue = []
    if not matrix[0][0]:
        task_queue.append([(0, 0)])

    while True:
        sent_workers = []

        for pid in range(1, workers):
            if len(task_queue):
                start = task_queue.pop(0)

                sent_workers.append(pid)
                MPI.COMM_WORLD.send(dict(matrix=matrix, start=start, status='process'), dest=pid, tag=1)

        if not len(task_queue) and not len(sent_workers):
            break

        for pid in sent_workers:
            data = MPI.COMM_WORLD.recv(source=pid, tag=pid)

            if len(data['result']) == 1:
                accumulator.append(data['result'][0])
            elif len(data['result']) > 1:
                task_queue += data['result']

    for pid in range(1, workers):
        MPI.COMM_WORLD.isend(dict(status='terminate'), dest=pid, tag=1)

    accumulator.sort(key=lambda sequence: len(sequence))

    result = [(col, row) for (row, col) in accumulator[0]] if len(accumulator) else []

    save_result(result)


def slave_operation():
    """
    Try to find all possible paths in a maze
    """

    while True:
        data = MPI.COMM_WORLD.recv(source=0, tag=1)
        if data['status'] == 'terminate':
            return

        matrix = data['matrix']
        task = data['start']

        result = find_path(matrix, task)

        rank = MPI.COMM_WORLD.Get_rank()
        MPI.COMM_WORLD.send(dict(result=result), dest=0, tag=rank)


def find_path(matrix, task):
    """
    Find a all available paths for a next step
    :param matrix:
    :param task:
    :return:
    """
    result = []
    goal = (len(matrix) - 1, len(matrix[-1]) - 1)

    while True:
        candidate = []
        last_point = task[-1]
        if len(matrix) > last_point[0] + 1:
            if len(matrix[last_point[0] + 1]) > last_point[1] + 1 and not matrix[last_point[0] + 1][last_point[1] + 1] and (last_point[0] + 1, last_point[1] + 1) not in task:
                candidate.append((last_point[0] + 1, last_point[1] + 1))

            if last_point[1] - 1 >= 0 and len(matrix[last_point[0] + 1]) > last_point[1] + 1 and not matrix[last_point[0] + 1][last_point[1] - 1] and (last_point[0] + 1, last_point[1] - 1) not in task:
                candidate.append((last_point[0] + 1, last_point[1] - 1))

        if len(matrix) > last_point[0] - 1 >= 0:
            if len(matrix[last_point[0] - 1]) > last_point[1] - 1 and not matrix[last_point[0] - 1][last_point[1] + 1] and (last_point[0] - 1, last_point[1] + 1) not in task:
                candidate.append((last_point[0] - 1, last_point[1] + 1))

            if 0 <= last_point[1] - 1 < len(matrix[last_point[0] - 1]) and not matrix[last_point[0] - 1][last_point[1] - 1] and (last_point[0] - 1, last_point[1] - 1) not in task:
                candidate.append((last_point[0] - 1, last_point[1] - 1))

        if not len(candidate):
            result = []
            break

        if len(candidate) == 1:
            task.append(candidate.pop())
            if task[-1] == goal:
                result = [task]
                break

        if len(candidate) > 1:
            for item in candidate:
                if item == goal:
                    result = [task + [item]]
                    return result

                result.append(task + [item])
            break

    return result
