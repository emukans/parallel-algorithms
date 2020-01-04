import random
import sys

row_size = int(sys.argv[1]) or 10
column_size = int(sys.argv[2]) or 10

with open('./maze_escape/Input.txt', 'w') as file:
    file.write(f'{row_size} {column_size}\n')

    for _ in range(row_size):
        line = [str(random.randrange(0, 2)) for _ in range(column_size)]
        line_str = " ".join(line)

        file.write(f'{line_str}\n')
