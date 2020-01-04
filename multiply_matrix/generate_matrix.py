import random
import sys

size = int(sys.argv[1]) or 10

with open('./multiply_matrix/Input.txt', 'w') as file:
    file.write(f'{size}\n')

    for _ in range(size):
        line = [str(random.randrange(0, 9)) for _ in range(size)]
        line_str = " ".join(line)

        file.write(f'{line_str}\n')
