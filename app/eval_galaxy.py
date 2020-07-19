import sys

sys.setrecursionlimit(999999)

from interpreter import Interpreter

interpreter = Interpreter()

with open('./app/galaxy.txt', 'r') as f:
    exceptions = []
    for i, line in enumerate(f.readlines()):
        try:
            interpreter.evaluate_assignment(line)
        except Exception as e:
            print(f"failed to parse line {i}")
            exceptions.append((i, line))
    while len(exceptions) > 0:
        exceptions2 = []
        for i, line in exceptions:
            try:
                interpreter.evaluate_assignment(line)
            except Exception as e:
                exceptions2.append((i, line))
        if len(exceptions) == len(exceptions2):
            print(f"infinite loop: left {len(exceptions)}")
            for i, line in exceptions:
                print(i + 1)
            break
        exceptions = exceptions2
