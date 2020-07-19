import sys
import traceback

sys.setrecursionlimit(999999)

from interpreter import Interpreter

interpreter = Interpreter()

with open('./app/galaxy.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        try:
            interpreter.evaluate_assignment(line)
        except Exception as e:
            print(f"failed to parse line {i}")
            print(traceback.format_exc())
    # print(
    #     interpreter.evaluate_expression(
    #         "ap ap ap interact galaxy nil ap ap vec 0 0").print())

    print(
        interpreter.evaluate_expression(
            "ap ap ap interact galaxy nil ap ap vec 1 1").print())
