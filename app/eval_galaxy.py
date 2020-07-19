import sys
import traceback

sys.setrecursionlimit(999999)

from interpreter import Interpreter, evaluate_all
import operations as op


def get(index: int, node: op.Cons):
    if index == 0:
        return op.Ap(op.Car(), node).evaluate(op.Environment())
    return get(index - 1, op.Ap(op.Cdr(), node))


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

    res = interpreter.evaluate_expression(
        "ap ap ap interact galaxy nil ap ap vec 1 1")
    newState = get(0, res)
    image = get(1, res)
    print(evaluate_all(interpreter.env, get(0, image)).print())
    print(evaluate_all(interpreter.env, get(1, image)).print())