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

    current_state = "nil"
    for i in range(10):
        res = interpreter.evaluate_expression(
            f"ap ap ap interact galaxy {current_state} ap ap vec 0 0 ")
        newState = get(0, res)
        image = get(1, res)
        interpreter.register_variable(f"state{i}", newState)
        current_state = f"state{i}"
        print(evaluate_all(interpreter.env, image, True).print())
        # print(evaluate_all(interpreter.env, get(0, image)).print())
        # print(image)
        # print(evaluate_all(interpreter.env, get(1, image)).print())
