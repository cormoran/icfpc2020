import sys
import traceback

sys.setrecursionlimit(999999)

from interpreter import Interpreter, evaluate_all
import operations as op


def get(index: int, node: op.Cons):
    if index == 0:
        return op.Ap(op.Car(), node).evaluate(op.Environment())
    return get(index - 1, op.Ap(op.Cdr(), node))


def print_picture_list(node: op.Cons):
    print(node.args[0].print())
    # for p in node.args[0].points:
    #     pygame.draw.rect(screen, (255, 255, 255), (x, y, x + 1, y + 1))

    if isinstance(node.args[1], op.Nil):
        return
    print()
    print_picture_list(node.args[1])


interpreter = Interpreter()

# import sys, pygame

# pygame.init()

# size = width, height = 320, 240
# speed = [2, 2]
# black = 0, 0, 0

# screen = pygame.display.set_mode(size)

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
    i = 0
    while True:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         sys.exit()
        print("input x, y > ", end="")
        x, y = map(int, input().split())
        res = interpreter.evaluate_expression(
            f"ap ap ap interact galaxy {current_state} ap ap vec {x} {y}")
        newState = get(0, res)
        image = get(1, res)
        interpreter.register_variable(f"state{i}", newState)
        current_state = f"state{i}"
        # print(evaluate_all(interpreter.env, image, True))
        print_picture_list(evaluate_all(interpreter.env, image, True))
        # print(evaluate_all(interpreter.env, get(0, image)).print())
        # print(image)
        # print(evaluate_all(interpreter.env, get(1, image)).print())
        i += 1

        # screen.fill(black)
        # pygame.display.flip()
