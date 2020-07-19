from interpreter import Interpreter

interpreter = Interpreter()

with open('./app/galaxy.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        print("line", i)
        interpreter.evaluate_assignment(line)