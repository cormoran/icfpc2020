import sys
import traceback

sys.setrecursionlimit(999999)

from interpreter import Interpreter

interpreter = Interpreter()

for expression in [
        "ap ap ap interact statelessdraw nil ap ap vec 1 0 = ( nil , ( [1,0] ) )",
        "ap ap ap interact statelessdraw nil ap ap vec 2 3 = ( nil , ( [2,3] ) )",
        "ap ap ap interact statelessdraw nil ap ap vec 4 1 = ( nil , ( [4,1] ) )",
]:
    print(interpreter.evaluate_expression(expression).print())
print('statefull draw')
val = interpreter.evaluate_expression(
    "ap ap ap interact statefulldraw ( ap ap vec 0 0 ) ap ap vec 2 3")
interpreter.register_variable("x2", val.args[0])
print(val.args[1].print())
val = interpreter.evaluate_expression(
    "ap ap ap interact statefulldraw x2 ap ap vec 1 2")
print(val.args[1].print())
interpreter.register_variable("x3", val.args[0])
val = interpreter.evaluate_expression(
    "ap ap ap interact statefulldraw x3 ap ap vec 3 2")
print(val.args[1].print())
interpreter.register_variable("x4", val.args[0])
val = interpreter.evaluate_expression(
    "ap ap ap interact statefulldraw x4 ap ap vec 4 0")
print(val.args[1].print())