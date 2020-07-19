from interpreter import Interpreter

interpreter = Interpreter()

print(interpreter.evaluate_expression("ap send ( 2 , 1 , ( ) )").print())

print(
    interpreter.evaluate_expression(
        "ap send ( 3 , 1 , ( 1 , 2 , 3 , 4 ) )").print())

print(interpreter.evaluate_expression("ap send ( 4 , 1 , ( ) )").print())