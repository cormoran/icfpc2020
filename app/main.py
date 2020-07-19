import requests
import sys
import math

from interpreter import Interpreter
from operations import set_parameter


def main():
    server_url = sys.argv[1]
    player_key = sys.argv[2]
    print('ServerUrl: %s; PlayerKey: %s' % (server_url, player_key))

    set_parameter(server_url, '')

    interpreter = Interpreter()

    print(
        interpreter.evaluate_expression(f"ap send ( 2 , {player_key} , ( ) )"))

    print(
        interpreter.evaluate_expression(
            f"ap send ( 3 , {player_key} , ( 1 , 2 , 3 , 4 ) )"))

    print(
        interpreter.evaluate_expression(f"ap send ( 4 , {player_key} , ( ) )"))


if __name__ == '__main__':
    main()
