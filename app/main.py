import requests
import sys
import math
import os

from interpreter import Interpreter
import operations


def print_game_response(response):
    assert isinstance(response, operations.Cons)
    assert isinstance(response.args[0], operations.Number)
    success = response.args[0].n
    if success == 0:
        print("Fail Response")
        return

    print("Success Response")
    assert isinstance(response.args[1], operations.Cons)
    response = response.args[1]
    assert isinstance(response.args[0], operations.Number)
    stage = response.args[0].n
    print("\tstage", stage)

    assert isinstance(response.args[1], operations.Cons)
    response = response.args[1]
    list_a = response.args[0]

    assert isinstance(response.args[1], operations.Cons)
    response = response.args[1]
    state = response.args[0]

    print("\tstage", stage)
    print("\tlist", list_a.print())
    print("\tstate", state.print())


def main():
    server_url = sys.argv[1]
    player_key = sys.argv[2]
    dev_mode = len(sys.argv) == 4 and sys.argv[3] == "1"

    print('ServerUrl: %s; PlayerKey: %s' % (server_url, player_key))

    operations.set_parameter(
        server_url, '' if not dev_mode else
        ("?apiKey=" + os.environ.get("API_KEY", "")))

    interpreter = Interpreter()

    print_game_response(
        interpreter.evaluate_expression(f"ap send ( 2 , {player_key} , ( ) )"))

    print_game_response(
        interpreter.evaluate_expression(
            f"ap send ( 3 , {player_key} , ( 1 , 2 , 3 , 4 ) )"))

    print_game_response(
        interpreter.evaluate_expression(f"ap send ( 4 , {player_key} , ( ) )"))


if __name__ == '__main__':
    main()
