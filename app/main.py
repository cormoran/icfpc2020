import requests
import sys
import math
import os
import dataclasses

from interpreter import Interpreter
import operations as op


def get(index: int, node: op.Cons):
    if index == 0:
        return op.Ap(op.Car(), node).evaluate(op.Environment())
    return get(index - 1, op.Ap(op.Cdr(), node))


@dataclasses.dataclass
class StaticGameInfo:
    x0: op.Node
    role: op.Node
    x2: op.Node
    x3: op.Node
    x4: op.Node

    def __init__(self, node: op.Node):
        self.x0 = get(0, node)
        self.role = get(1, node)
        self.x2 = get(2, node)
        self.x3 = get(3, node)
        self.x4 = get(4, node)


@dataclasses.dataclass
class Ship:
    role: op.Node
    shipId: op.Node
    position: op.Node
    velocity: op.Node
    x4: op.Node
    x5: op.Node
    x6: op.Node
    x7: op.Node

    def __init__(self, node: op.Node):
        self.role = get(0, node)
        self.shipId = get(1, node)
        self.position = get(2, node)
        self.velocity = get(3, node)
        self.x4 = get(4, node)
        self.x5 = get(5, node)
        self.x6 = get(6, node)
        self.x7 = get(7, node)


@dataclasses.dataclass
class GameState:
    gameTick: op.Node
    x1: op.Node
    shipAndCommands: op.Node

    def __init__(self, node: op.Node):
        if isinstance(node, op.Nil):
            return
        self.gameTick = get(0, node)
        self.x1 = get(1, node)
        self.shipAndCommands = self._parse_shipAndCommands(get(2, node))

    def _parse_shipAndCommands(self, node: op.Node):
        if isinstance(node, op.Nil):
            return []
        shipAndCommand = get(0, node)
        ship = Ship(get(0, shipAndCommand))
        appliedCommands = get(1, shipAndCommand)
        return [(ship, appliedCommands)]


@dataclasses.dataclass
class GameResponse:
    success: op.Node
    gameStage: op.Node
    staticGameInfo: StaticGameInfo

    def __init__(self, node: op.Node):
        self.success = get(0, node)
        if self.success != op.Number(1):
            return
        self.gameStage = get(1, node)
        self.staticGameInfo = StaticGameInfo(get(2, node))
        self.gameState = GameState(get(3, node))


def print_game_response(response):
    gresponse = GameResponse(response)
    print(gresponse)


def main():
    server_url = sys.argv[1]
    player_key = sys.argv[2]
    dev_mode = sys.argv[3] if len(sys.argv) == 4 else False

    print('ServerUrl: %s; PlayerKey: %s' % (server_url, player_key))

    op.set_parameter(
        server_url, '' if not dev_mode else
        ("?apiKey=" + os.environ.get("API_KEY", "")))

    interpreter = Interpreter()

    if dev_mode:
        res = interpreter.evaluate_expression("ap send ( 1 )")
        print(res)
        if not isinstance(res, op.Cons) or res.args[0] != op.Number(1):
            raise Exception("failed to CREATE player_key")
        attacker_player_key = op.Ap(
            op.Car(), op.Ap(op.Cdr(), op.Ap(op.Car(), op.Ap(op.Cdr(), res))))
        defender_player_key = op.Ap(
            op.Car(),
            op.Ap(op.Cdr(),
                  op.Ap(op.Car(), op.Ap(op.Cdr(), op.Ap(op.Cdr(), res)))))
        print(attacker_player_key)
        print(defender_player_key)
        exit(0)

    print_game_response(
        interpreter.evaluate_expression(f"ap send ( 2 , {player_key} , ( ) )"))

    print_game_response(
        interpreter.evaluate_expression(
            f"ap send ( 3 , {player_key} , ( 1 , 2 , 3 , 4 ) )"))

    print_game_response(
        interpreter.evaluate_expression(f"ap send ( 4 , {player_key} , ( ) )"))


if __name__ == '__main__':
    main()
