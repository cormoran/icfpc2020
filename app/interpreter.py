import typing
import dataclasses
import math
import requests
import traceback
import os

from operations import *


# evaluate all ap nodes on ast
# returns ast which **does not** include `ap`
def evaluate_all(env: Environment, node: Node):
    prev = node
    while True:
        node = node.evaluate(env)
        if prev == node:
            break
        prev = node
    if isinstance(node, NArgOp):
        node.args = list(map(lambda a: evaluate_all(env, a), node.args))
    return node


class Interpreter():
    def __init__(self):
        self.env = Environment()

    def evaluate_assignment(self, assignment_expression: str):
        tokens = assignment_expression.split()
        assert tokens[1] == "="
        var_name = tokens[0]
        assert var_name not in self.env.var_dict
        self.env.var_dict[var_name] = self._evaluate_expression(tokens[2:],
                                                                lazy_eval=True)
        # print(f"{var_name} = {self.env.var_dict[var_name].print()}")

    def register_variable(self, ident: str, node: Node):
        self.env.var_dict[ident] = node

    # evaluate expression such as `ap inc 0`
    # and returns ast root node
    def evaluate_expression(self, expression: str):
        return self._evaluate_expression(expression.split())

    def _evaluate_expression(self,
                             expression_tokens: typing.List[str],
                             lazy_eval=False):
        root, _ = self._build(0, expression_tokens)
        return evaluate_all(self.env, root) if not lazy_eval else root

    def _build(self, i, tokens):
        if tokens[i] == "ap":
            func, i = self._build(i + 1, tokens)
            arg, i = self._build(i, tokens)
            return Ap(func, arg), i
        elif tokens[i] == "(":
            # parse list construction syntax
            if tokens[i + 1] == ")":
                return Nil(), i + 1
            elem, i = self._build(i + 1, tokens)
            elements = [elem]
            while tokens[i] == ",":
                elem, i = self._build(i + 1, tokens)
                elements.append(elem)
            assert tokens[i] == ")"

            return self._build_list(0, elements), i + 1
        elif tokens[i] == "[":
            elem, i = self._build(i + 1, tokens)
            assert tokens[i] == "]"
            return Ap(Modulate(), elem), i + 1
        else:
            return self._token_to_node(tokens[i]), i + 1

    # (1, 2, ...) -> ap ap cons 1 ap ap cons 2, ..., nil
    def _build_list(self, i, elements):
        if i == len(elements):
            return Nil()
        return Ap(Ap(Cons(), elements[i]), self._build_list(i + 1, elements))

    def _token_to_node(self, token: str):
        if token in token_node_map:
            return token_node_map[token]()
        # if token in self.env.var_dict:
        #     return self.env.var_dict[token]
        try:
            return Number(int(token))
        except ValueError:
            return Variable(token)
