import typing
import dataclasses


class Interpreter():
    def __init__(self):
        self.var_dict = {}

    def evaluate_assignment(self, assignment_expression: str):
        tokens = assignment_expression.split()
        assert tokens[1] == "="
        var_name = tokens[0]
        assert var_name not in self.var_dict
        self.var_dict[var_name] = self._evaluate_expression(tokens)
        print(f"{var_name} = {self.var_dict[var_name]}")

    def _evaluate_expression(self, expression_tokens: typing.List[str]):
        ...


@dataclasses.dataclass
class Node:

    # lazy evaluation
    def ap(self, arg):  # -> Node:
        raise Exception('cannot ap to', self.__class__.__name__)

    # return Number ?
    def evaluate(self):  # -> Node:
        return None


def ensure_type(node: Node, t: typing.ClassVar) -> Node:
    assert isinstance(node, t)
    return node


@dataclasses.dataclass
class Number(Node):
    n: int

    def evaluate(self) -> Node:
        return self


@dataclasses.dataclass
class Boolean(Node):
    b: bool

    def evaluate(self) -> Node:
        return self


@dataclasses.dataclass
class Inc(Node):
    arg: Node = None

    def ap(self, arg) -> Node:
        assert self.arg is None
        return Inc(arg)

    def evaluate(self) -> Node:
        n = ensure_type(self.arg.evaluate(), Number)
        return Number(n.n + 1)


@dataclasses.dataclass
class Dec(Node):
    arg: Node = None

    def ap(self, arg) -> Node:
        assert self.arg is None
        return Dec(arg)

    def evaluate(self) -> Node:
        n = ensure_type(self.arg.evaluate(), Number)
        return Number(n.n - 1)


@dataclasses.dataclass
class TwoArgOp(Node):
    arg1: Node = None
    arg2: Node = None

    def ap(self, arg) -> Node:
        if self.arg1 is None:
            return Add(arg)
        elif self.arg2 is None:
            return Add(self.arg1, arg)
        assert False


@dataclasses.dataclass
class Add(TwoArgOp):
    def evaluate(self) -> Node:
        n1 = ensure_type(self.arg1.evaluate(), Number)
        n2 = ensure_type(self.arg2.evaluate(), Number)
        return Number(n1.n + n2.n)


@dataclasses.dataclass
class Mul(TwoArgOp):
    def evaluate(self) -> Node:
        n1 = ensure_type(self.arg1.evaluate(), Number)
        n2 = ensure_type(self.arg2.evaluate(), Number)
        return Number(n1.n * n2.n)


@dataclasses.dataclass
class Div(TwoArgOp):
    def evaluate(self) -> Node:
        n1 = ensure_type(self.arg1.evaluate(), Number)
        n2 = ensure_type(self.arg2.evaluate(), Number)
        return Number(n1.n // n2.n)


@dataclasses.dataclass
class Eq(TwoArgOp):
    def evaluate(self) -> Node:
        # TODO: 評価しなくても、subtree が同じ形ならよさそう
        n1 = ensure_type(self.arg1.evaluate(), Number)
        n2 = ensure_type(self.arg2.evaluate(), Number)
        return


@dataclasses.dataclass
class Ap(Node):
    func: typing.Optional[Node]
    arg: typing.Optional[Node]

    # return func or value
    def evaluate(self) -> Node:
        return self.func.ap(self.arg)


class AbstractSyntaxTree():
    def __init__(self, tokens):
        self.root = Node(None, None)


def evaluate_all(node: Node):
    while not isinstance(node, Number):
        node = node.evaluate()
    return node


if __name__ == '__main__':
    print(Number(1))
    print(Inc())
    print(Ap(Inc(), Number(1)))
    print(evaluate_all(Ap(Inc(), Number(1))))
