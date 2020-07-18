import typing
import dataclasses
import math
import requests
import traceback
import os

server_url = "https://icfpc2020-api.testkontur.ru"
query_param = "?apiKey=" + os.environ.get("API_KEY")


@dataclasses.dataclass
class Node:

    # lazy evaluation
    def ap(self, arg):  # -> Node:
        raise Exception('cannot ap to', self.__class__.__name__)

    # return Number ?
    def evaluate(self):  # -> Node:
        return None

    # compare all subtree
    def equal(self, target):
        return False

    def is_leaf(self):
        return True


def ensure_type(node: Node, t: typing.ClassVar) -> Node:
    if not isinstance(node, t):
        raise Exception(
            f"ensure type: expected {t}, but {node.__class__.__name__}")
    return node


@dataclasses.dataclass
class Number(Node):
    n: int

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        return isinstance(target, Number) and target.n == self.n


@dataclasses.dataclass
class ModulatedNumber(Node):
    n: str

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        return isinstance(target, ModulatedNumber) and target.n == self.n


# @dataclasses.dataclass
# class Boolean(Node):
#     b: bool

#     def evaluate(self) -> Node:
#         return self

#     def equal(self, target):
#         return isinstance(target, Boolean) and target.b == self.b


@dataclasses.dataclass
class NArgOp(Node):
    args: typing.List[Node] = dataclasses.field(default_factory=list)

    # n_args: int # abstract field...

    def ap(self, arg) -> Node:
        if len(self.args) < self.n_args - 1:
            return self.__class__(self.args + [arg])
        else:
            return self.__class__(self.args + [arg]).evaluate()

    def evaluate(self) -> Node:
        assert len(self.args) == self.n_args
        return self._evaluate()

    def _evaluate(self) -> Node:
        raise NotImplementedError()

    # compare all subtree
    def equal(self, target):
        if not self.__class__ != target.__class__:
            return False
        if len(self.args) != len(target.args):
            return False
        for a, b in zip(self.args, target.args):
            if not a.equal(b):
                return False
        return True

    def is_leaf(self):
        return False


@dataclasses.dataclass
class Inc(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(self.args[0].evaluate(), Number)
        return Number(n.n + 1)


@dataclasses.dataclass
class Dec(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(self.args[0].evaluate(), Number)
        return Number(n.n - 1)


@dataclasses.dataclass
class Add(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = ensure_type(self.args[0].evaluate(), Number)
        n2 = ensure_type(self.args[1].evaluate(), Number)
        return Number(n1.n + n2.n)


@dataclasses.dataclass
class Mul(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = ensure_type(self.args[0].evaluate(), Number)
        n2 = ensure_type(self.args[1].evaluate(), Number)
        return Number(n1.n * n2.n)


@dataclasses.dataclass
class Div(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = ensure_type(self.args[0].evaluate(), Number)
        n2 = ensure_type(self.args[1].evaluate(), Number)
        sign = 1 if n1.n * n2.n >= 0 else -1
        return Number(abs(n1.n) // abs(n2.n) * sign)


@dataclasses.dataclass
class Eq(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        # TODO: 評価しなくても、subtree が同じ形ならよさそう
        n1 = self.args[0].evaluate()
        n2 = self.args[1].evaluate()
        return T() if n1.equal(n2) else F()


@dataclasses.dataclass
class Lt(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = ensure_type(self.args[0].evaluate(), Number)
        n2 = ensure_type(self.args[1].evaluate(), Number)
        return T() if n1.n < n2.n else F()


@dataclasses.dataclass
class Modulate(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(self.args[0].evaluate(), Number)
        return ModulatedNumber(self.modulate(n.n))

    def modulate(self, x: int) -> str:
        res = ""
        # signal
        if x >= 0:
            res += "01"
        else:
            res += "10"
        x = abs(x)
        # bit length
        bit_length = 0
        while (1 << bit_length) <= x:
            bit_length += 1
        bit_length = math.ceil(bit_length / 4) * 4
        for i in range(bit_length // 4):
            res += "1"
        res += "0"

        # number
        res2 = ""
        while x > 0:
            res2 += "1" if x % 2 > 0 else "0"
            x = x // 2
        while len(res2) < bit_length:
            res2 += "0"
        res += res2[::-1]
        return res


@dataclasses.dataclass
class Demodulate(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(self.args[0].evaluate(), ModulatedNumber)
        return Number(self.demodulate(n.n))

    def demodulate(self, x: str) -> int:
        signal = None
        if x.startswith("01"):
            signal = 1
        elif x.startswith("10"):
            signal = -1
        else:
            raise Exception("unknown signal " + x[:2])
        x = x[2:]

        bit_length = 0
        while x[0] == "1":
            x = x[1:]
            bit_length += 1
        x = x[1:]

        x = x[::-1]
        num = 0
        for i in range(len(x)):
            if x[i] == "1":
                num += 2**i
        return num * signal


@dataclasses.dataclass
class Send(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(
            Ap(Modulate(), self.args[0]).evaluate(), ModulatedNumber)
        res = requests.post(server_url + '/alians/send', n.n)
        if res.status_code != 200:
            print('Unexpected server response:')
            print('HTTP code:', res.status_code)
            print('Response body:', res.text)
            raise Exception('Unexpected server response:')
        return ensure_type(Ap(Demodulate(), ModulatedNumber(res.text)), Number)


@dataclasses.dataclass
class Neg(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(self.args[0].evaluate(), Number)
        return Number(-n.n)


@dataclasses.dataclass
class Ap(Node):
    func: typing.Optional[Node]
    arg: typing.Optional[Node]

    def ap(self, arg):
        return self.evaluate().ap(arg)

    # return func or value
    def evaluate(self) -> Node:
        return self.func.ap(self.arg)

    def is_leaf(self):
        return False


@dataclasses.dataclass
class S(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        # TODO: 知らんぞ
        a = Ap(self.args[0], self.args[2])
        b = Ap(self.args[1], self.args[2])
        return Ap(a, b).evaluate()


@dataclasses.dataclass
class C(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        # TODO: 知らんぞ
        a = Ap(self.args[0], self.args[2])
        return Ap(a, self.args[1]).evaluate()


@dataclasses.dataclass
class B(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        # TODO: 知らんぞ
        a = Ap(self.args[1], self.args[2])
        return Ap(self.args[0], a).evaluate()


@dataclasses.dataclass
class T(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        return self.args[0].evaluate()

    def equal(self, target):
        if isinstance(target, T) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().equal(target)

    def is_leaf(self):
        return len(self.args) == 0


@dataclasses.dataclass
class F(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        return self.args[1].evaluate()

    def equal(self, target):
        if isinstance(target, F) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().equal(target)

    def is_leaf(self):
        return len(self.args) == 0


@dataclasses.dataclass
class Pwr2(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = ensure_type(self.args[0].evaluate(), Number)
        return Number(2**n.n)


@dataclasses.dataclass
class I(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        return self.args[0]  # TODO: evaluate or not


@dataclasses.dataclass
class Cons(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        a = Ap(self.args[2], self.args[0])
        return Ap(a, self.args[1]).evaluate()


@dataclasses.dataclass
class Car(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        assert isinstance(self.args[0], Cons)
        assert len(self.args[0].args) <= 2
        return self.args[0].args[0]  # TODO: evaluate?


@dataclasses.dataclass
class Cdr(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        assert isinstance(self.args[0], Cons)
        assert len(self.args[0].args) == 2
        return self.args[0].args[1]  # TODO: evaluate?


@dataclasses.dataclass
class Nil(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        if len(self.args) == 0:
            return self
        return T()

    def equal(self, target):
        if isinstance(target, Nil) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().equal(target)

    def is_leaf(self):
        return len(self.args) == 0


@dataclasses.dataclass
class IsNil(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = self.args[0].evaluate()
        return T() if isinstance(n, Nil) else F()


token_node_map = {
    "inc": Inc,
    "dec": Dec,
    "add": Add,
    "mul": Mul,
    "div": Div,
    "eq": Eq,
    "lt": Lt,
    "mod": Modulate,
    "dem": Demodulate,
    "send": Send,
    "neg": Neg,
    #"ap": Ap,
    "s": S,
    "c": C,
    "b": B,
    "t": T,
    "f": F,
    "pwr2": Pwr2,
    "i": I,
    "cons": Cons,
    "car": Car,
    "cdr": Cdr,
    "nil": Nil,
    "isnil": IsNil,
}


def token_to_node(token: str, var_map) -> Node:
    if token in token_node_map:
        return token_node_map[token]()
    if token in var_map:
        return var_map[token]
    return Number(int(token))


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

    def evaluate_expression(self, expression: str):
        return self._evaluate_expression(expression.split())

    def _evaluate_expression(self, expression_tokens: typing.List[str]):
        root, _ = self._build(0, expression_tokens)
        return evaluate_all(root)

    def _build(self, i, tokens):
        if tokens[i] == "ap":
            func, i = self._build(i + 1, tokens)
            arg, i = self._build(i, tokens)
            return Ap(func, arg), i
        elif tokens[i] == "(":
            if tokens[i + 1] == ")":
                return Nil(), i + 1
            elem, i = self._build(i + 1, tokens)
            elements = [elem]
            while tokens[i] == ",":
                elem, i = self._build(i + 1, tokens)
                elements.append(elem)
            assert tokens[i + 1] == ")"

            return self._build_list(0, elements), i + 1
        else:
            return token_to_node(tokens[i], self.var_dict), i + 1

    def _build_list(self, i, elements):
        if i == len(elements):
            return Nil()
        return Ap(Ap(Cons(), elements[i], self._build_list(i + 1, elements)))


def evaluate_all(node: Node):
    while not node.is_leaf():
        node = node.evaluate()
    return node


if __name__ == '__main__':
    interpreter = Interpreter()
    for i, test_case in enumerate([
        ("ap inc 0", Number(1)),
        ("ap inc 1", Number(2)),
        ("ap dec 1", Number(0)),
        ("ap dec 0", Number(-1)),
        ("ap ap add 1 2", Number(3)),
        ("ap ap add 2 1", Number(3)),
        ("ap ap mul 3 4", Number(12)),
        ("ap ap mul 3 -2", Number(-6)),
        ("ap ap div 4 3", Number(1)),
        ("ap ap div 4 4", Number(1)),
        ("ap ap div 4 5", Number(0)),
        ("ap ap div 5 2", Number(2)),
        ("ap ap div 6 -2", Number(-3)),
        ("ap ap div 5 -3", Number(-1)),
        ("ap ap div -5 3", Number(-1)),
        ("ap ap div -5 -3", Number(1)),
        ("ap ap eq 0 -2", F()),
        ("ap ap eq 0 0", T()),
        ("ap ap lt 0 -1", F()),
        ("ap ap lt 0 0", F()),
        ("ap ap lt 0 1", T()),
        ("ap mod 0", ModulatedNumber("010")),
        ("ap mod 1", ModulatedNumber("01100001")),
        ("ap mod -1", ModulatedNumber("10100001")),
        ("ap mod 2", ModulatedNumber("01100010")),
        ("ap mod -2", ModulatedNumber("10100010")),
        ("ap mod 16", ModulatedNumber("0111000010000")),
        ("ap mod -16", ModulatedNumber("1011000010000")),
        ("ap mod 255", ModulatedNumber("0111011111111")),
        ("ap mod -255", ModulatedNumber("1011011111111")),
        ("ap mod 256", ModulatedNumber("011110000100000000")),
        ("ap mod -256", ModulatedNumber("101110000100000000")),
        ("ap dem ap mod 0", Number(0)),
        ("ap dem ap mod 12341234", Number(12341234)),
        ("ap dem ap mod -12341234", Number(-12341234)),
        ("ap ap ap s add inc 1", Number(3)),
        ("ap ap ap s mul ap add 1 6", Number(42)),
        ("ap ap ap c add 1 2", Number(3)),
        ("ap ap ap b inc dec 10", Number(10)),
        ("ap ap t 1 5", Number(1)),
            # ("ap ap t t i", T()), # TODO:
            # ("ap ap t t ap inc 5", T()), # TODO:
        ("ap ap t ap inc 5 t", Number(6)),
        ("ap ap f 1 2", Number(2)),
            # ("ap s t", F()),# TODO:
        ("ap pwr2 2", Number(4)),
        ("ap pwr2 3", Number(8)),
        ("ap pwr2 4", Number(16)),
            # TODO:
        ("ap nil 10", T()),
            # ("ap isnil nil", T()), # TODO:
        ("ap isnil ap ap cons 10 11", F()),
        ("( )", Nil()),
    ]):
        try:
            val = interpreter.evaluate_expression(test_case[0])
            if not val.equal(test_case[1]):
                print(
                    f"case {i}: `{test_case[0]}` expected {test_case[1]}, but {val}"
                )
        except Exception as e:
            print(f"case {i}: `{test_case[0]}` exception {e}")
            print(traceback.format_exc())

    print("test finished!")