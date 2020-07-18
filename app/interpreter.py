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


def ensure_type(node: Node, t: typing.ClassVar) -> Node:
    if not isinstance(node, t):
        raise Exception(
            f"ensure type: expected {t}, but {node.__class__.__name__}")
    return node


# evaluate `node` until it's type become `t`
def evaluate_to_type(node: Node, t: typing.Union[type,
                                                 typing.List[type]]) -> Node:
    if isinstance(t, list):
        t = tuple(t)
    prev = None
    now = node
    while not isinstance(now, t):
        prev = now
        now = now.evaluate()
        if now == prev:
            raise Exception(
                f"infinite loop detected in evaluate_to_type({node}, {t})")
    return now


@dataclasses.dataclass
class Number(Node):
    n: int

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        return isinstance(target, Number) and target.n == self.n


@dataclasses.dataclass
class Picture(Node):
    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    points: typing.List[Point] = dataclasses.field(default_factory=list)

    def add_point(self, x, y):
        self.points.append(self.Point(x, y))

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        return False


@dataclasses.dataclass
class Modulated(Node):
    n: str

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        return isinstance(target, Modulated) and target.n == self.n


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
        if self.__class__ != target.__class__:
            return False
        if len(self.args) != len(target.args):
            return False
        for a, b in zip(self.args, target.args):
            if not a.equal(b):
                return False
        return True


@dataclasses.dataclass
class Inc(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = evaluate_to_type(self.args[0], Number)
        return Number(n.n + 1)


@dataclasses.dataclass
class Dec(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = evaluate_to_type(self.args[0], Number)
        return Number(n.n - 1)


@dataclasses.dataclass
class Add(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = evaluate_to_type(self.args[0], Number)
        n2 = evaluate_to_type(self.args[1], Number)
        return Number(n1.n + n2.n)


@dataclasses.dataclass
class Mul(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = evaluate_to_type(self.args[0], Number)
        n2 = evaluate_to_type(self.args[1], Number)
        return Number(n1.n * n2.n)


@dataclasses.dataclass
class Div(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = evaluate_to_type(self.args[0], Number)
        n2 = evaluate_to_type(self.args[1], Number)
        sign = 1 if n1.n * n2.n >= 0 else -1
        return Number(abs(n1.n) // abs(n2.n) * sign)


@dataclasses.dataclass
class Eq(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = self.args[0]
        n2 = self.args[1]
        return T() if n1.equal(n2) else F()


@dataclasses.dataclass
class Lt(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        n1 = evaluate_to_type(self.args[0], Number)
        n2 = evaluate_to_type(self.args[1], Number)
        return T() if n1.n < n2.n else F()


@dataclasses.dataclass
class Modulate(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = evaluate_to_type(self.args[0], [Number, Cons, Nil])
        return Modulated(self._modulate_node(n))

    def _modulate_node(self, node: Node):
        if isinstance(node, Number):
            return self.modulate_number(node.n)
        elif isinstance(node, Nil):
            return "00"
        elif isinstance(node, Cons):
            return self._evaluate_cons(node)
        else:
            raise Exception(f"modulate: unsupported Node {node}")

    def _evaluate_cons(self, cons: Node):
        n1 = evaluate_to_type(cons.args[0], [Number, Cons, Nil])
        n2 = evaluate_to_type(cons.args[1], [Number, Cons, Nil])
        return "11" + self._modulate_node(n1) + self._modulate_node(n2)

    def modulate_number(self, x: int) -> str:
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
        n = evaluate_to_type(self.args[0], Modulated)
        node, left = self._demodulate(n.n)
        if len(left) != 0:
            print(left)
            assert len(left) == 0
        return node

    def _demodulate(self, x: str) -> Node:
        xx = x
        if xx.startswith("11"):
            xx = xx[2:]
            a, xx = self._demodulate(xx)
            b, xx = self._demodulate(xx)
            return Cons([a, b]), xx
        elif xx.startswith("00"):
            xx = xx[2:]
            return Nil(), xx
        else:
            n, xx = self.demodulate_number(xx)
            return Number(n), xx

    # -> number, left str
    def demodulate_number(self, x: str) -> typing.Tuple[int, str]:
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
        bit_length *= 4
        left = x[bit_length:]
        x = x[:bit_length]
        x = x[::-1]
        num = 0
        for i in range(len(x)):
            if x[i] == "1":
                num += 2**i
        return num * signal, left


@dataclasses.dataclass
class Send(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = Ap(Modulate(), self.args[0])
        n = evaluate_to_type(n, Modulated)
        print('* [Human -> Alien]', n.n)
        res = requests.post(server_url + '/aliens/send' + query_param, n.n)
        if res.status_code != 200:
            print('Unexpected server response:')
            print('HTTP code:', res.status_code)
            print('Response body:', res.text)
            raise Exception('Unexpected server response:')
        print('* [Alien -> Human]', res.text)
        return Ap(Demodulate(), Modulated(res.text)).evaluate()


@dataclasses.dataclass
class Neg(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = evaluate_to_type(self.args[0], Number)
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

    # compare all subtree
    def equal(self, target):
        if not isinstance(target, Ap):
            return False
        return self.func.equal(target.func) and self.arg.equal(target.arg)


@dataclasses.dataclass
class S(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        # TODO: 知らんぞ
        a = Ap(self.args[0], self.args[2])
        b = Ap(self.args[1], self.args[2])
        return Ap(a, b)


@dataclasses.dataclass
class C(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        a = Ap(self.args[0], self.args[2])
        return Ap(a, self.args[1])


@dataclasses.dataclass
class B(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        # TODO: 知らんぞ
        a = Ap(self.args[1], self.args[2])
        return Ap(self.args[0], a)


@dataclasses.dataclass
class T(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        return self.args[0]

    def equal(self, target):
        if isinstance(target, T) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().equal(target)


@dataclasses.dataclass
class F(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        return self.args[1]

    def equal(self, target):
        if isinstance(target, F) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().equal(target)


@dataclasses.dataclass
class Pwr2(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = evaluate_to_type(self.args[0], Number)
        return Number(2**n.n)


@dataclasses.dataclass
class I(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        return self.args[0]


@dataclasses.dataclass
class Cons(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        a = Ap(self.args[2], self.args[0])
        return Ap(a, self.args[1])


@dataclasses.dataclass
class Car(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        arg = evaluate_to_type(self.args[0], Cons)
        assert len(arg.args) <= 2
        return arg.args[0]  # TODO: evaluate?


@dataclasses.dataclass
class Cdr(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        arg = evaluate_to_type(self.args[0], Cons)
        assert len(arg.args) == 2
        return arg.args[1]  # TODO: evaluate?


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


@dataclasses.dataclass
class IsNil(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        # TODO: これでいいのか？
        n = self.args[0]
        while isinstance(n, Ap):
            n = n.evaluate()
        return T() if isinstance(n, Nil) else F()


@dataclasses.dataclass
class Draw(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        arg = evaluate_to_type(self.args[0], [Cons, Nil])
        picture = Picture()
        # TODO: add points
        return Picture()


@dataclasses.dataclass
class MultipleDraw(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        n = evaluate_to_type(self.args[0], [Nil, Cons])
        if isinstance(n, Nil):
            return n
        elif isinstance(n, Cons):
            return Ap(Ap(Cons(), Ap(Draw(), n.args[0])),
                      Ap(MultipleDraw(), n.args[1]))
        else:
            raise Exception(f"unknown type node {n}")


@dataclasses.dataclass
class If0(NArgOp):
    n_args = 3

    def _evaluate(self) -> Node:
        condition = evaluate_to_type(self.args[0],
                                     Number)  # TODO: 実は number でなくてもよいかも
        return self.args[1 if condition.n == 0 else 2]


@dataclasses.dataclass
class Modem(NArgOp):
    n_args = 1

    def _evaluate(self) -> Node:
        return Ap(Demodulate(), Ap(Modulate(), self.args[0]))


@dataclasses.dataclass
class F38(NArgOp):
    n_args = 2

    # "ap ap f38 x2 x0 = ap ap ap if0 ap car x0 ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 ) ap ap ap interact x2 ap modem ap car ap cdr x0 ap send ap car ap cdr ap cdr x0"
    def _evaluate(self) -> Node:
        x2 = self.args[0]  # protocol
        x0 = self.args[1]  # (flag, newState, data)
        # ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 )
        list1 = Cons([
            Ap(Modem(), Ap(Car(), Ap(Cdr(), x0))),
            Ap(MultipleDraw(), Ap(Car(), Ap(Cdr(), Ap(Cdr(), x0)))),
        ])
        # yapf: disable
        a = Ap(Ap(Ap(Interact, x2),
                  Ap(Modem(), Ap(Car(),
                                 Ap(Cdr(), x0)))),
               Ap(Send(),
                  Ap(Car(),
                     Ap(Cdr(),
                        Ap(Cdr(), x0)))))

        return Ap(Ap(Ap(If0(),
                        Ap(Car(), x0)),
                     list1),
                  a)
        # yapf: enable


@dataclasses.dataclass
class Interact(NArgOp):
    n_args = 3

    # "ap ap ap interact x2 x4 x3 = ap ap f38 x2 ap ap x2 x4 x3"
    def _evaluate(self) -> Node:
        x2 = self.args[0]  # protocol: function (state, vector) -> value
        x4 = self.args[1]  # state
        x3 = self.args[2]  # vector

        return Ap(Ap(F38(), x2), Ap(Ap(x2, x4), x3))


@dataclasses.dataclass
class StatelessDraw(NArgOp):
    n_args = 2

    # ap ap c ap ap b b ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c ap ap b cons ap ap c cons nil nil
    def _evaluate(self) -> Node:
        if isinstance(self.args[0], Nil):
            return make_list(
                [Number(0),
                 Nil(),
                 make_list([make_list([self.args[1]])])])
        # yapf: disable
        x = Ap(Ap(C(),
                  Ap(Ap(B(), B()),
                     Ap(Ap(B(),
                           Ap(B(),
                              Ap(Cons(), Number(0)))),
                        Ap(Ap(C(), Ap(Ap(B(), B()), Cons())),
                           Ap(Ap(C(), Cons()), Nil()))))),
               Ap(Ap(C(),
                     Ap(Ap(B(), Cons()),
                        Ap(Ap(C(), Cons()),
                           Nil()))),
                  Nil()))
        # yapf: enable
        return x


@dataclasses.dataclass
class StatefullDraw(NArgOp):
    n_args = 2

    def _evaluate(self) -> Node:
        # ap ap statefulldraw x0 x1 = ( 0 , ap ap cons x1 x0 , ( ap ap cons x1 x0 ) )
        # statefulldraw = ap ap b ap b ap ap s ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c cons nil ap c cons
        raise NotImplementedError()


def make_list(elements: typing.List):
    if len(elements) == 0:
        return Nil()
    return Cons([elements[0], make_list(elements[1:])])


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
    "vec": Cons,
    "draw": Draw,
    "multipledraw": MultipleDraw,
    "if0": If0,
    "modem": Modem,
    "f38": F38,
    "interact": Interact,
    "statelessdraw": StatelessDraw,
    "statefulldraw": StatefullDraw,
}


def evaluate_all_ap(node: Node):
    while isinstance(node, Ap):
        node = node.evaluate()
    if isinstance(node, NArgOp):
        node.args = list(map(evaluate_all_ap, node.args))
    return node


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
        return evaluate_all_ap(root)

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
            assert tokens[i] == ")"

            return self._build_list(0, elements), i + 1
        else:
            return token_to_node(tokens[i], self.var_dict), i + 1

    # (1, 2, ...) -> ap ap cons 1 ap ap cons 2, ..., nil
    def _build_list(self, i, elements):
        if i == len(elements):
            return Nil()
        return Ap(Ap(Cons(), elements[i]), self._build_list(i + 1, elements))


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
        ("ap mod 0", Modulated("010")),
        ("ap mod 1", Modulated("01100001")),
        ("ap mod -1", Modulated("10100001")),
        ("ap mod 2", Modulated("01100010")),
        ("ap mod -2", Modulated("10100010")),
        ("ap mod 16", Modulated("0111000010000")),
        ("ap mod -16", Modulated("1011000010000")),
        ("ap mod 255", Modulated("0111011111111")),
        ("ap mod -255", Modulated("1011011111111")),
        ("ap mod 256", Modulated("011110000100000000")),
        ("ap mod -256", Modulated("101110000100000000")),
        ("ap dem ap mod 0", Number(0)),
        ("ap dem ap mod 12341234", Number(12341234)),
        ("ap dem ap mod -12341234", Number(-12341234)),
        ("ap ap ap s add inc 1", Number(3)),
        ("ap ap ap s mul ap add 1 6", Number(42)),
        ("ap ap ap c add 1 2", Number(3)),
        ("ap ap ap b inc dec 10", Number(10)),
        ("ap ap t 1 5", Number(1)),
        ("ap ap t t i", T()),
        ("ap ap t t ap inc 5", T()),
        ("ap ap t ap inc 5 t", Number(6)),
        ("ap ap f 1 2", Number(2)),
            # ("ap s t", F()),
        ("ap pwr2 2", Number(4)),
        ("ap pwr2 3", Number(8)),
        ("ap pwr2 4", Number(16)),
        ("ap i 10", Number(10)),
        ("ap i i", I()),
        ("ap i add", Add()),
            # ("ap ap ap cons x0 x1 x2   =   ap ap x2 x0 x1")
        ("ap ap ap cons 10 11 add", Number(21)),
        ("ap car ap ap cons 10 11", Number(10)),
        ("ap cdr ap ap cons 10 11", Number(11)),
            # ("ap cdr x2   =   ap x2 f")
        ("ap nil 10", T()),
        ("ap isnil nil", T()),  # TODO:
        ("ap isnil ap ap cons 10 11", F()),
        ("( )", Nil()),
        ("( 10 )", Cons([Number(10), Nil()])),
        ("( 10 , 11 )", Cons([Number(10),
                              Cons([Number(11), Nil()])])),
            # 32 Draw
        ("ap draw ( )", Picture()),
        ("ap draw ( ap ap vec 1 1 )", Picture()),
        ("ap draw ( ap ap vec 1 2 )", Picture()),
        ("ap draw ( ap ap vec 1 5 )", Picture()),
        ("ap draw ( ap ap vec 1 1 , ap ap vec 3 1 )", Picture()),
            # 34 Multiple Draw
        ("ap multipledraw nil", Nil()),
        ("ap multipledraw ap ap cons ap ap vec 1 1 ap ap cons ap ap vec 2 2 nil",
         make_list([Picture(), Picture()])),
            # 35 Modulate List
        ("ap mod nil", Modulated("00")),
        ("ap mod ap ap cons nil nil", Modulated("110000")),
        ("ap mod ap ap cons 0 nil", Modulated("1101000")),
        ("ap mod ap ap cons 1 2", Modulated("110110000101100010")),
        ("ap mod ap ap cons 1 ap ap cons 2 nil",
         Modulated("1101100001110110001000")),
        ("ap mod ( 1 , 2 )", Modulated("1101100001110110001000")),
        ("ap mod ( 1 , ( 2 , 3 ) , 4 )",
         Modulated("1101100001111101100010110110001100110110010000")),
            # 36 Send ( 0 ) # TODO: raise error everytime, because the number decrease by time...
            # ("ap send ( 0 )", Cons([Number(1), Cons([Number(0), Nil()])])),
            # 37 Is 0
        ("ap ap ap if0 0 10 11", Number(10)),
        ("ap ap ap if0 1 10 11", Number(11)),
            # 38 Interact
        ("ap modem 10", Number(10)),
            # "ap ap f38 x2 x0 = ap ap ap if0 ap car x0 ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 ) ap ap ap interact x2 ap modem ap car ap cdr x0 ap send ap car ap cdr ap cdr x0"
            # "ap ap ap interact x2 x4 x3 = ap ap f38 x2 ap ap x2 x4 x3"
            # 39 Interaction Protocol
            # ("ap ap ap interact x0 nil ap ap vec 0 0", Nil()), # TODO: x0 need to be function
        ("ap ap statelessdraw nil 10",
         make_list([Number(0),
                    Nil(),
                    make_list([make_list([Number(10)])])])),
            # 40 Stateless Drawing Protocol
            # TODO: ( nil , ( [1,0] ) ) になってほしいけど ( ) が少ない気がする...
        ("ap ap ap interact statelessdraw nil ap ap vec 1 0",
         make_list([Nil(), Picture()])),
            # 41 Statefull Drawing Protocol
            # ap ap ap interact statefulldraw nil ap ap vec 0 0 = ( ( ap ap vec 0 0 ) , ( [0,0] ) )
            # ap ap ap interact statefulldraw ( ap ap vec 0 0 ) ap ap vec 2 3 = ( x2 , ( [0,0;2,3] ) )
            # ap ap ap interact statefulldraw x2 ap ap vec 1 2 = ( x3 , ( [0,0;2,3;1,2] ) )
            # ap ap ap interact statefulldraw x3 ap ap vec 3 2 = ( x4 , ( [0,0;2,3;1,2;3,2] ) )
            # ap ap ap interact statefulldraw x4 ap ap vec 4 0 = ( x5 , ( [0,0;2,3;1,2;3,2;4,0] ) )
    ]):
        try:
            val = interpreter.evaluate_expression(test_case[0])
            if not val.equal(test_case[1]):
                print(
                    f"case {i}: `{test_case[0]}`\n\texpected {test_case[1]}\n\tbut      {val}"
                )
        except Exception as e:
            print(f"case {i}: `{test_case[0]}` exception {e}")
            print(traceback.format_exc())

    print("test finished!")
