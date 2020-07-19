import typing
import dataclasses
import math
import requests
import traceback
import os

server_url = "https://icfpc2020-api.testkontur.ru"
query_param = "?apiKey=" + os.environ.get("API_KEY", "")

import modulate


def set_parameter(a_server_url, a_query_param):
    global server_url
    global query_param
    server_url = a_server_url
    query_param = a_query_param


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
        if isinstance(target, int):
            return self.n == target
        return isinstance(target, Number) and target.n == self.n


@dataclasses.dataclass
class Point:
    x: int
    y: int

    def __lt__(self, other):
        return self.x < other.x if self.x != other.x else self.y < other.y


@dataclasses.dataclass
class Picture(Node):
    points: typing.List[Point] = dataclasses.field(default_factory=list)

    def add_point(self, x: int, y: int):
        self.points.append(Point(x, y))

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        if not isinstance(target, Picture):
            return False
        if len(self.points) != len(target.points):
            return False
        # TODO: 破壊的変更をしている
        self.points = sorted(self.points)
        target.points = sorted(target.points)
        for p1, p2 in zip(self.points, target.points):
            if p1.x != p2.x or p1.y != p2.y:
                return False
        return True

    def print(self):
        min_x, max_x = 0, 0
        min_y, max_y = 0, 0
        for p in self.points:
            min_x = min(min_x, p.x)
            max_x = max(max_x, p.x)
            min_y = min(min_y, p.y)
            max_y = max(max_y, p.y)
        canvas = [['.' for x in range(max_x - min_x + 1)]
                  for y in range(max_y - min_y + 1)]
        for p in self.points:
            canvas[p.y][p.x] = '#'
        for line in canvas:
            print()
            print(''.join(line))
            print()


@dataclasses.dataclass
class Modulated(Node):
    n: str

    def evaluate(self) -> Node:
        return self

    def equal(self, target):
        return isinstance(target, Modulated) and target.n == self.n


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
            return modulate.modulate_number(node.n)
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
            n, xx = modulate.demodulate_number(xx)
            return Number(n), xx


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
        picture = Picture()
        arg = evaluate_to_type(self.args[0], [Cons, Nil])
        while isinstance(arg, Cons):
            pair = evaluate_to_type(arg.args[0], [Cons, Nil])
            x = evaluate_to_type(pair.args[0], Number)
            y = evaluate_to_type(pair.args[1], Number)
            picture.add_point(x.n, y.n)
            arg = evaluate_to_type(arg.args[1], [Cons, Nil])
        return picture


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


# elements: [1, 2, 3]
# returns: Cons(1, Cons(2, Cons(3, nil)))
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