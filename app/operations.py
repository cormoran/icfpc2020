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
class Environment:
    var_dict: typing.Dict = dataclasses.field(default_factory=dict)

    def get_variable(self, ident):
        return self.var_dict.get(ident, None)


@dataclasses.dataclass
class Node:

    # lazy evaluation
    def ap(self, env: Environment, arg):  # -> Node:
        raise Exception('cannot ap to', self.__class__.__name__)

    def evaluate(self, env: Environment):  # -> Node:
        return self

    # compare all subtree
    def __eq__(self, target):
        return False

    def __ne__(self, target):
        return not self.__eq__(target)

    def print(self, indent=0):
        return '\t' * indent + str(self)


@dataclasses.dataclass
class Number(Node):
    n: int

    def __eq__(self, target):
        if isinstance(target, int):
            return self.n == target
        return isinstance(target, Number) and target.n == self.n

    def print(self, indent=0):
        return '\t' * indent + str(self.n)


@dataclasses.dataclass
class Variable(Node):
    ident: str
    args: typing.List[Node] = dataclasses.field(default_factory=list)

    def ap(self, env: Environment, arg: Node) -> Node:
        return Variable(self.ident, self.args + [arg])

    def evaluate(self, env: Environment) -> Node:
        node = env.get_variable(self.ident)
        if node is None:
            # 自由変数
            return self
        for arg in self.args:
            node = Ap(node, arg)
        return node

    def __eq__(self, target):
        # TODO: get real value from var dict
        if not isinstance(target, Variable):
            return False
        if target.ident != self.ident:
            return False
        if len(self.args) != len(target.args):
            return False
        for a, b in zip(self.args, target.args):
            if a != b:
                return False
        return True

    def print(self, indent=0):
        return '\t' * indent + 'Var(' + self.ident + ')'


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

    def __eq__(self, target):
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

    def print(self, indent=0):
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
        res = '\t' * indent + 'Picture(\n'
        for line in canvas:
            res += '\t' * indent + ''.join(line) + '\n'
        res += '\t' * indent + ')'


@dataclasses.dataclass
class Modulated(Node):
    n: str

    def __eq__(self, target):
        return isinstance(target, Modulated) and target.n == self.n


@dataclasses.dataclass
class NArgOp(Node):
    args: typing.List[Node] = dataclasses.field(default_factory=list)

    # n_args: int # abstract field...

    def ap(self, env: Environment, arg: Node) -> Node:
        return self.__class__(self.args + [arg])
        # if len(self.args) < self.n_args - 1:
        #     return self.__class__(self.args + [arg])
        # else:
        #     return self.__class__(self.args + [arg]).evaluate(env)

    def evaluate(self, env: Environment) -> Node:
        if len(self.args) == self.n_args:
            return self._evaluate(env)
        return self  # cannot evaluate now

    def _evaluate(self, env: Environment) -> Node:
        raise NotImplementedError()

    # compare all subtree
    def __eq__(self, target):
        if self.__class__ != target.__class__:
            return False
        if len(self.args) != len(target.args):
            return False
        for a, b in zip(self.args, target.args):
            if a != b:
                return False
        return True

    def print(self, indent=0):
        res = '\t' * indent + self.__class__.__name__ + '(\n'
        for arg in self.args:
            res += arg.print(indent=indent + 1) + '\n'
        res += '\t' * indent + ')'
        return res


@dataclasses.dataclass
class Inc(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # n = evaluate_to_type(env, self.args[0], Number)
        n = self.args[0].evaluate(env)
        if isinstance(n, Number):
            return Number(n.n + 1)
        return Inc([n])


@dataclasses.dataclass
class Dec(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # n = evaluate_to_type(env, self.args[0], Number)
        n = self.args[0].evaluate(env)
        if isinstance(n, Number):
            return Number(n.n - 1)
        return Dec([n])


@dataclasses.dataclass
class Add(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        # n1 = evaluate_to_type(env, self.args[0], Number)
        # n2 = evaluate_to_type(env, self.args[1], Number)
        n1 = self.args[0].evaluate(env)
        n2 = self.args[1].evaluate(env)
        if isinstance(n1, Number) and isinstance(n2, Number):
            return Number(n1.n + n2.n)
        if isinstance(n1, Number) and n1.n == 0:
            return n2
        if isinstance(n2, Number) and n2.n == 0:
            return n1
        return Add([n1, n2])

    # def __eq__(self, target):
    #     if self.__class__ != target.__class__:
    #         return False
    #     return set(self.args) == set(target.args)


@dataclasses.dataclass
class Mul(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        # n1 = evaluate_to_type(env, self.args[0], Number)
        #n2 = evaluate_to_type(env, self.args[1], Number)
        n1 = self.args[0].evaluate(env)
        n2 = self.args[1].evaluate(env)
        if isinstance(n1, Number) and isinstance(n2, Number):
            return Number(n1.n * n2.n)
        if isinstance(n1, Number) and n1.n == 0:
            return Number(0)
        if isinstance(n2, Number) and n2.n == 0:
            return Number(0)
        if isinstance(n1, Number) and n1.n == 1:
            return n2
        if isinstance(n2, Number) and n2.n == 1:
            return n1
        return Mul([n1, n2])

    # def __eq__(self, target):
    #     if self.__class__ != target.__class__:
    #         return False
    #     return set(self.args) == set(target.args)


@dataclasses.dataclass
class Div(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        # n1 = evaluate_to_type(env, self.args[0], Number)
        # n2 = evaluate_to_type(env, self.args[1], Number)
        n1 = self.args[0].evaluate(env)
        n2 = self.args[1].evaluate(env)
        if isinstance(n1, Number) and isinstance(n2, Number):
            sign = 1 if n1.n * n2.n >= 0 else -1
            return Number(abs(n1.n) // abs(n2.n) * sign)
        return Div([n1, n2])


@dataclasses.dataclass
class Eq(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        n1 = self.args[0].evaluate(env)
        n2 = self.args[1].evaluate(env)
        return T() if n1 == n2 else F()


@dataclasses.dataclass
class Lt(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        # n1 = evaluate_to_type(env, self.args[0], Number)
        # n2 = evaluate_to_type(env, self.args[1], Number)
        n1 = self.args[0].evaluate(env)
        n2 = self.args[1].evaluate(env)
        if isinstance(n1, Number) and isinstance(n2, Number):
            return T() if n1.n < n2.n else F()
        return Lt([n1, n2])


@dataclasses.dataclass
class Modulate(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # n = evaluate_to_type(env, self.args[0], [Number, Cons, Nil])
        n = self.args[0].evaluate(env)
        if isinstance(n, (Number, Cons, Nil)):
            res = self._modulate_node(env, n)
            return Modulated(res) if isinstance(res, str) else Modulate([res])
        return Modulate([n])

    def _modulate_node(self, env: Environment,
                       node: Node) -> typing.Union[Node, str]:
        if isinstance(node, Number):
            return modulate.modulate_number(node.n)
        elif isinstance(node, Nil):
            return "00"
        elif isinstance(node, Cons):
            return self._evaluate_cons(env, node)
        else:
            raise Exception(f"modulate: unsupported Node {node}")

    def _evaluate_cons(self, env: Environment,
                       cons: Node) -> typing.Union[Node, str]:
        # n1 = evaluate_to_type(env, cons.args[0], [Number, Cons, Nil])
        # n2 = evaluate_to_type(env, cons.args[1], [Number, Cons, Nil])
        n1 = cons.args[0].evaluate(env)
        n2 = cons.args[1].evaluate(env)
        if isinstance(n1, (Number, Cons, Nil)) and isinstance(
                n2, (Number, Cons, Nil)):
            return "11" + self._modulate_node(env, n1) + self._modulate_node(
                env, n2)
        return Cons([n1, n2])


@dataclasses.dataclass
class Demodulate(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # n = evaluate_to_type(env, self.args[0], Modulated)
        n = self.args[0].evaluate(env)
        if isinstance(n, Modulated):
            node, left = self._demodulate(n.n)
            if len(left) != 0:
                print(left)
                assert len(left) == 0
            return node
        return Demodulate([n])

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

    def _evaluate(self, env: Environment) -> Node:
        n = Ap(Modulate(), self.args[0]).evaluate(env)
        if isinstance(n, Modulated):
            print('* [Human -> Alien]', n.n)
            res = requests.post(server_url + '/aliens/send' + query_param, n.n)
            if res.status_code != 200:
                print('Unexpected server response:')
                print('HTTP code:', res.status_code)
                print('Response body:', res.text)
                raise Exception('Unexpected server response:')
            print('* [Alien -> Human]', res.text)
            return Ap(Demodulate(), Modulated(res.text)).evaluate(env)
        return Send([n])


@dataclasses.dataclass
class Neg(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # n = evaluate_to_type(env, self.args[0], Number)
        n = self.args[0].evaluate(env)
        if isinstance(n, Number):
            return Number(-n.n)
        return Neg([n])


@dataclasses.dataclass
class Ap(Node):
    func: typing.Optional[Node]
    arg: typing.Optional[Node]

    # def ap(self, env: Environment, arg: Node):
    #     # TODO
    #     return self.evaluate(env).ap(env, arg)

    # return func or value
    def evaluate(self, env: Environment) -> Node:
        func = self.func
        while isinstance(func, Ap):
            func = func.evaluate(env)
        return func.ap(env, self.arg).evaluate(env)

    # compare all subtree
    def __eq__(self, target):
        if not isinstance(target, Ap):
            return False
        return self.func == target.func and self.arg == target.arg

    def print(self, indent=0):
        res = '\t' * indent + 'Ap(\n'
        res += '\t' * indent + 'func\n'
        res += self.func.print(indent=indent + 1)
        res += '\t' * indent + 'arg\n'
        res += self.arg.print(indent=indent + 1)
        res += '\t' * indent + ')'


@dataclasses.dataclass
class S(NArgOp):
    n_args = 3

    def _evaluate(self, env: Environment) -> Node:
        a = Ap(self.args[0], self.args[2])
        b = Ap(self.args[1], self.args[2])
        return Ap(a, b).evaluate(env)


@dataclasses.dataclass
class C(NArgOp):
    n_args = 3

    def _evaluate(self, env: Environment) -> Node:
        a = Ap(self.args[0], self.args[2])
        return Ap(a, self.args[1]).evaluate(env)


@dataclasses.dataclass
class B(NArgOp):
    n_args = 3

    def _evaluate(self, env: Environment) -> Node:
        a = Ap(self.args[1], self.args[2])
        return Ap(self.args[0], a).evaluate(env)


@dataclasses.dataclass
class T(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        return self.args[0].evaluate(env)

    def __eq__(self, target):
        if isinstance(target, T) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().__eq__(target)

    def print(self, indent=0):
        return '\t' * indent + 'True'


@dataclasses.dataclass
class F(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
        return self.args[1].evaluate(env)

    def __eq__(self, target):
        if isinstance(target, F) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().__eq__(target)

    def print(self, indent=0):
        return '\t' * indent + 'False'


@dataclasses.dataclass
class Pwr2(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # n = evaluate_to_type(env, self.args[0], Number)
        n = self.args[0].evaluate(env)
        if isinstance(n, Number):
            return Number(2**n.n)
        return Pwr2([n])


@dataclasses.dataclass
class I(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        return self.args[0].evaluate(env)


@dataclasses.dataclass
class Cons(NArgOp):
    n_args = 3

    def _evaluate(self, env: Environment) -> Node:
        a = Ap(self.args[2], self.args[0])
        return Ap(a, self.args[1]).evaluate(env)

    def print(self, indent=0):
        if len(self.args) < 2:
            return super().print(indent)
        res = '\t' * indent + '('
        cons = self
        while isinstance(cons, Cons) and len(cons.args) == 2:
            res += cons.args[0].print() + ' , '
            cons = cons.args[1]
        if not isinstance(cons, Nil):
            res += cons.print()
        else:
            res = res[:-3]  # remove ` , `
        res += ')'
        return res


@dataclasses.dataclass
class Car(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # arg = evaluate_to_type(env, self.args[0], Cons)
        arg = self.args[0].evaluate(env)
        if isinstance(arg, Cons):
            return arg.args[0].evaluate(env)
        return Car([arg])


@dataclasses.dataclass
class Cdr(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        arg = self.args[0].evaluate(env)
        if isinstance(arg, Cons):
            return arg.args[1].evaluate(env)
        return Cdr([arg])


@dataclasses.dataclass
class Nil(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        if len(self.args) == 0:
            return self
        return T()

    def __eq__(self, target):
        if isinstance(target, Nil) and len(self.args) == 0 and len(
                target.args) == 0:
            return True
        return super().__eq__(target)

    def print(self, indent=0):
        return '\t' * indent + 'Nil'


@dataclasses.dataclass
class IsNil(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        n = self.args[0].evaluate(env)
        return T() if isinstance(n, Nil) else F()


@dataclasses.dataclass
class Draw(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        # TODO: support variable
        picture = Picture()
        arg = self.args[0].evaluate(env)
        while isinstance(arg, Cons):
            pair = arg.args[0].evaluate(env)
            x = pair.args[0].evaluate(env)
            y = pair.args[1].evaluate(env)
            # TODO
            assert isinstance(x, Number) and isinstance(y, Number)
            picture.add_point(x.n, y.n)
            arg = arg.args[1].evaluate(env)
            # arg = evaluate_to_type(env, arg.args[1], [Cons, Nil])
        assert isinstance(arg, Nil)
        return picture


@dataclasses.dataclass
class MultipleDraw(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        n = self.args[0].evaluate(env)
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

    def _evaluate(self, env: Environment) -> Node:
        # condition = evaluate_to_type(env, self.args[0],
        #                              Number)  # TODO: 実は number でなくてもよいかも
        condition = self.args[0].evaluate(env)
        if isinstance(condition, Number):
            return self.args[1 if condition.n == 0 else 2]
        return If0(self.args.copy())


@dataclasses.dataclass
class Modem(NArgOp):
    n_args = 1

    def _evaluate(self, env: Environment) -> Node:
        return Ap(Demodulate(), Ap(Modulate(), self.args[0])).evaluate(env)


@dataclasses.dataclass
class F38(NArgOp):
    n_args = 2

    # "ap ap f38 x2 x0 = ap ap ap if0 ap car x0 ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 ) ap ap ap interact x2 ap modem ap car ap cdr x0 ap send ap car ap cdr ap cdr x0"
    def _evaluate(self, env: Environment) -> Node:
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
                  a).evaluate(env)
        # yapf: enable


@dataclasses.dataclass
class Interact(NArgOp):
    n_args = 3

    # "ap ap ap interact x2 x4 x3 = ap ap f38 x2 ap ap x2 x4 x3"
    def _evaluate(self, env: Environment) -> Node:
        x2 = self.args[0]  # protocol: function (state, vector) -> value
        x4 = self.args[1]  # state
        x3 = self.args[2]  # vector

        return Ap(Ap(F38(), x2), Ap(Ap(x2, x4), x3)).evaluate(env)


@dataclasses.dataclass
class StatelessDraw(NArgOp):
    n_args = 2

    # ap ap c ap ap b b ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c ap ap b cons ap ap c cons nil nil
    def _evaluate(self, env: Environment) -> Node:
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
                  Nil())).evaluate(env)
        # yapf: enable
        return x


@dataclasses.dataclass
class StatefullDraw(NArgOp):
    n_args = 2

    def _evaluate(self, env: Environment) -> Node:
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