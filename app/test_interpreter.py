import typing
import traceback
import os

from operations import *
from interpreter import Interpreter

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
        ("ap draw ( ap ap vec 1 1 )", Picture([Point(1, 1)])),
        ("ap draw ( ap ap vec 1 2 )", Picture([Point(1, 2)])),
        ("ap draw ( ap ap vec 1 5 )", Picture([Point(1, 5)])),
        ("ap draw ( ap ap vec 1 1 , ap ap vec 3 1 )",
         Picture([Point(1, 1), Point(3, 1)])),
            # 34 Multiple Draw
        ("ap multipledraw nil", Nil()),
        ("ap multipledraw ( ( ap ap vec 1 2 ) , ( ap ap vec 3 4 ) )",
         make_list([Picture([Point(1, 2)]),
                    Picture([Point(3, 4)])])),
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
         make_list([Nil(), Picture([Point(1, 0)])])),
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
