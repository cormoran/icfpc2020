import typing
import math


def modulate_number(x: int) -> str:
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


# returns tuple of (demodulated number, left string)
def demodulate_number(x: str) -> typing.Tuple[int, str]:
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
