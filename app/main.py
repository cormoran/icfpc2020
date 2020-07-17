import requests
import sys
import math


def umodulate(x: str) -> int:
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


def modulate_list(x: str):
    return "11" + xx + "00"


def modulate(x: int) -> str:
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


def test_modulate():
    assert modulate(0) == "010"
    assert modulate(1) == "01100001"
    assert modulate(-1) == "10100001"
    assert modulate(2) == "01100010"
    assert modulate(-2) == "10100010"
    assert modulate(16) == "0111000010000"
    assert modulate(-16) == "1011000010000"
    assert modulate(255) == "0111011111111"
    assert modulate(-255) == "1011011111111"
    assert modulate(256) == "011110000100000000"
    assert modulate(-256) == "101110000100000000"
    print("test_modulate ok")
    print(modulate(42))


def test_umodulate():
    for i in range(0, 100000, 100):
        assert umodulate(modulate(i)) == i
    print("test_umodulate ok")


def ensure_no_error(response: requests.Response) -> requests.Response:
    if response.status_code != 200:
        print('Unexpected server response:')
        print('HTTP code:', response.status_code)
        print('Response body:', response.text)
        exit(2)
    return response


def main():
    server_url = sys.argv[1]
    player_key = sys.argv[2]
    print('ServerUrl: %s; PlayerKey: %s' % (server_url, player_key))

    data = modulate_list(modulate(0))
    print('Send ', data)
    res = ensure_no_error(requests.post(server_url + '/aliens/send',
                                        data=data))
    print('Server response:', res.text)


if __name__ == '__main__':
    main()
