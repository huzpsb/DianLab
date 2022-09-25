def _gcd(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    if a >= b:
        return _gcd(a % b, b)
    return _gcd(a, b % a)


def gcd(*args):
    if len(args) == 0:
        return -1
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return _gcd(args[0], args[1])
    now_gcd = _gcd(args[0], args[1])
    for now in range(2, len(args)):
        now_gcd = _gcd(now_gcd, args[now])
    return now_gcd


def nop(*args):
    pass


print(gcd(114514, 1919810, 1))
print(gcd(6, 4, 12))
print(gcd(15, 20))
print(gcd(666))
print(gcd())
nop()
