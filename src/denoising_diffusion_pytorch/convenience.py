import math
from functools import wraps
from einops import pack, unpack


def exists(x):
    return x is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def xnor(x, y):
    return not (x ^ y)


def append(arr, el):
    arr.append(el)


def prepend(arr, el):
    arr.insert(0, el)


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]
