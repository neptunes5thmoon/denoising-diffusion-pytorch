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
