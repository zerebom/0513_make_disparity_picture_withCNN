def gen_1():
    for i in range(4):
        yield '<< %s' % i


def gen_2():
    for i in range(3, 0, -1):
        yield '<< %s' % i


def zip_func(gen_1, gen_2):
    for i, j in zip(gen_1(), gen_2()):
        yield i, j
    # yield from (gen_1, gen_2)
    #↑だめ


for _ in zip_func(gen_1, gen_2):
#tuple
    print(_)

