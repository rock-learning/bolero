import numpy as np

def hello_world():
    print("Hello world")

def produce_int():
    return 9

def produce_double():
    return 8.0

def produce_bool():
    return True

def produce_string():
    return "Test string"

def produce_array():
    return np.array([1.3, 2.4, 3.5])

def produce_list():
    return [1.1, 2.2, 3.3]

def produce_tuple():
    return (1.4, 2.5, 3.6)

def take_int(i):
    assert(isinstance(i, int))

def take_double(d):
    assert(isinstance(d, float))

def take_bool(b):
    assert(isinstance(b, bool))

def take_string(s):
    assert(isinstance(s, str))

def take_array(a):
    assert(isinstance(a, np.ndarray))

def multiple_io(a, b):
    return [a, float(b)]

def raise_import_error():
    import bla

def print_list(l):
    print(l)
