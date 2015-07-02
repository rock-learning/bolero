import itertools
import numpy as np


def constant(_):
    return np.array([1.0])


def linear(context):
    return context


def affine(context):
    return polynomial(context, n_degrees=1)


def quadratic(context):
    return polynomial(context, n_degrees=2)


def cubic(context):
    return polynomial(context, n_degrees=3)


def polynomial(context, n_degrees=2):
    # From sklearn.preprocessing.PolynomialFeatures
    # Find permutations/combinations which add to degree or less
    context = np.asarray(context)
    n_features = context.shape[0]
    powers = itertools.product(*(range(n_degrees + 1)
                                 for i in range(n_features)))
    powers = np.array([c for c in powers if 0 <= np.sum(c) <= n_degrees])
    # Sort so that the order of the powers makes sense
    i = np.lexsort(np.vstack([powers.T, powers.sum(axis=1)]))
    powers = powers[i][::-1]
    return (context ** powers).prod(-1)


CONTEXT_TRANSFORMATIONS = {
    "constant": constant,
    "linear": linear,
    "affine": affine,
    "quadratic": quadratic,
    "cubic": cubic}
