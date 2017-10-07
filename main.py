from numpy.polynomial.chebyshev import chebval
import numpy as np
import pandas as pd
from sklearn import preprocessing
from cvxopt.solvers import qp
from cvxopt import matrix


# TODO: Document me
def scale(data, feature_range=(-1, 1), columnar=False):
    bounds = data if columnar else np.reshape(data, (-1, 1))
    scale_func = preprocessing.MinMaxScaler(feature_range=feature_range).fit(bounds)
    return scale_func.transform(data), scale_func


def generate_cheb(x, num_coeffs):
    return chebval(x, np.eye(num_coeffs)).squeeze()


def absolute_error(data, error=.1):
    return np.c_[data - error, data + error]


def gen_a(basis_functions, bounds):
    return np.c_[
        np.r_[basis_functions, -basis_functions],
        np.r_[-basis_functions * bounds[:, 0], basis_functions * bounds[:, 1]]
    ].transpose()


def main():
    filename = 'Kirby2.dat'
    data = pd.read_csv(filename, delim_whitespace=True).as_matrix()

    vals, data_scale = scale(absolute_error(data[:, -1]))
    inputs, inputs_scale = scale(data[:, 0:-1], columnar=True)
    n = 2
    b = generate_cheb(inputs, n)

    a = gen_a(b, vals)

    print(a.shape)
    P = matrix(np.eye(n*2))
    q = matrix(np.zeros(n*2))
    d = matrix(1/np.linalg.norm(a, axis=1))
    problem = qp(P, q, G=matrix(-a), h=-d)

    print(problem)


if __name__ == '__main__':
    main()

# Normalize inputs
# Normalize data

# simple just scale between max and mix w.r.t error margin
#   min = mindata-lower_bound
#   max = maxdata+upper_bound
