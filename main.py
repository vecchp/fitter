from numpy.polynomial.chebyshev import chebval
import numpy as np
import pandas as pd
from sklearn import preprocessing


def scale(data, feature_range=(-1, 1), full=True):
    bounds = data if full else np.reshape(data, (-1, 1))
    scale_func = preprocessing.MinMaxScaler(feature_range=feature_range).fit(bounds)
    return scale_func.transform(data), scale_func


def generate_cheb(x, num_coeffs):
    return chebval(x, np.eye(num_coeffs))


def absolute_error(data, error=.01):
    return np.c_[data, data - error, data + error]


def main():
    filename = 'Kirby2.dat'
    data = pd.read_csv(filename, delim_whitespace=True).as_matrix()

    vals, unscale = scale(absolute_error(data[:, -1]))
    print(vals)

    inputs, unscale = scale(data[:, 0:-1], full=False)
    print(inputs)
    # print(generate_cheb([1, 2, 3, 4], 4))


if __name__ == '__main__':
    main()

# Normalize inputs
# Normalize data

# simple just scale between max and mix w.r.t error margin
#   min = mindata-lower_bound
#   max = maxdata+upper_bound
