from numpy.polynomial.chebyshev import chebval
import numpy as np
import pandas as pd
from sklearn import preprocessing


def normalize_data(lower, upper):
    data = np.concatenate((lower, upper), axis=0).reshape(-1, 1)
    scale = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)
    print(scale.transform(data))


def generate_cheb(x, num_coeffs):
    return chebval(x, np.eye(num_coeffs))


def absolute_error(data, error=.01):
    return data - error, data + error


def main():
    filename = 'Kirby2.dat'
    data = pd.read_csv(filename, delim_whitespace=True).as_matrix()

    lower, upper = absolute_error(data[:, -1])

    normalize_data(lower, upper)

    # print(generate_cheb([1, 2, 3, 4], 4))


if __name__ == '__main__':
    main()

# Normalize inputs
# Normalize data

# simple just scale between max and mix w.r.t error margin
#   min = mindata-lower_bound
#   max = maxdata+upper_bound
