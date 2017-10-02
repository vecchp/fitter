import pandas as pd

from numpy.polynomial.chebyshev import chebval
import numpy as np

def normalize_data():

    pass



def generate_cheb(x, num_coeffs):
    return chebval(x, np.eye(num_coeffs))

def main():
    data_set = 'Kirby2.dat'
    # data = pd.read_csv(data_set, delim_whitespace=True).as_matrix()


    print(generate_cheb([1,2,3,4], 4))


if __name__ == '__main__':
    main()


# error bounds
# normalize data

# data - lower bounds
# data + upper bounds
