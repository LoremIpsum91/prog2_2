import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import matrix


matrix1 = [[2, 3, 0, 5], [4, -3, -1, 1], [2, 5, 1, 3], [2, 7, 2, -2]]
matrix2 = [[-3, -1, 1, 4], [5, 1, 3, 6], [7, 2, -2, -2], [7, 2, -2, -2]]


def first():
    print('own multiply: ', *matrix.multiply_matrix(matrix1, matrix2), sep='\n')
    print('\n')
    print('numpy multiply: ', '\n', matrix.checking_with_numpy(matrix1, matrix2))
    print('\n')
    print('transpose matrix: ', *matrix.trans(matrix1), sep='\n')
    print('\n')
    print('determinant of matrix: ', matrix.determinant(matrix2))
    print('\n')
    print('numpy det: ', np.linalg.det(matrix2))
    print('\n')
    print('matrix in power 2: ', *matrix.pow_matrix(matrix1), sep='\n')
    print('\n')
    print('numpy pow: ', *np.array(matrix1) @ matrix1, sep='\n')
    return


def relife():
    table = [random.randint(0, 30) for i in range(30)]
    for i in range(10):
        table.pop(random.randint(0, len(table)))
    return table


def func(t, A, h, T, phi):
    return A*np.exp(-h*t)*np.sin(2*np.pi/T*t + phi)


def second():
    t = np.array([i for i in range(20)])
    y = np.array(relife())

    popt, pcov = curve_fit(func, t, y, (1e3, 1e-2, 1., -1e1), maxfev=10 ** 6)
    A, h, T, phi = popt

    print('A={0}\nh={1}\nT={2}\nphi={3}'.format(*tuple(popt)))

    plt.scatter(t, y, s=30, color='orange')
    plt.plot(t, func(t, *popt))
    plt.show()
    return


def third(n):
    numbers = [[random.randint(0,30) for _ in range(n)] for _ in range(n)]

    for i, j in enumerate(numbers):
        m = (sum(j) / len(j))
        result = 0
        for u in j:
            result += (u - m)**2
        print('Table number: ', i, ' Expected value: ', m, ' Variance: ', result)
    return

print(first())
print(second())
print(third(10))