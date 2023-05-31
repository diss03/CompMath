import numpy as np


def transpose(A):
    A_T = np.copy(A)
    for i in range(len(A)):
        for j in range(len(A)):
            A_T[i][j] = A[j][i]
    return A_T


# нахождение определителя по формуле Лапласа
def determinant(A):
    n = len(A)
    if n == 1:
        return A[0][0]
    elif n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    else:
        det = 0
        for j in range(n):
            minor = np.empty((n - 1, n - 1))
            for i in range(1, n):
                for k in range(n):
                    if k < j:
                        minor[i - 1][k] = A[i][k]
                    elif k > j:
                        minor[i - 1][k - 1] = A[i][k]
            # уточнить знак!!!
            sign = (-1) ** j
            det += sign * A[0][j] * determinant(minor)
        return det


def unionMatrix(A):
    n = len(A)
    A_star = np.zeros((n, n))
    minor = np.zeros((n - 1, n - 1))
    for i in range(n):
        for j in range(n):

            for l in range(n):
                for m in range(n):
                    if l < i and m < j:
                        minor[l][m] = A[l][m]
                    elif l > i and m < j:
                        minor[l - 1][m] = A[l][m]
                    elif l < i and m > j:
                        minor[l][m - 1] = A[l][m]
                    elif l > i and m > j:
                        minor[l - 1][m - 1] = A[l][m]

            A_star[i][j] = ((-1) ** (i + j)) * determinant(minor)
    return A_star


def CholeskyDecompositionSolver(A, b):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                if i == 0:
                    L[i][i] = np.sqrt(A[i][i])
                else:
                    L[i][i] = np.sqrt(A[i][i] - np.sum(L[i] ** 2))
            else:
                if i == 0:
                    L[j][i] = A[j][i] / L[i][i]
                elif i < j:
                    L[j][i] = (A[j][i] - np.sum(L[i] * L[j])) / L[i][i]

    L_T = transpose(L)
    L_inv = transpose(unionMatrix(L)) / determinant(L)
    L_T_inv = transpose(unionMatrix(L_T)) / determinant(L_T)

    # L @ y = b
    # L_T @ x = y
    # остюда:

    y = L_inv @ b
    x = L_T_inv @ y

    return x
