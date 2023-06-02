import numpy as np
from direct_problem_sol import createArr, createArr_
from matplotlib import pyplot as plt
from math import fabs
import time

class JacoianMatrix():
    def __init__(self, len1, len2):
        self.J = np.zeros((len1, len2))


class X2Array():
    def __init__(self, len):
        self.p = np.zeros(len)
        self.m = np.zeros(len)


class Ecuation():
    def __init__(self, len):
        self.A = np.zeros((len, len))
        self.b = np.zeros(len)
        self.x = np.zeros(len)
        self.y = np.zeros(len)
        self.L = np.zeros((len, len))
        self.L_T = np.zeros((len, len))


class Residual():
    def __init__(self, len):
        self.r_beta = np.zeros(len)


class Beta():
    def __init__(self, init_beta):
        self.new = np.copy(init_beta)
        self.old = np.copy(init_beta)


def Jacobian(t, J, x2_arr, m1, m2, k1, k2, beta):
    for j in range(len(beta)):
        # эпсилон изменяется в зависимости от значения элемента вектора бета, по которому дифференцируем
        eps = 0.01 * beta[j]

        # численное дифференцирование
        beta[j] += eps
        x2_arr.p = createArr_(m1, m2, k1, k2, beta, t)
        beta[j] -= 2 * eps
        x2_arr.m = createArr_(m1, m2, k1, k2, beta, t)
        # устанавливаем столбец с производными
        J.J[:, j] = (x2_arr.p - x2_arr.m) / (2 * eps)


def CholeskyDecompositionSolver(ec):
    n = len(ec.A)
    ec.L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            sum = 0
            if (i == j):
                for k in range(j):
                    sum += ec.L[j][k] ** 2
                ec.L[j][j] = np.sqrt(ec.A[j][j] - sum)
            else:
                for k in range(j):
                    sum += (ec.L[i][k] * ec.L[j][k])
                ec.L[i][j] = (ec.A[i][j] - sum) / ec.L[j][j]

    ec.L_T = ec.L.T

    for i in range(n):
        sum = 0
        for l in range(i):
            sum += ec.y[l] * ec.L[i][l]
        ec.y[i] = (ec.b[i] - sum) / ec.L[i][i]

    for i in range(n - 1, -1, -1):
        sum = 0
        for l in range(n - 1, i, -1):
            sum += ec.x[l] * ec.L_T[i][l]
        ec.x[i] = (ec.y[i] - sum) / ec.L_T[i][i]


def GaussNewton(t, x2_observed, m1, m2, k1, k2, beta, tol, max_iter):
    cost_new = 0
    J = JacoianMatrix(len(x2_observed), len(beta.new))
    x2_arr = X2Array(len(x2_observed))
    ec = Ecuation(len(beta.new))
    r = Residual(len(x2_observed))

    for k in range(max_iter):
        # рассчет вектора невязки
        r.r_beta = createArr_(m1, m2, k1, k2, beta.new, t) - x2_observed

        # рассчет якобианы
        Jacobian(t, J, x2_arr, m1, m2, k1, k2, beta.new)
        # рассчет нового вектора бета
        beta.old = beta.new

        ec.A = np.dot(J.J.T, J.J)
        ec.b = np.dot(J.J.T, r.r_beta)
        CholeskyDecompositionSolver(ec)

        beta.new = beta.old - ec.x.T

        cost_old = cost_new
        cost_new = np.linalg.norm(r.r_beta)

        if fabs(cost_old - cost_new) < tol:
            return k + 1

    return max_iter


def main():
    stop_time = 3
    h = 0.01
    t = np.arange(0, stop_time, h)
    m1, m2 = 1, 2.03
    k1, k2, k3 = 20, 20, 30
    x10, x20, v10, v20 = -0.5, 0.5, 0, 0
    # получение массива из прямой задачи
    true_solution = createArr(m1, m2, k1, k2, k3, x10, x20, v10, v20, t)

    # добавление шумов
    x2_observed = true_solution[:, 1] + np.random.normal(scale=0.05, size=len(t))

    tic = time.perf_counter()
    # запуск решения
    beta_init = np.array([40, 1, -1, -0.3, 0.6], dtype=float)
    beta = Beta(beta_init)
    tol = 0.000001  # погрешность для условия выхода (использую разность норм бета текущей и предыдущей итерации)
    max_iter = 20
    iter = GaussNewton(t, x2_observed, m1, m2, k1, k2, beta, tol, max_iter)
    toc = time.perf_counter()

    print("Первоначальные данные: k3 = ", k3, ", x10 = ", x10, ", x20 = ", x20, ", v10 = ", v10, ", v20 = ", v20,
          sep='')
    print("Приближенные данные: k3 = ", beta_init[0], ", x10 = ", beta_init[1], ", x20 = ", beta_init[2], ", v10 = ", beta_init[3], ", v20 = ", beta_init[4],
          sep='')
    print("Количество пройденных итераций:", iter)
    print("Ответ: k3 = ", beta.new[0], ", x10 = ", beta.new[1], ", x20 = ", beta.new[2], ", v10 = ",
          beta.new[3], ", v20 = ", beta.new[4], sep='')
    print(f"Вычисление заняло: {toc - tic:0.4f} секунд")

    # отрисовка зашумленных значений и графика по полученному вектору бета
    plt.plot(t, true_solution[:, 1], color='blue', linewidth=1.5, label='true chart')
    plt.plot(t, x2_observed, 'o', markersize=1.5, color='black', label='noise values')
    plt.plot(t, createArr_(m1, m2, k1, k2, beta.new, t), color='red', linewidth=1.5,
             label='optimized value chart', linestyle='dashed')

    plt.title('Mass-Spring System', fontsize=15)
    plt.xlabel('Time, sec', fontsize=10)
    plt.ylabel('Coordinates, m', fontsize=10)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
