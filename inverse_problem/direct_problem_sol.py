import numpy as np
from inverse_proble_sol import createArr, createArr_
from matplotlib import pyplot as plt
from math import fabs


def Jacobian(t, x2_observed, m1, m2, k1, k2, beta):
    J = np.zeros((len(x2_observed), len(beta)))

    for j in range(len(beta)):
        # эпсилон изменяется в зависимости от значения элемента вектора бета, по которому дифференцируем (иначе не работает)
        eps = 0.01 * beta[j]

        # численное дифференцирование
        new_beta = np.copy(beta)
        new_beta[j] += eps
        x2_p = createArr_(m1, m2, k1, k2, new_beta, t)
        new_beta[j] -= 2 * eps
        x2_m = createArr_(m1, m2, k1, k2, new_beta, t)
        dx2_dbeta_j = (x2_p - x2_m) / (2 * eps)

        # устанавливаем столбец с производными
        J[:, j] = dx2_dbeta_j
    return J


def GaussNewton(t, x2_observed, m1, m2, k1, k2, beta_init, tol, max_iter):
    beta = np.copy(beta_init)
    cost_new = 0

    for k in range(max_iter):
        # рассчет вектора невязки
        r_beta = createArr_(m1, m2, k1, k2, beta, t) - x2_observed

        # рассчет якобианы
        J = Jacobian(t, x2_observed, m1, m2, k1, k2, beta)
        # рассчет нового вектора бета
        beta = beta - np.linalg.inv(J.T @ J) @ J.T @ r_beta

        cost_old = cost_new
        cost_new = np.linalg.norm(r_beta)

        if fabs(cost_old - cost_new) < tol:
            return [beta, k + 1]

    return [beta, max_iter]


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


    # отрисовка идеальных значений и зашумленных значений
    plt.plot(t, true_solution[:, 1], color='y', linewidth=1.5, label='true chart')
    plt.plot(t, x2_observed, 'o', markersize=1.5, color='black', label='noise values')

    plt.title('Mass-Spring System', fontsize=15)
    plt.xlabel('Time, sec', fontsize=10)
    plt.ylabel('Coordinates, m', fontsize=10)

    plt.legend()
    plt.show()

    # запуск решения
    beta_init = np.array([40, 1, -1, -0.3, 0.6], dtype=float)
    tol = 0.000001  # погрешность для условия выхода (использую разность норм бета текущей и предыдущей итерации)
    max_iter = 10
    beta_opt, iter = GaussNewton(t, x2_observed, m1, m2, k1, k2, beta_init, tol, max_iter)

    print("Количество пройденных итераций = ", iter)
    print("Ответ: k3 = ", beta_opt[0], ", x10 = ", beta_opt[1], ", x20 = ", beta_opt[2], ", v10 = ",
          beta_opt[3], ", v20 = ", beta_opt[4], sep='')

    # отрисовка зашумленных значений и графика по полученному вектору бета
    plt.plot(t, x2_observed, 'o', markersize=1.5, color='black', label='noise values')
    plt.plot(t, createArr_(m1, m2, k1, k2, beta_opt, t), color='red', linewidth=1.5,
             label='optimized value chart', linestyle='dashed')

    plt.title('Mass-Spring System', fontsize=15)
    plt.xlabel('Time, sec', fontsize=10)
    plt.ylabel('Coordinates, m', fontsize=10)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
# 4.60971112e-01  4.20622065e-01  5.57000270e-01  5.00885586e-01
#   4.69845978e-01  4.95324455e-01  5.06263435e-01  4.69116813e-01
# [ 0.5         0.49913823  0.49655652  0.49226566  0.48628354  0.47863513
#   0.46935232  0.4584738   0.44604486  0.4321172   0.4167487   0.40000312