import numpy as np


class MassSpringSystem:
    def __init__(self, m1, m2, k1, k2, k3, x10, x20, v10, v20):
        self.K = np.empty([2, 2], dtype=float)
        self.K[0][0] = -(k1 + k2) / m1
        self.K[0][1] = k2 / m1
        self.K[1][0] = k2 / m2
        self.K[1][1] = -(k2 + k3) / m2

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.m1 = m1
        self.m2 = m2

        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        self.initial_state = np.array([x10, v10, x20, v20])

    def f(self, state):
        self.K[0][0] = -(self.k1 + self.k2) / self.m1
        self.K[0][1] = self.k2 / self.m1
        self.K[1][0] = self.k2 / self.m2
        self.K[1][1] = -(self.k2 + self.k3) / self.m2

        a = np.matmul(self.K, [state[0], state[2]])
        self.state[1] = a[0]
        self.state[3] = a[1]
        self.state[0] = state[1]
        self.state[2] = state[3]


class RungeKutta4MethodSolver:
    def __init__(self, system, h=0.01):
        self.system = system
        self.h = h
        self.k = np.empty([4, 4], dtype=float)
        self.state = np.empty([4, 1], dtype=float)
        self.tmp_arr = np.empty([4], dtype=float)

    def RK4(self):
        self.system.f(self.state)
        self.k[0] = np.copy(self.system.state)

        np.divide(self.k[0], 2, out=self.tmp_arr)
        np.dot(self.tmp_arr, self.h, out=self.tmp_arr)
        np.add(self.state, self.tmp_arr, out=self.tmp_arr)
        self.system.f(self.tmp_arr)
        self.k[1] = np.copy(self.system.state)

        np.divide(self.k[1], 2, out=self.tmp_arr)
        np.dot(self.tmp_arr, self.h, out=self.tmp_arr)
        np.add(self.state, self.tmp_arr, out=self.tmp_arr)
        self.system.f(self.tmp_arr)
        self.k[2] = np.copy(self.system.state)

        np.dot(self.k[2], self.h, out=self.tmp_arr)
        np.add(self.state, self.tmp_arr, out=self.tmp_arr)
        self.system.f(self.tmp_arr)
        self.k[3] = np.copy(self.system.state)

        np.dot(self.k[1], 2, out=self.tmp_arr)
        np.add(self.k[0], self.tmp_arr, out=self.tmp_arr)
        np.dot(self.k[2], 2, out=self.k[2])
        np.add(self.tmp_arr, self.k[2], out=self.tmp_arr)
        np.add(self.tmp_arr, self.k[3], out=self.tmp_arr)
        np.divide(self.tmp_arr, 6, out=self.tmp_arr)
        np.dot(self.tmp_arr, self.h, out=self.tmp_arr)
        np.add(self.state, self.tmp_arr, out=self.state)


def createArr(m1, m2, k1, k2, k3, x10, x20, v10, v20, t):
    sys = MassSpringSystem(m1, m2, k1, k2, k3, x10, x20, v10, v20)
    solver = RungeKutta4MethodSolver(sys, t[1])
    solver.state = np.copy(sys.initial_state)

    res = np.empty((len(t), 4), dtype=float)

    for i in range(0, (len(t))):
        res[i][0] = solver.state[0]
        res[i][1] = solver.state[2]
        res[i][2] = solver.state[1]
        res[i][3] = solver.state[3]

        solver.RK4()

    return res


def createArr_(m1, m2, k1, k2, beta, t):
    sys = MassSpringSystem(m1, m2, k1, k2, beta[0], beta[1], beta[2], beta[3], beta[4])
    solver = RungeKutta4MethodSolver(sys, t[1])  # t[1] = h
    solver.state = np.copy(sys.initial_state)

    res = np.empty((len(t), 4), dtype=float)

    for i in range(0, (len(t))):
        res[i][0] = solver.state[0]  # x1
        res[i][1] = solver.state[2]  # x2
        res[i][2] = solver.state[1]  # v1
        res[i][3] = solver.state[3]  # v2

        solver.RK4()

    return res[:, 1]
