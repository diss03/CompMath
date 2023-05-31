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

    def RK4(self, state):
        # self.system.f(state * self.h)
        self.system.f(state)
        self.k[0] = self.system.state

        self.system.f(state + self.k[0] / 2 * self.h)
        self.k[1] = self.system.state

        self.system.f(state + self.k[1] / 2 * self.h)
        self.k[2] = self.system.state

        self.system.f(state + self.k[2] * self.h)
        self.k[3] = self.system.state

        return state + (self.k[0] + 2 * self.k[1] + 2 * self.k[2] + self.k[3]) / 6 * self.h


def createArr(m1, m2, k1, k2, k3, x10, x20, v10, v20, t):
    sys = MassSpringSystem(m1, m2, k1, k2, k3, x10, x20, v10, v20)
    solver = RungeKutta4MethodSolver(sys, t[1])
    state = sys.initial_state

    res = np.empty((len(t), 4), dtype=float)

    for i in range(0, (len(t))):
        res[i][0] = state[0]
        res[i][1] = state[2]
        res[i][2] = state[1]
        res[i][3] = state[3]

        state = solver.RK4(state)

    return res


def createArr_(m1, m2, k1, k2, beta, t):
    sys = MassSpringSystem(m1, m2, k1, k2, beta[0], beta[1], beta[2], beta[3], beta[4])
    solver = RungeKutta4MethodSolver(sys, t[1])  # t[1] = h
    state = sys.initial_state

    res = np.empty((len(t), 4), dtype=float)

    for i in range(0, (len(t))):
        res[i][0] = state[0]  # x1
        res[i][1] = state[2]  # x2
        res[i][2] = state[1]  # v1
        res[i][3] = state[3]  # v2

        state = solver.RK4(state)

    return res[:, 1]