import numpy as np
import matplotlib.pyplot as plt


class MassSpringSystem:
    def __init__(self, m1, m2, k1, k2, k3, x10, x20, v10=0.0, v20=0.0):
        self.K = np.empty([2, 2], dtype=float)
        self.K[0][0] = -(k1 + k2) / m1
        self.K[0][1] = k2 / m1
        self.K[1][0] = k2 / m2
        self.K[1][1] = -(k2 + k3) / m2
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        self.initial_state = np.array([x10, v10, x20, v20])

    def f(self, state):
        a = np.matmul(self.K, [state[0], state[2]])
        self.state[1] = a[0]
        self.state[3] = a[1]
        self.state[0] = state[1]
        self.state[2] = state[3]


class RungeKutta4MethodSolver:
    def __init__(self, sys, stop=10, h=0.001):
        self.sys = sys
        self.stop = stop
        self.h = h
        self.k = np.empty([4, 4], dtype=float)

    def RK4(self, state):
        self.sys.f(state * self.h)
        self.k[0] = self.sys.state

        self.sys.f(state + self.k[0] / 2 * self.h)
        self.k[1] = self.sys.state

        self.sys.f(state + self.k[1] / 2 * self.h)
        self.k[2] = self.sys.state

        self.sys.f(state + self.k[2] * self.h)
        self.k[3] = self.sys.state

        return state + (self.k[0] + 2 * self.k[1] + 2 * self.k[2] + self.k[3]) / 6 * self.h


def createArr(sys, stop, h):
    time_arr = np.arange(0, stop, h)
    count = round(stop / h)

    solver = RungeKutta4MethodSolver(sys, stop, h)
    state = sys.initial_state

    arrX1 = [state[0]]
    arrX2 = [state[2]]

    for i in range(count - 1):
        # print(state)
        state = solver.RK4(state)

        arrX1.append(state[0])
        arrX2.append(state[2])

    return arrX1, arrX2, time_arr

def makeNose(arr):
    arr_with_noise = []
    for i in range(len(arr)):
        arr_with_noise.append(arr[i] + np.random.normal(scale=0.01))
    return arr_with_noise

def staticGraph(arrX1, arrX2, arrT):
    plt.plot(arrT, arrX1, color='y', label='first mass')
    plt.plot(arrT, arrX2, color='b', label='second mass')

    zero_arr = [0] * len(arrT)
    plt.plot(arrT, zero_arr, color='black', linewidth=0.7)

    plt.title('Mass-Spring System', fontsize=15)
    plt.xlabel('Time, sec', fontsize=10)
    plt.ylabel('Coordinates, m', fontsize=10)

    plt.legend()
    plt.show()
    return

def staticGraphInverseProblem(arrX_noise, arrX, arrT):
    zero_arr = [0] * len(arrT)
    plt.plot(arrT, zero_arr, color='black', linewidth=0.7)

    plt.plot(arrT, arrX_noise, 'bo', markersize=1)

    plt.plot(arrT, arrX, color='r', label='first mass')

    plt.title('Mass-Spring System', fontsize=15)
    plt.xlabel('Time, sec', fontsize=10)
    plt.ylabel('Coordinates, m', fontsize=10)

    plt.legend()
    plt.show()
    return

def main():
    m1 = 1
    m2 = 2.03
    k1 = 20
    k2 = 20
    k3 = 20
    x10 = 0.05
    x20 = -0.05
    v10 = 0
    v20 = 0

    sys = MassSpringSystem(m1, m2, k1, k2, k3, x10, x20, v10, v20)

    # временные характеристики:
    h = 0.001
    stop = 3
    arrX1, arrX2, arrT = createArr(sys, stop, h)

    arrX1_noise = makeNose(arrX1)
    arrX2_noise = makeNose(arrX2)

    # staticGraph(arrX1, arrX2, arrT)
    staticGraphInverseProblem(arrX1_noise, arrX1, arrT)
    # staticGraphInverseProblem(arrX2_noise, arrX2, arrT)

if __name__ == '__main__':
    main()