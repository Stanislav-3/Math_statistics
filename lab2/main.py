import random
import matplotlib.pyplot as plt
import numpy as np
import enum


def get_x():
    a_, b_ = -5, 5
    return a_ + random.random() * (b_ - a_)


def Y_x(x):
    if x < -1:
        return -2
    elif x > 1:
        return 2
    else:
        return 2 * x


def get_y():
    return Y_x(get_x())


def generate_y(n_: int) -> list:
    y_ = []

    for i in range(n_):
        y_.append(get_y())

    y_.sort()
    return y_


def get_frequency(y_: list, i: int) -> int:
    count = 1

    while i < len(y_) - 1 and y_[i] == y_[i + 1]:
        i += 1
        count += 1

    return count


def F_y(y_):
    if y_ < -2:
        return 0
    if y_ > 2:
        return 1
    else:
        return 0.05 * y_ + 0.5


def f_y(y_):
    if y_ < -2 or y_ > 2:
        return 0
    # TODO: maybe infinity
    elif y_ == -2 or y_ == 2:
        return 0.4
    else:
        return 0.05


# TODO: fix practical f(y)
def plot_f(y, add_empirical=False, add_analytical=False):
    x_, y_ = [-2.5, -2], [0, 0]
    if add_empirical:
        for i in range(len(y)):
            y_i = y[i]
            x_.append(y_i)
            y_.append(f_y(y_i))

        x_.extend([2, 2.5])
        y_.extend([0, 0])
        plt.plot(x_, y_, label=f'Эмперическая ф-ия плотности (n = {n})')

    if add_analytical:
        ls = np.linspace(-2.5, -2, 10000).tolist() \
             + np.linspace(-2, 2, 10000).tolist() \
             + np.linspace(2, 2.5, 10000).tolist()
        plt.plot(ls, [f_y(y_i) for y_i in ls], label='Аналитическая ф-ия плотности')
        plt.suptitle('Сравнение аналитической и эмперической функции плотности f(y)')
    else:
        plt.suptitle('Эмперическая функция плотности f(y)')

    plt.xlabel('y')
    plt.ylabel('f(y)')
    plt.legend()
    plt.show()


class NormalizationOptions(enum.Enum):
    HEIGHT_IS_AMOUNT = 1,
    TOTAL_HEIGHT_IS_ONE = 2,
    TOTAL_AREA_IS_ONE = 3


def get_data_1(y_: list, n_: int, a_, b_, option: NormalizationOptions) -> list:
    M = (b_ - a_) / n_
    left_bound_and_height = [[a_ + i * M, 0] for i in range(n_)]

    interval, i = 0, 0
    while i < len(y_):
        if interval == n_ - 1 or y_[i] < left_bound_and_height[interval + 1][0]:
            left_bound_and_height[interval][1] += 1
            i += 1
        else:
            interval += 1

    if option == NormalizationOptions.TOTAL_HEIGHT_IS_ONE:
        for lh in left_bound_and_height:
            lh[1] /= len(y_)
        S = 0
        for lh in left_bound_and_height:
            S += lh[1]
        print(S)

    elif option == NormalizationOptions.TOTAL_AREA_IS_ONE:
        S = 0
        o = 0
        v = 0
        for lh in left_bound_and_height:
            S += M * lh[1]
        for lh in left_bound_and_height:
            lh[1] /= S
            if lh[1] > 0.5:
                continue
            o += lh[1]
            v += 1
        print(o / v)

    elif option != NormalizationOptions.HEIGHT_IS_AMOUNT:
        raise TypeError(f'Option \'{option}\' is not defined...')

    return left_bound_and_height


# TODO: Improve normalization
def plot_histogram_1(y_, n_, a_=-2.5, b_=2.5,
                     option: NormalizationOptions = NormalizationOptions.TOTAL_AREA_IS_ONE):
    left_bound_and_height = get_data_1(y_, n_, a_=-2, b_=2, option=option)

    # Plotting histogram:
    plt.step([a_] + [info[0] for info in left_bound_and_height] + [2, b_],
             [0] + [info[1] for info in left_bound_and_height] + [0, 0],
             where='post', label='Гистограмма (равноинтервальный метод)')

    # Plotting polygon:
    middles = []
    last_i = len(left_bound_and_height) - 1
    for i in range(last_i):
        middles.append((left_bound_and_height[i][0] + left_bound_and_height[i + 1][0]) / 2)
    middles.append(1 + left_bound_and_height[last_i][0] / 2)
    plt.plot(middles, [hd[1] for hd in left_bound_and_height], label='Полигон распределения')

    # Plotting analytic f(y):
    if option == 'f':
        ls = np.linspace(a_, -2, 10000).tolist() \
             + np.linspace(-2, 2, 10000).tolist() \
             + np.linspace(2, b_, 10000).tolist()
        plt.plot(ls, [f_y(y_i) for y_i in ls], label='Аналитическая функция плотности')

    plt.legend()
    plt.show()


def get_accumulated_group_data(y_: list, n_: int) -> list:
    left_bound_and_height = get_data_1(y_, n_, a_=-2, b_=2, option=NormalizationOptions.TOTAL_HEIGHT_IS_ONE)

    for i in range(len(left_bound_and_height) - 1):
        left_bound_and_height[i + 1][1] += left_bound_and_height[i][1]

    return left_bound_and_height


def plot_group_F(y_: list, n_: int) -> list:
    left_bound_and_height = get_accumulated_group_data(y_, n_)

    # Plotting analytic:
    ls = np.linspace(-3, 3, 10000)
    plt.plot(ls, [F_y(y_i) for y_i in ls], label='Аналитическая функция распределения')

    # Plotting empirical:
    X, Y = [-3], [0]
    for i in range(len(left_bound_and_height)):
        X.append(left_bound_and_height[i][0])
        Y.append(left_bound_and_height[i][1])
    X.append(3)
    Y.append(1)

    plt.step(X, Y, where='post', label='эмпирическая функция распределения')

    plt.suptitle('Сравнение функции рапределения аналитической\n'
                 'и по сгрупированным данным')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 100000
    y = generate_y(n)
    m = 10000
    plot_histogram_1(y, m, option=NormalizationOptions.TOTAL_AREA_IS_ONE)
    # plot_f(y, add_analytical=True)
    # plot_group_F(y, n)
    # print(y)