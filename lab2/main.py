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
    elif y_ == -2 or y_ == 2:
        return 0.4
    else:
        return 0.05


class NormalizationOptions(enum.Enum):
    HEIGHT_IS_AMOUNT = 1,
    TOTAL_HEIGHT_IS_ONE = 2,
    TOTAL_AREA_IS_ONE = 3,
    TOTAL_AREA_IS_ONE_2 = 4,
    f_x = 5,

    CONTINUOUS_VALUE = 6,
    MIXED_VALUE = 7


class HistogramType(enum.Enum):
    EQUIDISTANT_METHOD = 0,
    EQUIPROBABLE_METHOD = 1


def get_data_for_equidistant_method(y_: list, n_: int, a_, b_, option: NormalizationOptions) -> list:
    dx = (b_ - a_) / n_
    left_bound_and_height = [[a_ + i * dx, 0] for i in range(n_)]

    interval, i = 0, 0
    while i < len(y_):
        if interval == n_ - 1 or y_[i] < left_bound_and_height[interval + 1][0]:
            left_bound_and_height[interval][1] += 1
            i += 1
        else:
            interval += 1

    if option == NormalizationOptions.TOTAL_HEIGHT_IS_ONE:
        h, h_n = 0, 0
        for lh in left_bound_and_height:
            lh[1] /= len(y_)

            if lh[0] != -2 and lh[0] != 2 - dx:
                h += lh[1]
                h_n += 1
        print('M[h ∈ (-2, 2)]', h / h_n)

    elif option == NormalizationOptions.TOTAL_AREA_IS_ONE:
        S = 0
        h, h_n = 0, 0
        for lh in left_bound_and_height:
            S += dx * lh[1]
        print('*', S / dx)
        for lh in left_bound_and_height:
            lh[1] /= S
            if lh[0] != -2 and lh[0] != 2 - dx:
                h += lh[1]
                h_n += 1
        print('M[h ∈ (-2, 2)]', h / h_n)

    elif option == NormalizationOptions.TOTAL_AREA_IS_ONE_2:
        h, h_n = 0, 0
        for lh in left_bound_and_height:
            lh[1] /= (len(y_) * dx)

            if lh[0] != -2 and lh[0] != 2 - dx:
                h += lh[1]
                h_n += 1
        print('M[h ∈ (-2, 2)]', h / h_n)

    elif option == NormalizationOptions.f_x:
        for lh in left_bound_and_height:
            lh[1] /= (len(y_) * dx)
        left_bound_and_height = [[-2, 0.4]] + left_bound_and_height[1:-1] + [[2, 0.4]]

    elif option != NormalizationOptions.HEIGHT_IS_AMOUNT:
        raise TypeError(f'Option \'{option}\' is not defined...')

    return left_bound_and_height


def get_data_for_equiprobable_method(y_: list, n_: int, a_, b_, option: NormalizationOptions) -> list:
    left_bound_and_height = [[a_, None]]
    y_ = remove_discrete_points(y_)

    obligatory = len(y_) // n_
    additional = len(y_) % n_

    S = 0
    i, interval = 0, 0
    while interval < n_:
        if additional:
            additional -= 1
            amount = obligatory + 1
        else:
            amount = obligatory

        i += amount

        if interval == n_ - 1 or i > len(y_):
            r = 2
        else:
            r = (y_[i] + y_[i - 1]) / 2
            left_bound_and_height.append([r, None])

        dx = r - left_bound_and_height[interval][0]

        h = amount / dx
        left_bound_and_height[interval][1] = h
        S += h * dx
        if r == 2:
            break
        interval += 1

    for lh in left_bound_and_height:
        lh[1] /= S * 5

    if option == option.CONTINUOUS_VALUE:
        left_bound_and_height = [[-2, 0.4]] + left_bound_and_height + [[2, 0.4]]
    elif option == option.MIXED_VALUE:
        left_bound_and_height = [[-2, float("inf")]] + left_bound_and_height + [[2, float('inf')]]
    else:
        raise TypeError(f'Option \'{option}\' is not defined...')

    return left_bound_and_height


def plot_histogram(y_, n_, a_=-2.5, b_=2.5,
                   hist_type: HistogramType = HistogramType.EQUIDISTANT_METHOD,
                   option: NormalizationOptions = NormalizationOptions.HEIGHT_IS_AMOUNT):

    if hist_type == HistogramType.EQUIDISTANT_METHOD:
        left_bound_and_height = get_data_for_equidistant_method(y_, n_, a_=-2, b_=2, option=option)
        method_name = 'равноинтервальный'
    elif hist_type == HistogramType.EQUIPROBABLE_METHOD:
        left_bound_and_height = get_data_for_equiprobable_method(y_, n_, a_=-2, b_=2, option=option)
        method_name = 'равновероятностный'
    else:
        raise Exception('Invalid method')

    # Plotting histogram:
    plt.step([a_] + [info[0] for info in left_bound_and_height] + [2, b_],
             [0] + [info[1] for info in left_bound_and_height] + [0, 0],
             where='post', label=f'Гистограмма ({method_name} метод)')

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
    left_bound_and_height = get_data_for_equidistant_method(y_, n_, a_=-2, b_=2, option=NormalizationOptions.TOTAL_HEIGHT_IS_ONE)

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


def remove_discrete_points(y_):
    discrete_points = (-2, 2)
    continuous_points = []

    for val in y_:
        if val not in discrete_points:
            continuous_points.append(val)

    return continuous_points


def check(l):
    S = 0
    for i in range(len(l)):
        left = l[i][0]
        right = l[i + 1][0] if len(l) != i + 1 else 2
        dx = right - left
        S += dx * l[i][1]

    print('S = ', S)

if __name__ == '__main__':
    n = 10
    m = 1000
    y = generate_y(n)
    plot_histogram(y, m,
                   hist_type=HistogramType.EQUIDISTANT_METHOD,
                   option=NormalizationOptions.f_x)

    # plot_histogram_2(y, m)
    # plot_f(y, add_analytical=True)
    # plot_group_F(y, n)