import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import enum
import math

import matplotlib
matplotlib.use('Qt5Agg')


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

    # elif option == NormalizationOptions.f_x:
    #     for lh in left_bound_and_height:
    #         lh[1] /= (len(y_) * dx)
    #     left_bound_and_height = [[-2, 0.4]] + left_bound_and_height[1:-1] + [[2, 0.4]]

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

        if interval == n_ - 1 or i > len(y_) - 1:
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


# Functions for plotting histogram
def get_data_for_step_plot(left_bound_and_height, a_=-2.5, b_=2.5):
    return [[a_] + [info[0] for info in left_bound_and_height] + [2, b_],
            [0] + [info[1] for info in left_bound_and_height] + [0, 0]]


def get_data_for_mids_plot(left_bound_and_height, a_=-2.5, b_=2.5):
    middles = []
    last_i = len(left_bound_and_height) - 1
    for i in range(last_i):
        middles.append((left_bound_and_height[i][0] + left_bound_and_height[i + 1][0]) / 2)
    middles.append(1 + left_bound_and_height[last_i][0] / 2)

    return [middles, [hd[1] for hd in left_bound_and_height]]


# Handlers for sliders and button
def update(val):
    n_ = int(round(n_slider.val))
    m_ = int(round(m_slider.val))
    y_ = generate_y(n_)
    resize = False
    if g_method == HistogramType.EQUIDISTANT_METHOD:
        data = get_data_for_equidistant_method(y_, m_, a_=-2, b_=2, option=g_option)
        if g_option == NormalizationOptions.HEIGHT_IS_AMOUNT:
            resize = True
    else:
        data = get_data_for_equiprobable_method(y_, m_, a_=-2, b_=2,  option=g_option)

    if resize:
        max = 1
        for lh in data:
            if lh[1] > max:
                max = lh[1]
        ax.set_ylim(0, max + 1)

    step_plot.set_data(*get_data_for_step_plot(data))
    mids_plot.set_data(*get_data_for_mids_plot(data))
    fig.canvas.draw_idle()


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

    # For interactive tools
    global g_method, g_option, ax
    g_method, g_option = hist_type, option
    global fig, step_plot, mids_plot, n_slider, m_slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Plotting histogram:
    step_plot, = plt.step(*get_data_for_step_plot(left_bound_and_height),
                          where='post', label=f'Гистограмма ({method_name} метод)')

    # Plotting polygon:
    mids_plot, = plt.plot(*get_data_for_mids_plot(left_bound_and_height), label='Полигон распределения')

    # Plotting analytic f(y):
    ls = np.linspace(a_, -2, 10000).tolist() \
        + np.linspace(-2, 2, 10000).tolist() \
        + np.linspace(2, b_, 10000).tolist()
    plt.plot(ls, [f_y(y_i) for y_i in ls], label='Аналитическая функция плотности')

    # Interactive tools
    n_slider = Slider(ax=plt.axes([0.1, 0.15, 0.8, 0.03]), label='n = ', valmin=1, valmax=10000, valinit=n_init)
    n_slider.on_changed(update)

    m_slider = Slider(ax=plt.axes([0.1, 0.1, 0.8, 0.03]), label='m = ', valmin=1, valmax=500, valinit=m_init)
    m_slider.on_changed(update)

    button = Button(plt.axes([0.1, 0.03, 0.12, 0.05]), 'Generate')
    button.on_clicked(update)

    plt.legend()
    plt.show()


# Plotting F(x) with group data
def get_accumulated_group_data(y_: list, n_: int) -> list:
    left_bound_and_height = get_data_for_equidistant_method(y_, n_, a_=-2, b_=2, option=NormalizationOptions.TOTAL_HEIGHT_IS_ONE)

    for i in range(len(left_bound_and_height) - 1):
        left_bound_and_height[i + 1][1] += left_bound_and_height[i][1]

    return left_bound_and_height


def update_2(val):
    n_ = int(round(n_slider.val))
    m_ = int(round(m_slider.val))
    y_ = generate_y(n_)

    F_x.set_data(*get_data_for_F_x(y_, m_))
    fig.canvas.draw_idle()


def get_data_for_F_x(y_, n_):
    left_bound_and_height = get_accumulated_group_data(y_, n_)

    # Plotting empirical:
    X, Y = [-3], [0]
    for i in range(len(left_bound_and_height)):
        X.append(left_bound_and_height[i][0])
        Y.append(left_bound_and_height[i][1])
    X.append(3)
    Y.append(1)
    return X, Y


def plot_group_F(y_: list, n_: int) -> list:
    # For interactive tools
    global fig, F_x, n_slider, m_slider
    fig, _ = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Plotting analytic:
    ls = np.linspace(-3, 3, 10000)
    plt.plot(ls, [F_y(y_i) for y_i in ls], label='Аналитическая функция распределения')

    F_x, = plt.step(*get_data_for_F_x(y_, n_), where='post', label='эмпирическая функция распределения')

    plt.suptitle('Сравнение функции рапределения аналитической\n'
                 'и по сгрупированным данным')

    # Interactive tools
    n_slider = Slider(ax=plt.axes([0.1, 0.15, 0.8, 0.03]), label='n = ', valmin=1, valmax=10000, valinit=n_init)
    n_slider.on_changed(update_2)

    m_slider = Slider(ax=plt.axes([0.1, 0.1, 0.8, 0.03]), label='m = ', valmin=1, valmax=500, valinit=m_init)
    m_slider.on_changed(update_2)

    button = Button(plt.axes([0.1, 0.03, 0.12, 0.05]), 'Generate')
    button.on_clicked(update_2)

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
    n_init = 10
    if n_init < 100:
        m_init = int(math.sqrt(n_init))
    else:
        m_init = int(2 * 3 * 4 * math.log(n_init))
    best = False
    y = generate_y(n_init)

    # plot_histogram(y, m_init,
    #                hist_type=HistogramType.EQUIDISTANT_METHOD,
    #                option=NormalizationOptions.HEIGHT_IS_AMOUNT)
    # plot_histogram(y, m_init,
    #                hist_type=HistogramType.EQUIDISTANT_METHOD,
    #                option=NormalizationOptions.TOTAL_HEIGHT_IS_ONE)
    # plot_histogram(y, m_init,
    #                hist_type=HistogramType.EQUIDISTANT_METHOD,
    #                option=NormalizationOptions.TOTAL_AREA_IS_ONE)
    # plot_histogram(y, m_init,
    #                hist_type=HistogramType.EQUIDISTANT_METHOD,
    #                option=NormalizationOptions.TOTAL_AREA_IS_ONE_2)

    # plot_histogram(y, m_init,
    #                hist_type=HistogramType.EQUIPROBABLE_METHOD,
    #                option=NormalizationOptions.CONTINUOUS_VALUE)
    # plot_histogram(y, m_init,
    #                hist_type=HistogramType.EQUIPROBABLE_METHOD,
    #                option=NormalizationOptions.MIXED_VALUE)

    # plot_group_F(y, m_init)