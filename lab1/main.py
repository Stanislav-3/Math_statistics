import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
matplotlib.use('Qt5Agg')

import numpy as np
import random
from prettytable import PrettyTable


def get_x():
    return a + random.random() * (b - a)


def get_y():
    return Y_f(get_x())


def Y_f(x):
    if x < -1:
        return -2
    elif x > 1:
        return 2
    else:
        return 2 * x


def generate_y(n):
    y_ = []

    for i in range(n):
        y_.append(get_y())

    y_.sort()
    return y_


def get_frequency(y_: list, i: int):
    count = 1

    while i < len(y_) - 1 and y_[i] == y_[i + 1]:
        i += 1
        count += 1

    return count


def print_variation_range(y_):
    print('Вариационный ряд:')
    table = PrettyTable(['yᵢ', 'nᵢ'])

    i = 0
    while i < len(y_):
        n_i = get_frequency(y_, i)
        table.add_row([round(y_[i], 3), n_i])
        i += n_i

    print(table)


def F_y(y_):
    if y_ < -2:
        return 0
    if y_ > 2:
        return 1
    else:
        return 0.05 * y_ + 0.5


def get_X_Y(y_):
    X, Y = [-3], [0]
    p = 1 / len(y_)

    i = 0
    while i < len(y_):
        X.append(y_[i])
        i += get_frequency(y_, i)
        Y.append(p * i)
    X.append(3)
    Y.append(1)
    return X, Y


def update(val):
    n_ = int(round(n_slider.val))
    print(f'{n_slider.val} | {int(round(n_slider.val))}')

    y_ = generate_y(n_)
    F_x_line.set_data(*get_X_Y(y_))
    fig.canvas.draw_idle()


def plot_F_y(y_, add_analytical=False):
    X, Y = get_X_Y(y_)

    # For interactive tools
    global fig, F_x_line, n_slider
    fig, _ = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Analytical F(x)
    if add_analytical:
        ls = np.linspace(-3, 3, 6000)
        plt.plot(ls, [F_y(y_i) for y_i in ls], label='Аналитическая ф-ия распределения')
        plt.suptitle('Аналитическая и эмперическая функции распредления F(y)')
    else:
        plt.suptitle('Эмперическая функция распредления F(y)')

    # Empirical F(x)
    plt.xlabel('y')
    plt.ylabel('F(y)')
    plt.legend()
    F_x_line, = plt.step(X, Y, where='post', label=f'Эмперическая ф-ия распределения')

    # Interactive tools
    n_slider = Slider(ax=plt.axes([0.1, 0.15, 0.8, 0.03]), label='n = ', valmin=1, valmax=1000, valinit=n_init)
    n_slider.on_changed(update)

    button = Button(plt.axes([0.1, 0.05, 0.12, 0.05]), 'Generate')
    button.on_clicked(update)
    plt.show()


if __name__ == '__main__':
    a, b = -5, 5
    n_init = 10
    y = generate_y(n_init)
    print_variation_range(y)
    plot_F_y(y, add_analytical=True)
