import random
import matplotlib.pyplot as plt
import numpy as np
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


def plot_F_y(y, add_analytical=False):
    x_, y_ = [], []
    p = 1 / len(y)

    i = 0
    while i < len(y):
        x_.append(y[i])
        i += get_frequency(y, i)
        y_.append(p * i)

    plt.step([-3] + x_ + [3], [0, 0] + y_, label=f'Эмперическая ф-ия распределения (n = {n})')

    if add_analytical:
        ls = np.linspace(-3, 3, 6000)
        plt.plot(ls, [F_y(y_i) for y_i in ls], label='Аналитическая ф-ия распределения')
        plt.suptitle('Сравнение аналитической и эмперической функции распредления F(y)')
    else:
        plt.suptitle('Эмперическая функция распредления F(y)')

    plt.xlabel('y')
    plt.ylabel('F(y)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    a, b = -5, 5
    n = 100
    y = generate_y(n)
    print_variation_range(y)
    plot_F_y(y)
    plot_F_y(y, add_analytical=True)
