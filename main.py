import numpy as np
import matplotlib.pyplot as plt

def psi(x):
    return 5 * np.sin(np.pi * x / l_x)

def get_arr(left, n, h):
    #c = 0.0
    arr = [0] * n
    arr[0] = left
    for i in range(1, n):
        arr[i] = arr[i - 1] + h
    return arr


#c - массив поддиагональных элементов (первый элемент всегда = 0)
#d - массив диагональных элементов
#e - массив наддиагональных элементов (последний элемент всегда = 0)
#f - правая часть
def get_tridiagonal_solve(c, d, e, f): #OK
    n = len(f)
    #Массивы прогоночных коэффициентов
    a = [0] * n
    b = [0] * n
    x = [0] * n
    #1 шаг
    a[0] = e[0] / d[0]
    b[0] = f[0] / d[0]
    #2 шаг - прямой ход
    for i in range(1, n):
        denominator = d[i] - c[i] * a[i - 1]
        a[i] = e[i] / denominator
        b[i] = (f[i] - c[i] * b[i - 1]) / denominator
    #3 шаг
    x[n - 1] = b[n - 1]
    #4 шаг - обратный ход
    for i in range(n - 2, -1, -1):
        x[i] = b[i] - a[i] * x[i + 1]
    return x

def get_numerical_solution(I, K):
    # Вспомогательные величины
    h_x = l_x / I
    h_t = T / K
    gamma = beta * h_t / h_x ** 2
    n = I - 1
    x_arr = get_arr(0, I + 1, h_x)
    t_arr = get_arr(0, K + 1, h_t)
    V = np.zeros((K + 1, I + 1))
    a = (1 - 1 / (2 * gamma)) / (h * h_x + 1 - 1 / (2 * gamma))
    b = (1 + 1 / (2 * gamma)) / (h * h_x + 1 + 1 / (2 * gamma))

    d = [0] * n
    g = 1 + 2 * gamma
    d[0] = g - gamma * a
    d[n - 1] = g - gamma * b
    for i in range(1, n - 1):
        d[i] = g

    c = [0] * n
    e = [0] * n
    for i in range(n):
        c[i] = e[i] = -gamma
    c[0] = e[n - 1] = 0
    # f = [0] * n

    # Инициализируем нулевой слой (начальное условие)
    for i in range(I + 1):
        V[0, i] = psi(x_arr[i])

    for k in range(1, K + 1):
        f = V[k - 1, 1: I]
        x = get_tridiagonal_solve(c, d, e, f)
        for i in range(n):
            V[k, i + 1] = x[i]
        # Добавляем граничные условия
        V[k, 0] = a * V[k, 1]
        V[k, I] = b * V[k, I - 1]
    return V

def draw_numerical_solution(V):
    t_values = [0, 5, 20, 75, 200]
    x_values = [0, 1, 2, 3, 4, 5]
    x_arr = get_arr(0, I + 1, l_x / I)
    t_arr = get_arr(0, K + 1, T / K)

    plt.figure(1, label="График численного решения: зависимости температуры от x при фиксированных значениях t")
    plt.xlabel('Длина x, см')
    plt.ylabel('Температура v, ℃')
    for t in t_values:
        plt.plot(x_arr, V[(int)(t / (T / K)), :], label=f"t = {t}")
    plt.legend()

    plt.figure(2, label="График численного решения: зависимости температуры от времени t при фиксированных значениях x")
    plt.subplot(1, 1, 1)
    for x in x_values:
        plt.plot(t_arr, V[:, (int)(x / (l_x / I))], label=f'x = {x}')
    plt.xlabel('Время t, с')
    plt.ylabel('Температура v, ℃')
    plt.legend()

    plt.show()

def get_Cn(n):
    p_n = P_n[n]
    return (5 * np.pi * l_x * np.cos(p_n * l_x / 2)) / (get_Gamma_n(n) * (np.pi ** 2 - (l_x * p_n) ** 2))

def get_Gamma_n(n):
    p_n = P_n[n]
    return l_x / 4 + np.sin(p_n * l_x) / (4 * p_n)

def get_z(x):
    return x - l_x / 2

#n - число слагаемых для суммирования членов ряда Фурье
def get_solve_value(x, t, n):
    value = 0
    z = get_z(x)
    for i in range(1, n):
        p_i = P_n[i]
        value += get_Cn(i) * np.exp(-beta * p_i ** 2 * t) * np.cos(p_i * z)
    return value

def meth_dich(a, b, eps):
    while abs(b - a) > 2 * eps:
        mid = (a + b) / 2
        if fun(mid) * fun(b) <= 0:
            a = mid
        else:
            b = mid
    return mid

def fun(x):
    return np.tan(l_x * x / 2) - h / x

def draw_analytical_solution(n):
    #Рассматриваем правую половину отрезка, т.е. l_x / 2 < x < l_x (она симметрична левой)
    x_arr = get_arr(l_x / 2, I + 1, l_x / (2 * I))
    t_values = [0, 5, 20, 75, 200]
    x_values = [5, 6, 7, 8, 9, 10]

    plt.figure(3, label="График аналитического решения: зависимости температуры от x при t = 100")
    plt.xlabel('Длина x, см')
    plt.ylabel('Температура v, ℃')

    t = 5
    V_analytic = [0] * (I + 1)
    for i in range(I + 1):
        V_analytic[i] = get_solve_value(x_arr[i], t, n)
    plt.plot(x_arr, V_analytic, label=f"Аналитическое решение при t = {t}, n = {n}", color="blue")

    x_arr = get_arr(0, I + 1, l_x / (2 * I))
    V_analytic = [0] * (I + 1)
    for i in range(I + 1):
        V_analytic[i] = get_solve_value(x_arr[i], t, n)
    plt.plot(x_arr, V_analytic, color="blue")



    splitting_values = [10, 40, 160, 320, 640, 1280]
    for splitting_value in splitting_values:
        V_numerical = get_numerical_solution(splitting_value, splitting_value)
        x_arr = get_arr(0, (splitting_value + 1), l_x / splitting_value)
        plt.plot(x_arr, V_numerical[(int)(t * splitting_value / T), :], label=f"I = {splitting_value}, K = {splitting_value}")





    #t_arr = get_arr(0, K + 1 , T / K)
    #t_arr = get_arr(0, K + 1 , T / K)
    #plt.figure(2, label="График зависимости температуры от времени t при фиксированных значениях x")
    #plt.subplot(1, 1, 1)
    #for x in x_values:
    #    plt.plot(t_arr, V[:, (int)(x / h_x)], label=f'x = {x}')
    #plt.xlabel('Время, с')
    #plt.ylabel('Температура, ℃')


    plt.legend()
    plt.show()


#Вводимые параметры
T = 200
l_x = 10
alpha = 0.004
k = 0.13
c = 1.84
u_0 = 0
I = 50
K = 200

beta = k / c
h = alpha / k

V = get_numerical_solution(I, K)
#draw_numerical_solution(V)

#Поиск трансцендентных корней
P_n = []
P_n.append(0)
d = 10 ** (-7)
eps = 10 ** (-12)
l = d
for i in range(100):
    r_asymp = np.pi * (2 * i + 1) / l_x
    P_n.append(meth_dich(l, r_asymp - d, eps))
    l = r_asymp + d

draw_analytical_solution(100)