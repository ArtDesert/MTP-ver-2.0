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

def get_numerical_solution_with_common_schema(I, K):
    # Вспомогательные величины
    h_x = l_x / I
    h_t = T / K
    gamma = beta * h_t / h_x ** 2
    n = I - 1
    x_arr = get_arr(0, I + 1, h_x)
    t_arr = get_arr(0, K + 1, h_t)
    V = np.zeros((K + 1, I + 1))

    d = [0] * n
    g = 1 + 2 * gamma
    a = 1 / (1 + h * h_x)
    d[0] = d[n - 1] = g - gamma * a
    for i in range(1, n - 1):
        d[i] = g

    c = [0] * n
    e = [0] * n
    for i in range(n):
        c[i] = e[i] = -gamma
    c[0] = e[n - 1] = 0

    # Инициализируем нулевой слой (начальное условие)
    for i in range(I + 1):
        V[0, i] = psi(x_arr[i])

    for k in range(1, K + 1):
        f = V[k - 1, 1 : I]
        x = get_tridiagonal_solve(c, d, e, f)
        for i in range(n):
            V[k, i + 1] = x[i]

        # Добавляем граничные условия
        V[k, 0] =   V[k, 1]   / (1 + h * h_x)
        V[k, I] = V[k, I - 1] / (1 + h * h_x)
    return V

def get_numerical_solution_with_modified_schema(I, K):
    # Вспомогательные величины
    h_x = l_x / I
    h_t = T / K
    gamma = beta * h_t / h_x ** 2
    n = I - 1
    x_arr = get_arr(0, I + 1, h_x)
    t_arr = get_arr(0, K + 1, h_t)
    V = np.zeros((K + 1, I + 1))

    d = [0] * n
    g = 1 + 2 * gamma
    a = 1 / (g + 2 * gamma * h * h_x)
    d[0] = d[n - 1] = g - 2 * a * gamma ** 2
    for i in range(1, n - 1):
        d[i] = g

    c = [0] * n
    e = [0] * n
    for i in range(n):
        c[i] = e[i] = -gamma
    c[0] = e[n - 1] = 0

    # Инициализируем нулевой слой (начальное условие)
    for i in range(I + 1):
        V[0, i] = psi(x_arr[i])

    for k in range(1, K + 1):
        f = V[k - 1, 1 : I].copy()
        f[0] += gamma * a * V[k - 1, 0]
        f[n - 1] += gamma * a * V[k - 1, I]
        x = get_tridiagonal_solve(c, d, e, f)
        for i in range(n):
            V[k, i + 1] = x[i]

        # Добавляем граничные условия
        V[k, 0] = a * (V[k - 1, 0] + 2 * gamma * V[k, 1])
        V[k, I] = a * (V[k - 1, I] + 2 * gamma * V[k, I - 1])
    return V

def draw_numerical_solution(V):
    t_values = [0, 5, 20, 75, 200]
    x_values = [0, 1, 2, 3, 4, 5]
    x_arr = get_arr(0, I + 1, l_x / I)
    t_arr = get_arr(0, K + 1, T / K)

    plt.figure(1, label="График численного решения: зависимости температуры от x при фиксированных значениях t")
    plt.xlabel('Координата x, см')
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

def draw_analytical_solution_for_fixed_t(n):
    #Рассматриваем правую половину отрезка, т.е. l_x / 2 < x < l_x (она симметрична левой)
    x_arr = get_arr(l_x / 2, I + 1, l_x / (2 * I))
    t = 5

    plt.figure(3, label=f"График аналитического решения: зависимости температуры от x при t = {t}")
    plt.xlabel('Координата x, см')
    plt.ylabel('Температура v, ℃')

    #Строим правую половину
    V_analytic = [0] * (I + 1)
    for i in range(I + 1):
        V_analytic[i] = get_solve_value(x_arr[i], t, n)
    plt.plot(x_arr, V_analytic, label=f"Аналитическое решение, t = {t}, n = {n}", color="blue")

    #Строим симметрично левую половину
    x_arr = get_arr(0, I + 1, l_x / (2 * I))
    V_analytic = [0] * (I + 1)
    for i in range(I + 1):
        V_analytic[i] = get_solve_value(x_arr[i], t, n)
    plt.plot(x_arr, V_analytic, color="blue")

    splitting_values = [(10, 10), (20, 40), (40, 160), (80, 640), (160, 2560), (320, 10240)]
    for (cur_I, cur_K) in splitting_values:
        V_numerical = get_numerical_solution_with_modified_schema(cur_I, cur_K)
        x_arr = get_arr(0, (cur_I + 1), l_x / cur_I)
        plt.plot(x_arr, V_numerical[(int)(t * cur_K / T), :], label=f"I = {cur_I}, K = {cur_K}")

    plt.legend()

def draw_analytical_solution_for_fixed_x(n):
    t_arr = get_arr(0, K + 1 , T / K)
    x = 3
    plt.figure(4, label=f"График аналитического решения: зависимости температуры от t при x = {x}")

    V_analytic = [0] * (K + 1)
    for i in range(K + 1):
        V_analytic[i] = get_solve_value(x, t_arr[i], n)
    plt.plot(t_arr, V_analytic, label=f"Аналитическое решение, x = {x}, n = {n}", color="blue")
    plt.xlabel('Время, с')
    plt.ylabel('Температура, ℃')

    splitting_values = [(10, 10), (20, 40), (40, 160), (80, 640), (160, 2560), (320, 10240)]
    for (cur_I, cur_K) in splitting_values:
        V_numerical = get_numerical_solution_with_modified_schema(cur_I, cur_K)
        t_arr = get_arr(0, (cur_K + 1), T / cur_K)
        plt.plot(t_arr, V_numerical[:, (int)(x * cur_I / l_x)], label=f"I = {cur_I}, K = {cur_K}")

    plt.legend()
    plt.show()


#V - сеточное решение
def get_norm(V_numerical, x_arr, t_arr, n):
    max = 0
    cur_I = len(x_arr) - 1
    cur_K = len(t_arr) - 1
    for x_i in range(int(cur_I / 2), cur_I + 1):
       for t_k in range(cur_K + 1):
           analytical_value = get_solve_value(x_arr[x_i], t_arr[t_k], n)
           numerical_value = V_numerical[t_k, x_i]
           cur_value = np.abs(numerical_value - analytical_value)
           max = cur_value if cur_value > max else max
    return max


def print_error_dependency_results_with_common_schema(n):
    #values_for_error_dependency = [(10, 10), (20, 20), (40, 40), (80, 80), (160, 160), (320, 320), (640, 640)]
    values_for_error_dependency = [(10, 5), (20, 20), (40, 100), (80, 400), (160, 1600)]
    result = []
    for (I, K) in values_for_error_dependency:
        V_numerical = get_numerical_solution_with_common_schema(I, K)
        x_arr = get_arr(0, I + 1, l_x / I)
        t_arr = get_arr(0, K + 1, T / K)
        eps = get_norm(V_numerical, x_arr, t_arr, n)
        print(f"I = {I}, K = {K}, eps = {eps}")
        result.append(eps)
    print("--------------------")
    for i in range(len(result) - 1):
        print(result[i] / result[i + 1])


def print_error_dependency_results_with_modified_schema(n):
    #values_for_error_dependency = [(10, 5), (20, 20), (40, 80), (80, 320), (160, 1280)]
    values_for_error_dependency = [(5, 3), (10, 12), (20, 48), (40, 192), (80, 768), (160, 3072)]
    result = []
    for (I, K) in values_for_error_dependency:
        V_numerical = get_numerical_solution_with_modified_schema(I, K)
        x_arr = get_arr(0, I + 1, l_x / I)
        t_arr = get_arr(0, K + 1, T / K)
        eps = get_norm(V_numerical, x_arr, t_arr, n)
        print(f"I = {I}, K = {K}, eps = {eps}")
        result.append(eps)
    print("--------------------")
    for i in range(len(result) - 1):
        print(result[i] / result[i + 1])


#Вводимые параметры
T = 200
l_x = 10
alpha = 0.004
k = 0.13
c = 1.84
u_0 = 0
I = 200
K = 200

beta = k / c
h = alpha / k

#V_numerical = get_numerical_solution_with_modified_schema(I, K)
#draw_numerical_solution(V_numerical)

#Поиск трансцендентных корней
n = 100
P_n = []
P_n.append(0)
d = 10 ** (-7)
eps = 10 ** (-12)
l = d
for i in range(n):
    r_asymp = np.pi * (2 * i + 1) / l_x
    P_n.append(meth_dich(l, r_asymp - d, eps))
    l = r_asymp + d


#draw_analytical_solution_for_fixed_t(n)
#draw_analytical_solution_for_fixed_x(n)

print_error_dependency_results_with_modified_schema(n)

