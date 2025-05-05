import sympy as sp
import json
from itertools import product


def dynamic_kkt_system(obj_func, constraints, variables):
    """
    Формирует систему KKT для заданной целевой функции и ограничений.

    Args:
        obj_func (sympy expr): Целевая функция
        constraints (list): Список ограничений вида g(x) <= 0
        variables (list): Список переменных оптимизации
    """
    # Динамическое создание переменных
    n_constraints = len(constraints)
    lambdas = [sp.symbols(f'lambda{i + 1}', real=True) for i in range(n_constraints)]

    # Переменные v для условий неотрицательности
    vs = [sp.symbols(f'v{i + 1}', real=True) for i in range(len(variables))]

    # Переменные w для ограничений
    ws = [sp.symbols(f'w{i + 1}', real=True) for i in range(n_constraints)]

    # Построение функции Лагранжа
    L = obj_func + sum(lambdas[i] * constraints[i] for i in range(n_constraints))

    # Частные производные по переменным
    dL_dx = [sp.diff(L, var) for var in variables]

    # Частные производные по множителям Лагранжа (g_i(x) <= 0)
    dL_dlambda = constraints

    # Формирование условий KKT с дополнительными переменными
    kkt_conditions = []
    for i in range(len(variables)):
        kkt_conditions.append(dL_dx[i] - vs[i])  # dL/dx_i - v_i = 0 (v_i вводится с минусом)

    for i in range(n_constraints):
        kkt_conditions.append(dL_dlambda[i] + ws[i])  # g_i(x) + w_i = 0 (w_i вводится с плюсом)

    # Условия допустимости
    primal_feasibility = [constr <= 0 for constr in constraints] + [var >= 0 for var in variables]
    dual_feasibility = [lam >= 0 for lam in lambdas] + [v >= 0 for v in vs] + [w >= 0 for w in ws]

    # Условия дополняющей нежесткости
    complementary_slackness = [variables[i] * vs[i] == 0 for i in range(len(variables))]
    complementary_slackness += [lambdas[i] * ws[i] == 0 for i in range(n_constraints)]

    # Все переменные для решения системы
    all_variables = variables + lambdas + vs + ws

    # Вывод системы
    print("### Задача оптимизации ###")
    print(f"Целевая функция: {obj_func}")
    print("Ограничения:")
    for i, g in enumerate(constraints):
        print(f"g{i + 1}(x) = {g} <= 0")

    print("\n### Переменные системы ###")
    print(f"Переменные оптимизации: {variables}")
    print(f"Множители Лагранжа: {lambdas}")
    print(f"Переменные неотрицательности v: {vs}")
    print(f"Переменные нежесткости w: {ws}")

    print("\n### Функция Лагранжа ###")
    print(sp.pretty(L))

    print("\n### Частные производные ###")
    for i, var in enumerate(variables):
        print(f"dL/d{var} = {dL_dx[i]}")

    for i, lam in enumerate(lambdas):
        print(f"dL/d{lam} = {dL_dlambda[i]}")

    print("\n### Условия KKT ###")
    for i, eq in enumerate(kkt_conditions):
        print(f"Уравнение {i + 1}: {eq} = 0")

    print("\n### Условия допустимости ###")
    print("Примарные условия:")
    for cond in primal_feasibility:
        print(sp.pretty(cond))

    print("\nДвойственные условия:")
    for cond in dual_feasibility:
        print(sp.pretty(cond))

    print("\n### Условия дополняющей нежесткости ###")
    for cond in complementary_slackness:
        print(sp.pretty(cond))

    return {
        'lagrangian': L,
        'kkt_conditions': kkt_conditions,
        'primal_feasibility': primal_feasibility,
        'dual_feasibility': dual_feasibility,
        'complementary_slackness': complementary_slackness,
        'all_variables': all_variables,
        'lambdas': lambdas,
        'vs': vs,
        'ws': ws
    }


# Пример использования с захардкоженной функцией и ограничениями
def example_usage():
    # Определение переменных
    x, y = sp.symbols('x y', real=True)
    variables = [x, y]

    # Целевая функция
    obj_func = 2 * x ** 2 + 2 * x * y + 2 * y ** 2 - 4 * x - 6 * y
    # obj_func = 2 * x**2 + 3 * y**2 + 4*x*y - 6*x - 3*y

    # Ограничения в форме g(x) <= 0
    constraints = [
        x + 2 * y - 2,  # x + 2y - 2 <= 0
        # x + y - 1,
        # 2 * x + 3 * y - 4
        # Здесь можно добавить дополнительные ограничения
    ]

    # Формирование системы KKT
    kkt_system = dynamic_kkt_system(obj_func, constraints, variables)

    print("\n### Система готова для решения ###")
    print("Чтобы решить систему, нужно исследовать все возможные комбинации")
    print("условий дополняющей нежесткости.")


if __name__ == "__main__":
    example_usage()