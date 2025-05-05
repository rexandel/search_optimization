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
    complementary_slackness = [(variables[i], vs[i]) for i in range(len(variables))]
    complementary_slackness += [(lambdas[i], ws[i]) for i in range(n_constraints)]

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

    print("\nКоэффициенты в условиях KKT (для анализа знаков):")
    for i, eq in enumerate(kkt_conditions):
        print(f"Уравнение {i + 1}: {eq} = 0")
        # Выводим коэффициенты для проверки знаков
        if i < len(variables):
            print(f"  Уравнение для переменной {variables[i]}")
        else:
            print(f"  Уравнение для ограничения {i - len(variables) + 1}")

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
    for var, v in complementary_slackness:
        if var in variables:
            i = variables.index(var)
            print(f"{var} * v{i + 1} = 0, {var} >= 0, v{i + 1} >= 0")
        else:
            i = lambdas.index(var)
            print(f"{var} * w{i + 1} = 0, {var} >= 0, w{i + 1} >= 0")

    # Возвращаем результаты
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


def add_artificial_variables(kkt_system, kkt_conditions):
    """
    Добавляет искусственные переменные z_i в уравнения системы KKT,
    где знаки свободных членов совпадают со знаками дополнительных переменных v_i или w_j.

    Args:
        kkt_system (dict): Результат работы функции dynamic_kkt_system
        kkt_conditions (list): Список уравнений системы KKT
    """
    variables = kkt_system['all_variables']
    vs = kkt_system['vs']
    ws = kkt_system['ws']

    # Создаем список для искусственных переменных z_i
    z_vars = []

    # Список модифицированных уравнений
    modified_equations = []

    # Словарь для хранения выражений искусственных переменных
    z_expressions = {}

    print("\n### Ввод искусственных переменных ###")

    # Анализируем каждое уравнение системы
    for i, eq in enumerate(kkt_conditions):
        # Преобразуем уравнение в формат левая_часть = правая_часть (первоначально все в левой части)
        left_part = eq
        right_part = 0

        # Проверяем наличие свободного члена
        free_term = left_part.as_coeff_add(*variables)[0]

        # Определяем, какая дополнительная переменная (v_i или w_j) присутствует в уравнении
        if i < len(vs):
            # Уравнение с v_i
            dop_var = vs[i]  # v_i проверяется отдельно, т.к. в уравнении входит с отрицательным знаком
            dop_var_name = f'v{i + 1}'
            dop_var_sign = -1  # v_i входит с отрицательным знаком в уравнение
        else:
            # Уравнение с w_j
            dop_var = ws[i - len(vs)]
            dop_var_name = f'w{i - len(vs) + 1}'
            dop_var_sign = 1  # w_j входит с положительным знаком в уравнение

        # Проверяем совпадение знаков:
        # Если доп. переменная входит с отрицательным знаком и свободный член отрицательный,
        # или доп. переменная входит с положительным знаком и свободный член положительный
        need_artificial = False

        if (dop_var_sign < 0 and free_term < 0) or (dop_var_sign > 0 and free_term > 0):
            need_artificial = True

        print(f"Анализ уравнения {i + 1}: {eq} = 0")
        print(f"  Свободный член: {free_term}")
        print(f"  Доп. переменная: {dop_var_name} входит со знаком {'+' if dop_var_sign > 0 else '-'}")
        print(f"  Совпадение знаков: {'Да' if need_artificial else 'Нет'}")

        if need_artificial:
            # Создаем искусственную переменную z_i
            z_var = sp.symbols(f'z{i + 1}', real=True)
            z_vars.append(z_var)

            # Добавляем z_i в уравнение (z_i входит с положительным знаком)
            modified_eq = left_part + z_var

            # Запоминаем выражение для z_i
            z_expressions[z_var] = -left_part

            print(
                f"В уравнение {i + 1} введена искусственная переменная z{i + 1}, т.к. совпал знак {dop_var_name} со знаком свободного члена {free_term}")
            print(f"Уравнение с z{i + 1}: {modified_eq} = {right_part}")
        else:
            modified_eq = left_part
            print(f"В уравнение {i + 1} не требуется вводить искусственную переменную")
            print(f"Уравнение остается: {modified_eq} = {right_part}")

        # Добавляем модифицированное уравнение
        modified_equations.append((modified_eq, right_part))

    # Формируем вспомогательную целевую функцию F(z) = sum(z_i)
    if z_vars:
        F_z = sum(z_vars)

        # Выражаем F(z) через остальные переменные
        F_z_expanded = sum(z_expressions[z] for z in z_vars)

        print("\n### Вспомогательная функция ЛП ###")
        print(f"F(z) = {F_z}")
        print(f"F(z) = {F_z_expanded}")

        # Приводим уравнения к окончательному виду (переносим свободные члены вправо)
        print("\n### Окончательная система уравнений ###")
        for i, (eq, right) in enumerate(modified_equations):
            # Разделяем уравнение на части с переменными и свободный член
            coeff_dict = {}
            for var in variables + z_vars:
                coeff = eq.coeff(var)
                if coeff != 0:
                    coeff_dict[var] = coeff

            # Свободный член
            free_term = eq.as_coeff_add(*(variables + z_vars))[0]

            # Формируем левую часть уравнения без свободного члена
            left_part_no_free = sum(coeff * var for var, coeff in coeff_dict.items())

            # Правая часть уравнения (с противоположным знаком свободного члена)
            right_part_final = right - free_term

            print(f"Уравнение {i + 1}: {left_part_no_free} = {right_part_final}")

        # Добавляем условия неотрицательности искусственных переменных
        print("\n### Условия неотрицательности для искусственных переменных ###")
        for z in z_vars:
            print(f"{z} >= 0")
    else:
        print("\nИскусственных переменных не требуется вводить в данной задаче.")

    return {
        'z_vars': z_vars,
        'z_expressions': z_expressions,
        'F_z': F_z if z_vars else None,
        'F_z_expanded': F_z_expanded if z_vars else None,
        'modified_equations': modified_equations
    }


def solve_kkt_example():
    # Определение переменных
    x, y = sp.symbols('x y', real=True)
    variables = [x, y]

    # Целевая функция
    obj_func = 2 * x ** 2 + 3 * y ** 2 + 4 * x * y - 6 * x - 3 * y
    # obj_func = 2 * x**2 + 2 * x * y + 2 * y**2 - 4 * x - 6 * y

    # Ограничения в форме g(x) <= 0
    constraints = [
        # x + 2*y - 2
        x + y - 1,
        2 * x + 3 * y - 4
    ]

    print("\n#########################################################")
    print("# РЕШЕНИЕ ЗАДАЧИ ОПТИМИЗАЦИИ МЕТОДОМ ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ #")
    print("#########################################################\n")

    # Формирование системы KKT
    kkt_system = dynamic_kkt_system(obj_func, constraints, variables)

    # Добавление искусственных переменных
    artificial_system = add_artificial_variables(kkt_system, kkt_system['kkt_conditions'])

    print("\n### Система готова для решения ###")
    print("Чтобы решить систему, нужно исследовать все возможные комбинации")
    print("условий дополняющей нежесткости.")


if __name__ == "__main__":
    solve_kkt_example()