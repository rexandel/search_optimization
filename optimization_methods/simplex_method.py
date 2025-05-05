import sympy as sp
from prettytable import PrettyTable


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
        'ws': ws,
        'variables': variables  # Добавьте этот ключ
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
        # Преобразуем уравнение в формат левая_часть = правая_часть
        left_part = eq
        right_part = 0

        # Проверяем наличие свободного члена
        free_term = left_part.as_coeff_add(*variables)[0]

        # Определяем, какая дополнительная переменная (v_i или w_j) присутствует
        if i < len(vs):
            dop_var = vs[i]
            dop_var_name = f'v{i + 1}'
            dop_var_sign = -1  # v_i входит с отрицательным знаком
        else:
            dop_var = ws[i - len(vs)]
            dop_var_name = f'w{i - len(vs) + 1}'
            dop_var_sign = 1  # w_j входит с положительным знаком

        # Проверяем совпадение знаков
        need_artificial = False
        if (dop_var_sign < 0 and free_term < 0) or (dop_var_sign > 0 and free_term > 0):
            need_artificial = True

        print(f"Анализ уравнения {i + 1}: {eq} = 0")
        print(f"  Свободный член: {free_term}")
        print(f"  Доп. переменная: {dop_var_name} входит со знаком {'+' if dop_var_sign > 0 else '-'}")
        print(f"  Совпадение знаков: {'Да' if need_artificial else 'Нет'}")

        # Переносим свободный член в правую часть
        right_part_final = -free_term
        modified_eq = left_part - free_term  # Убираем свободный член из левой части

        if need_artificial:
            # Создаем искусственную переменную z_i
            z_var = sp.symbols(f'z{i + 1}', real=True)
            z_vars.append(z_var)

            # Добавляем z_i в уравнение (z_i входит с положительным знаком)
            modified_eq = modified_eq + z_var

            # Запоминаем выражение для z_i
            z_expressions[z_var] = -left_part

            print(f"В уравнение {i + 1} введена искусственная переменная z{i + 1}")
            print(f"Уравнение с z{i + 1}: {modified_eq} = {right_part_final}")
        else:
            print(f"В уравнение {i + 1} не требуется вводить искусственную переменную")
            print(f"Уравнение остается: {modified_eq} = {right_part_final}")

        # Добавляем модифицированное уравнение
        modified_equations.append((modified_eq, right_part_final))

    # Формируем вспомогательную целевую функцию F(z) = sum(z_i)
    if z_vars:
        F_z = sum(z_vars)
        F_z_expanded = sum(z_expressions[z] for z in z_vars)

        print("\n### Вспомогательная функция ЛП ###")
        print(f"F(z) = {F_z}")
        print(f"F(z) = {F_z_expanded}")

        # Выводим окончательную систему уравнений
        print("\n### Окончательная система уравнений ###")
        for i, (eq, right) in enumerate(modified_equations):
            left_part_no_free = eq
            print(f"Уравнение {i + 1}: {left_part_no_free} = {right}")

        # Условия неотрицательности
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

def build_simplex_table(kkt_system, artificial_system):
    """
    Строит первую симплекс-таблицу для задачи ЛП на основе системы KKT с искусственными переменными.

    Args:
        kkt_system (dict): Результат работы функции dynamic_kkt_system.
        artificial_system (dict): Результат работы функции add_artificial_variables.

    Returns:
        PrettyTable: Симплекс-таблица в виде объекта PrettyTable.
    """
    variables_order = (
        kkt_system['variables'] +
        kkt_system['lambdas'] +
        kkt_system['vs'] +
        kkt_system['ws'] +
        artificial_system['z_vars']
    )

    headers = ['Базис', 'Св.член'] + [str(var) for var in variables_order]
    table = PrettyTable(headers)
    table.float_format = ".2f"  # Форматирование чисел

    # Обрабатываем уравнения с искусственными переменными и ограничения
    for eq, rhs in artificial_system['modified_equations']:
        coeffs = [eq.coeff(var) for var in variables_order]
        basis_var = None

        # Определяем базисную переменную (искусственную или w)
        for z_var in artificial_system['z_vars']:
            if eq.coeff(z_var) == 1:
                basis_var = z_var
                break
        if not basis_var:
            for w_var in kkt_system['ws']:
                if eq.coeff(w_var) == 1:
                    basis_var = w_var
                    break

        row = [str(basis_var), rhs] + coeffs
        table.add_row(row)

    # Добавляем целевую функцию F
    if artificial_system['F_z_expanded'] is not None:
        f_coeffs = [-artificial_system['F_z_expanded'].coeff(var) for var in variables_order]
        f_constant = artificial_system['F_z_expanded'].as_coeff_add(*variables_order)[0]
        row = ['F', f_constant] + f_coeffs
        table.add_row(row)

    return table


def solve_simplex_table(kkt_system, artificial_system, simplex_table):
    variables_order = (
        kkt_system['variables'] +
        kkt_system['lambdas'] +
        kkt_system['vs'] +
        kkt_system['ws'] +
        artificial_system['z_vars']
    )
    # Условия дополняющей нежесткости
    complementary_slackness = kkt_system['complementary_slackness']
    # Преобразуем таблицу в список строк для удобства обработки
    table_rows = simplex_table._rows
    headers = simplex_table.field_names
    iteration = 1
    while True:
        print(f"\n### Итерация {iteration} ###")
        print(simplex_table)
        # Извлекаем строку целевой функции
        f_row = [row for row in table_rows if row[0] == 'F'][0]
        f_coeffs = {headers[i]: float(coef) for i, coef in enumerate(f_row[2:], 2)}
        # Проверяем оптимальность (все коэффициенты в строке F <= 0)
        is_optimal = all(coef <= 0 for coef in f_row[2:])
        if is_optimal:
            print("Найдено оптимальное решение.")
            solution = {}
            for row in table_rows:
                if row[0] != 'F':
                    basis_var = row[0]
                    free_term = row[1]
                    solution[basis_var] = free_term
            for var in variables_order:
                if str(var) not in solution:
                    solution[str(var)] = 0.0
            print("Решение:")
            for var, value in solution.items():
                print(f"{var} = {value:.6f}")
            print(f"Значение целевой функции: {f_row[1]:.6f}")
            return solution
        # Выбираем ведущий столбец с учетом условий дополняющей нежесткости
        max_coeff = -float('inf')
        pivot_col_idx = None
        pivot_col_var = None
        for i, var in enumerate(variables_order, 2):
            coef = f_coeffs[str(var)]
            if coef > max_coeff:
                # Проверяем условия дополняющей нежесткости
                can_use = True
                for var1, var2 in complementary_slackness:
                    if var == var1:
                        # Проверяем, является ли var2 базисной и положительной
                        for row in table_rows:
                            if row[0] == str(var2) and row[1] > 0:
                                can_use = False
                                break
                    elif var == var2:
                        for row in table_rows:
                            if row[0] == str(var1) and row[1] > 0:
                                can_use = False
                                break
                if can_use:
                    max_coeff = coef
                    pivot_col_idx = i
                    pivot_col_var = var
        if pivot_col_idx is None:
            print("Нет подходящего ведущего столбца с учетом условий дополняющей нежесткости.")
            return None
        print(f"Ведущий столбец: {pivot_col_var} (коэффициент = {max_coeff:.2f})")
        # Выбираем ведущую строку
        min_ratio = float('inf')
        pivot_row_idx = None
        pivot_row_var = None
        for i, row in enumerate(table_rows):
            if row[0] == 'F':
                continue
            free_term = float(row[1])
            pivot_col_val = float(row[pivot_col_idx])
            if pivot_col_val > 0:
                ratio = free_term / pivot_col_val
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_idx = i
                    pivot_row_var = row[0]
        if pivot_row_idx is None:
            print("Задача неограничена (нет ведущей строки).")
            return None
        print(f"Ведущая строка: {pivot_row_var} (отношение = {min_ratio:.6f})")
        # Опорный элемент
        pivot_element = float(table_rows[pivot_row_idx][pivot_col_idx])
        print(f"Опорный элемент: {pivot_element:.6f}")
        # Создаем новую таблицу
        new_table = PrettyTable(headers)
        new_table.float_format = ".2f"
        new_rows = []
        for i, row in enumerate(table_rows):
            new_row = row[:]  # Копия строки
            if i == pivot_row_idx:
                # Делим ведущую строку на опорный элемент
                new_row = [row[0]] + [float(val) / pivot_element for val in row[1:]]
            else:
                # Обновляем остальные строки
                factor = float(row[pivot_col_idx])
                new_row = [row[0]] + [
                    float(row[j]) - factor * float(table_rows[pivot_row_idx][j]) / pivot_element
                    for j in range(1, len(row))
                ]
            new_rows.append(new_row)
        # Обновляем базисную переменную в ведущей строке
        new_rows[pivot_row_idx][0] = str(pivot_col_var)
        # Добавляем строки в новую таблицу
        for row in new_rows:
            new_table.add_row(row)
        table_rows = new_rows
        simplex_table = new_table
        iteration += 1

def solve_kkt_example():
    # Определение переменных
    x, y = sp.symbols('x y', real=True)
    variables = [x, y]

    # Целевая функция
    obj_func = 2 * x ** 2 + 3 * y ** 2 + 4 * x * y - 6 * x - 3 * y
    # obj_func = 2 * x**2 + 2 * x * y + 2 * y**2 - 4 * x - 6 * y

    # Ограничения в форме g(x) <= 0
    constraints = [
        x + y - 1,
        2 * x + 3 * y - 4

        # x + 2*y - 2
    ]

    print("\n#########################################################")
    print("# РЕШЕНИЕ ЗАДАЧИ ОПТИМИЗАЦИИ МЕТОДОМ ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ #")
    print("#########################################################\n")
    kkt_system = dynamic_kkt_system(obj_func, constraints, variables)
    artificial_system = add_artificial_variables(kkt_system, kkt_system['kkt_conditions'])
    simplex_table = build_simplex_table(kkt_system, artificial_system)
    print("\n### Первая симплекс-таблица ###")
    print(simplex_table)
    solution = solve_simplex_table(kkt_system, artificial_system, simplex_table)
    print("\n### Результаты решения ###")
    if solution:
        print("Найдено решение:")
        for var, value in solution.items():
            print(f"{var} = {value:.6f}")
    else:
        print("Решение не найдено.")
    return solution


if __name__ == "__main__":
    solve_kkt_example()