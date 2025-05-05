import sympy as sp
from prettytable import PrettyTable
from fractions import Fraction
import copy

def dynamic_kkt_system(obj_func, constraints, variables):
    n_constraints = len(constraints)
    lambdas = [sp.symbols(f'lambda{i + 1}', real=True) for i in range(n_constraints)]
    vs = [sp.symbols(f'v{i + 1}', real=True) for i in range(len(variables))]
    ws = [sp.symbols(f'w{i + 1}', real=True) for i in range(n_constraints)]
    L = obj_func + sum(lambdas[i] * constraints[i] for i in range(n_constraints))
    dL_dx = [sp.diff(L, var) for var in variables]
    dL_dlambda = constraints
    kkt_conditions = []
    for i in range(len(variables)):
        kkt_conditions.append(dL_dx[i] - vs[i])
    for i in range(n_constraints):
        kkt_conditions.append(dL_dlambda[i] + ws[i])
    primal_feasibility = [constr <= 0 for constr in constraints] + [var >= 0 for var in variables]
    dual_feasibility = [lam >= 0 for lam in lambdas] + [v >= 0 for v in vs] + [w >= 0 for w in ws]
    complementary_slackness = [(variables[i], vs[i]) for i in range(len(variables))]
    complementary_slackness += [(lambdas[i], ws[i]) for i in range(n_constraints)]
    all_variables = variables + lambdas + vs + ws
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
        'variables': variables
    }

def add_artificial_variables(kkt_system, kkt_conditions):
    variables = kkt_system['all_variables']
    vs = kkt_system['vs']
    ws = kkt_system['ws']
    z_vars = []
    modified_equations = []
    z_expressions = {}
    print("\n### Ввод искусственных переменных ###")
    for i, eq in enumerate(kkt_conditions):
        left_part = eq
        right_part = 0
        free_term = left_part.as_coeff_add(*variables)[0]
        if i < len(vs):
            dop_var = vs[i]
            dop_var_name = f'v{i + 1}'
            dop_var_sign = -1
        else:
            dop_var = ws[i - len(vs)]
            dop_var_name = f'w{i - len(vs) + 1}'
            dop_var_sign = 1
        need_artificial = (dop_var_sign < 0 and free_term < 0) or (dop_var_sign > 0 and free_term > 0)
        print(f"Анализ уравнения {i + 1}: {eq} = 0")
        print(f"  Свободный член: {free_term}")
        print(f"  Доп. переменная: {dop_var_name} входит со знаком {'+' if dop_var_sign > 0 else '-'}")
        print(f"  Совпадение знаков: {'Да' if need_artificial else 'Нет'}")
        right_part_final = -free_term
        modified_eq = left_part - free_term
        if need_artificial:
            z_var = sp.symbols(f'z{i + 1}', real=True)
            z_vars.append(z_var)
            modified_eq = modified_eq + z_var
            z_expressions[z_var] = -left_part
            print(f"В уравнение {i + 1} введена искусственная переменная z{i + 1}")
            print(f"Уравнение с z{i + 1}: {modified_eq} = {right_part_final}")
        else:
            print(f"В уравнение {i + 1} не требуется вводить искусственную переменную")
            print(f"Уравнение остается: {modified_eq} = {right_part_final}")
        modified_equations.append((modified_eq, right_part_final))
    if z_vars:
        F_z = sum(z_vars)
        F_z_expanded = sum(z_expressions[z] for z in z_vars)
        print("\n### Вспомогательная функция ЛП ###")
        print(f"F(z) = {F_z}")
        print(f"F(z) = {F_z_expanded}")
        print("\n### Окончательная система уравнений ###")
        for i, (eq, right) in enumerate(modified_equations):
            print(f"Уравнение {i + 1}: {eq} = {right}")
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
    variables_order = (
        kkt_system['variables'] +
        kkt_system['lambdas'] +
        kkt_system['vs'] +
        kkt_system['ws'] +
        artificial_system['z_vars']
    )
    headers = ['Базис', 'Св.член'] + [str(var) for var in variables_order]
    table = PrettyTable(headers)
    table.float_format = ".2f"
    for eq, rhs in artificial_system['modified_equations']:
        coeffs = [eq.coeff(var) for var in variables_order]
        basis_var = None
        for z_var in artificial_system['z_vars']:
            if eq.coeff(z_var) == 1:
                basis_var = z_var
                break
        if not basis_var:
            for w_var in kkt_system['ws']:
                if eq.coeff(w_var) == 1:
                    basis_var = w_var
                    break
        row = [str(basis_var), float(rhs)] + [float(coef) for coef in coeffs]
        table.add_row(row)
    if artificial_system['F_z_expanded'] is not None:
        f_coeffs = [-artificial_system['F_z_expanded'].coeff(var) for var in variables_order]
        f_constant = artificial_system['F_z_expanded'].as_coeff_add(*variables_order)[0]
        row = ['F', float(f_constant)] + [float(coef) for coef in f_coeffs]
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
    z_vars = artificial_system['z_vars']
    complementary_slackness = kkt_system['complementary_slackness']
    table_rows = simplex_table._rows
    headers = simplex_table.field_names
    iteration = 1
    max_iterations = 100
    basis_history = set()
    while iteration <= max_iterations:
        print(f"\n### Итерация {iteration} ###")
        print(simplex_table)
        # Конвертируем строки в Fraction для точности
        frac_rows = []
        for row in table_rows:
            new_row = [row[0]] + [Fraction(str(val)).limit_denominator() for val in row[1:]]
            frac_rows.append(new_row)
        # Извлекаем текущий базис
        current_basis = tuple(row[0] for row in frac_rows if row[0] != 'F')
        basis_str = str(current_basis)
        if basis_str in basis_history:
            print("Обнаружено зацикливание: текущий базис уже встречался.")
            print(f"Базис: {current_basis}")
            # Проверяем значение F и искусственные переменные
            f_row = [row for row in frac_rows if row[0] == 'F'][0]
            f_value = float(f_row[1])
            artificial_in_basis = any(row[0] in [str(z) for z in z_vars] and float(row[1]) > 0 for row in frac_rows if row[0] != 'F')
            if f_value == 0 and not artificial_in_basis:
                print("Найдено допустимое оптимальное базисное решение (F = 0, искусственные переменные выведены).")
                solution = {}
                for row in frac_rows:
                    if row[0] != 'F':
                        basis_var = row[0]
                        free_term = float(row[1])
                        solution[basis_var] = free_term
                for var in variables_order:
                    if str(var) not in solution:
                        solution[str(var)] = 0.0
                print("Решение:")
                for var, value in solution.items():
                    print(f"{var} = {value:.6f}")
                print(f"Значение целевой функции: {f_value:.6f}")
                return solution
            else:
                print(f"Система не имеет допустимого базисного решения (F = {f_value:.6f}, искусственные переменные в базисе: {artificial_in_basis}).")
                return None
        basis_history.add(basis_str)
        # Извлекаем строку целевой функции
        f_row = [row for row in frac_rows if row[0] == 'F'][0]
        f_coeffs = {headers[i]: coef for i, coef in enumerate(f_row[2:], 2)}
        # Проверяем оптимальность
        is_optimal = all(coef <= 0 for coef in f_row[2:])
        if is_optimal:
            f_value = float(f_row[1])
            artificial_in_basis = any(row[0] in [str(z) for z in z_vars] and float(row[1]) > 0 for row in frac_rows if row[0] != 'F')
            if f_value == 0 and not artificial_in_basis:
                print("Найдено допустимое оптимальное базисное решение (F = 0, искусственные переменные выведены).")
                solution = {}
                for row in frac_rows:
                    if row[0] != 'F':
                        basis_var = row[0]
                        free_term = float(row[1])
                        solution[basis_var] = free_term
                for var in variables_order:
                    if str(var) not in solution:
                        solution[str(var)] = 0.0
                print("Решение:")
                for var, value in solution.items():
                    print(f"{var} = {value:.6f}")
                print(f"Значение целевой функции: {f_value:.6f}")
                return solution
            else:
                print(f"Система не имеет допустимого базисного решения (F = {f_value:.6f}, искусственные переменные в базисе: {artificial_in_basis}).")
                return None
        # Выбираем ведущий столбец
        max_coeff = float('-inf')
        pivot_col_idx = None
        pivot_col_var = None
        basis_vars = {row[0] for row in frac_rows if row[0] != 'F'}
        print("Коэффициенты в строке F:")
        for i, var in enumerate(variables_order, 2):
            coef = float(f_coeffs[str(var)])
            print(f"{var}: {coef:.2f}")
            if coef > max_coeff and str(var) not in basis_vars:
                can_use = True
                for var1, var2 in complementary_slackness:
                    if var == var1:
                        for row in frac_rows:
                            if row[0] == str(var2) and row[1] > 0:
                                can_use = False
                                print(f"Переменная {var} не может быть введена: {var2} базисная и положительная ({row[1]})")
                                break
                    elif var == var2:
                        for row in frac_rows:
                            if row[0] == str(var1) and row[1] > 0:
                                can_use = False
                                print(f"Переменная {var} не может быть введена: {var1} базисная и положительная ({row[1]})")
                                break
                if can_use:
                    max_coeff = coef
                    pivot_col_idx = i
                    pivot_col_var = var
        if pivot_col_idx is None:
            print("Нет подходящего ведущего столбца с учетом условий дополняющей нежесткости и базисных переменных.")
            return None
        print(f"Ведущий столбец: {pivot_col_var} (коэффициент = {max_coeff:.2f})")
        # Выбираем ведущую строку
        min_ratio = float('inf')
        pivot_row_idx = None
        pivot_row_var = None
        print("Отношения для выбора ведущей строки:")
        for i, row in enumerate(frac_rows):
            if row[0] == 'F' or row[0] == str(pivot_col_var):
                print(f"Строка {row[0]} пропущена (F или совпадает с ведущим столбцом)")
                continue
            free_term = row[1]
            pivot_col_val = row[pivot_col_idx]
            if pivot_col_val > 0:
                ratio = float(free_term / pivot_col_val)
                print(f"Строка {row[0]}: {float(free_term)} / {float(pivot_col_val)} = {ratio:.6f}")
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_idx = i
                    pivot_row_var = row[0]
        if pivot_row_idx is None:
            print("Задача неограничена (нет ведущей строки).")
            return None
        print(f"Ведущая строка: {pivot_row_var} (отношение = {min_ratio:.6f})")
        # Опорный элемент
        pivot_element = frac_rows[pivot_row_idx][pivot_col_idx]
        print(f"Опорный элемент: {float(pivot_element):.6f}")
        # Создаем новую таблицу
        new_table = PrettyTable(headers)
        new_table.float_format = ".2f"
        new_rows = []
        for i, row in enumerate(frac_rows):
            if i == pivot_row_idx:
                new_row = [row[0]] + [val / pivot_element for val in row[1:]]
            else:
                factor = row[pivot_col_idx]
                pivot_row = [val / pivot_element for val in frac_rows[pivot_row_idx][1:]]
                new_row = [row[0]] + [
                    row[j] - factor * pivot_row[j - 1]
                    for j in range(1, len(row))
                ]
            new_rows.append(new_row)
        # Обновляем базисную переменную
        new_rows[pivot_row_idx][0] = str(pivot_col_var)
        # Конвертируем Fraction в float для отображения
        table_rows = []
        for row in new_rows:
            display_row = [row[0]] + [float(val) for val in row[1:]]
            new_table.add_row(display_row)
            table_rows.append(display_row)
        frac_rows = new_rows
        simplex_table = new_table
        iteration += 1
    # Проверка по достижении максимального количества итераций
    print(f"Достигнуто максимальное количество итераций ({max_iterations}).")
    f_row = [row for row in frac_rows if row[0] == 'F'][0]
    f_value = float(f_row[1])
    artificial_in_basis = any(row[0] in [str(z) for z in z_vars] and float(row[1]) > 0 for row in frac_rows if row[0] != 'F')
    if f_value == 0 and not artificial_in_basis:
        print("Найдено допустимое оптимальное базисное решение (F = 0, искусственные переменные выведены).")
        solution = {}
        for row in frac_rows:
            if row[0] != 'F':
                basis_var = row[0]
                free_term = float(row[1])
                solution[basis_var] = free_term
        for var in variables_order:
            if str(var) not in solution:
                solution[str(var)] = 0.0
        print("Решение:")
        for var, value in solution.items():
            print(f"{var} = {value:.6f}")
        print(f"Значение целевой функции: {f_value:.6f}")
        return solution
    else:
        print(f"Система не имеет допустимого базисного решения (F = {f_value:.6f}, искусственные переменные в базисе: {artificial_in_basis}).")
        return None

def solve_kkt_example():
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
        print("Найдено допустимое оптимальное базисное решение для фазы I:")
        for var, value in solution.items():
            print(f"{var} = {value:.6f}")
    else:
        print("Система не имеет допустимого базисного решения.")
    return solution

if __name__ == "__main__":
    solve_kkt_example()

    from scipy.optimize import minimize


    print()
    print()
    print()
    print("Проверка с помощью библиотеки")

    # Определение целевой функции
    def objective(vars):
        x, y = vars
        return 2 * x ** 2 + 3 * y ** 2 + 4 * x * y - 6 * x - 3 * y


    # Ограничения в форме g(x) <= 0 преобразуем в стандартный вид для minimize
    constraints = [
        {'type': 'ineq', 'fun': lambda vars: 1 - (vars[0] + vars[1])},  # x + y <= 1
        {'type': 'ineq', 'fun': lambda vars: 4 - (2 * vars[0] + 3 * vars[1])}  # 2x + 3y <= 4
    ]

    # Границы переменных x >= 0, y >= 0
    bounds = [(0, None), (0, None)]

    # Начальное приближение
    initial_guess = [0, 0]

    # Решение задачи оптимизации
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Вывод результатов
    print("Оптимальные значения:")
    print(f"x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"Значение целевой функции: {result.fun:.4f}")
    print("Проверка ограничений:")
    print(f"x + y = {result.x[0] + result.x[1]:.4f} (должно быть <= 1)")
    print(f"2x + 3y = {2 * result.x[0] + 3 * result.x[1]:.4f} (должно быть <= 4)")