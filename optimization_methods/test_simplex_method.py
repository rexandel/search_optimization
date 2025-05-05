import sympy as sp
from prettytable import PrettyTable
from fractions import Fraction
import copy
from scipy.optimize import minimize
import numpy as np


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
        'variables': variables,
        'dL_dx': dL_dx,
        'dL_dlambda': dL_dlambda,
        'obj_func': obj_func,
        'constraints': constraints
    }


def add_artificial_variables(kkt_system, kkt_conditions):
    variables = kkt_system['all_variables']
    vs = kkt_system['vs']
    ws = kkt_system['ws']
    z_vars = []
    modified_equations = []
    z_expressions = {}
    artificial_info = []

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

        eq_info = {
            'equation_index': i + 1,
            'equation': eq,
            'free_term': free_term,
            'dop_var_name': dop_var_name,
            'dop_var_sign': dop_var_sign,
            'need_artificial': need_artificial
        }

        right_part_final = -free_term
        modified_eq = left_part - free_term

        if need_artificial:
            z_var = sp.symbols(f'z{i + 1}', real=True)
            z_vars.append(z_var)
            modified_eq = modified_eq + z_var
            z_expressions[z_var] = -left_part
            eq_info['z_var'] = z_var
            eq_info['modified_eq'] = modified_eq
            eq_info['right_part'] = right_part_final
        else:
            eq_info['modified_eq'] = modified_eq
            eq_info['right_part'] = right_part_final

        artificial_info.append(eq_info)
        modified_equations.append((modified_eq, right_part_final))

    result = {
        'z_vars': z_vars,
        'z_expressions': z_expressions,
        'F_z': sum(z_vars) if z_vars else None,
        'F_z_expanded': sum(z_expressions[z] for z in z_vars) if z_vars else None,
        'modified_equations': modified_equations,
        'artificial_info': artificial_info
    }

    return result


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

    rows_data = []
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
        rows_data.append({
            'basis_var': basis_var,
            'rhs': rhs,
            'coeffs': coeffs
        })

    f_row_data = None
    if artificial_system['F_z_expanded'] is not None:
        f_coeffs = [-artificial_system['F_z_expanded'].coeff(var) for var in variables_order]
        f_constant = artificial_system['F_z_expanded'].as_coeff_add(*variables_order)[0]
        row = ['F', float(f_constant)] + [float(coef) for coef in f_coeffs]
        table.add_row(row)
        f_row_data = {
            'constant': f_constant,
            'coeffs': f_coeffs
        }

    return {
        'table': table,
        'rows_data': rows_data,
        'f_row_data': f_row_data,
        'variables_order': variables_order,
        'headers': headers
    }


def solve_simplex_table(kkt_system, artificial_system, simplex_data):
    variables_order = simplex_data['variables_order']
    z_vars = artificial_system['z_vars']
    vs = kkt_system['vs']
    ws = kkt_system['ws']
    complementary_slackness = kkt_system['complementary_slackness']

    simplex_table = simplex_data['table']
    table_rows = simplex_table._rows
    headers = simplex_table.field_names

    iteration = 1
    max_iterations = 100
    basis_history = set()

    iterations_data = []

    while iteration <= max_iterations:
        iteration_info = {
            'iteration': iteration,
            'table': copy.deepcopy(simplex_table),
        }

        # Конвертируем строки в Fraction для точности
        frac_rows = []
        for row in table_rows:
            new_row = [row[0]] + [Fraction(str(val)).limit_denominator() for val in row[1:]]
            frac_rows.append(new_row)

        # Извлекаем текущий базис
        current_basis = tuple(row[0] for row in frac_rows if row[0] != 'F')
        basis_str = str(current_basis)
        iteration_info['current_basis'] = current_basis

        if basis_str in basis_history:
            iteration_info['cycle_detected'] = True
            iterations_data.append(iteration_info)
            final_solution = handle_final_solution(frac_rows, variables_order, z_vars, vs, ws, complementary_slackness)
            return {'iterations': iterations_data, 'solution': final_solution}

        basis_history.add(basis_str)

        # Извлекаем строку целевой функции
        f_row = [row for row in frac_rows if row[0] == 'F'][0]
        f_coeffs = {headers[i]: coef for i, coef in enumerate(f_row[2:], 2)}
        iteration_info['f_coeffs'] = {str(var): float(f_coeffs[str(var)]) for var in variables_order}

        # Проверяем оптимальность
        is_optimal = all(coef <= 0 for coef in f_row[2:])
        iteration_info['is_optimal'] = is_optimal

        if is_optimal:
            iterations_data.append(iteration_info)
            final_solution = handle_final_solution(frac_rows, variables_order, z_vars, vs, ws, complementary_slackness)
            return {'iterations': iterations_data, 'solution': final_solution}

        # Выбираем ведущий столбец
        max_coeff = float('-inf')
        pivot_col_idx = None
        pivot_col_var = None
        basis_vars = {row[0] for row in frac_rows if row[0] != 'F'}

        pivot_col_candidates = []
        for i, var in enumerate(variables_order, 2):
            coef = float(f_coeffs[str(var)])
            is_basis = str(var) in basis_vars
            candidate = {
                'var': var,
                'coef': coef,
                'is_basis': is_basis,
                'disqualified_reason': None
            }

            if coef > max_coeff and not is_basis:
                can_use = True
                for var1, var2 in complementary_slackness:
                    if var == var1:
                        for row in frac_rows:
                            if row[0] == str(var2) and row[1] > 0:
                                can_use = False
                                candidate['disqualified_reason'] = f"{var2} базисная и положительная ({row[1]})"
                                break
                    elif var == var2:
                        for row in frac_rows:
                            if row[0] == str(var1) and row[1] > 0:
                                can_use = False
                                candidate['disqualified_reason'] = f"{var1} базисная и положительная ({row[1]})"
                                break

                if can_use:
                    max_coeff = coef
                    pivot_col_idx = i
                    pivot_col_var = var

            pivot_col_candidates.append(candidate)

        iteration_info['pivot_col_candidates'] = pivot_col_candidates
        iteration_info['pivot_col'] = {'var': pivot_col_var, 'index': pivot_col_idx, 'coef': max_coeff}

        if pivot_col_idx is None:
            iteration_info['no_pivot_col'] = True
            iterations_data.append(iteration_info)
            return {'iterations': iterations_data, 'solution': None}

        # Выбираем ведущую строку
        min_ratio = float('inf')
        pivot_row_idx = None
        pivot_row_var = None

        ratio_data = []
        for i, row in enumerate(frac_rows):
            if row[0] == 'F' or row[0] == str(pivot_col_var):
                ratio_data.append({
                    'row_var': row[0],
                    'skipped': True,
                    'reason': 'F или совпадает с ведущим столбцом'
                })
                continue

            free_term = row[1]
            pivot_col_val = row[pivot_col_idx]

            if pivot_col_val > 0:
                ratio = float(free_term / pivot_col_val)
                ratio_data.append({
                    'row_var': row[0],
                    'free_term': float(free_term),
                    'pivot_col_val': float(pivot_col_val),
                    'ratio': ratio
                })

                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_idx = i
                    pivot_row_var = row[0]
            else:
                ratio_data.append({
                    'row_var': row[0],
                    'free_term': float(free_term),
                    'pivot_col_val': float(pivot_col_val),
                    'skipped': True,
                    'reason': 'Коэффициент в ведущем столбце <= 0'
                })

        iteration_info['ratio_data'] = ratio_data
        iteration_info['pivot_row'] = {'var': pivot_row_var, 'index': pivot_row_idx, 'ratio': min_ratio}

        if pivot_row_idx is None:
            iteration_info['unbounded'] = True
            iterations_data.append(iteration_info)
            return {'iterations': iterations_data, 'solution': None}

        # Опорный элемент
        pivot_element = frac_rows[pivot_row_idx][pivot_col_idx]
        iteration_info['pivot_element'] = float(pivot_element)

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

        iterations_data.append(iteration_info)
        iteration += 1

    # Достигнуто максимальное количество итераций
    final_solution = handle_final_solution(frac_rows, variables_order, z_vars, vs, ws, complementary_slackness)
    return {'iterations': iterations_data, 'solution': final_solution, 'max_iterations_reached': True}


def handle_final_solution(frac_rows, variables_order, z_vars, vs, ws, complementary_slackness):
    f_row = [row for row in frac_rows if row[0] == 'F'][0]
    f_value = float(f_row[1])

    # Проверяем искусственные переменные
    artificial_in_basis = any(
        row[0] in [str(z) for z in z_vars] and float(row[1]) > 0 for row in frac_rows if row[0] != 'F')

    # Проверяем дополнительные переменные
    additional_vars = [str(v) for v in vs] + [str(w) for w in ws]
    additional_in_basis = [
        (row[0], float(row[1]))
        for row in frac_rows
        if row[0] != 'F' and row[0] in additional_vars and float(row[1]) > 0
    ]

    # Формируем решение
    solution = {}
    for row in frac_rows:
        if row[0] != 'F':
            basis_var = row[0]
            free_term = float(row[1])
            solution[basis_var] = free_term

    for var in variables_order:
        if str(var) not in solution:
            solution[str(var)] = 0.0

    # Анализ дополнительных переменных
    additional_analysis = []
    for var, value in additional_in_basis:
        analysis = {'var': var, 'value': value, 'implications': []}

        for var1, var2 in complementary_slackness:
            if str(var1) == var:
                analysis['implications'].append({
                    'condition': f"{var1} * {var2} = 0",
                    'result': f"{var2} = 0"
                })
            elif str(var2) == var:
                analysis['implications'].append({
                    'condition': f"{var1} * {var2} = 0",
                    'result': f"{var1} = 0"
                })

        additional_analysis.append(analysis)

    result = {
        'f_value': f_value,
        'artificial_in_basis': artificial_in_basis,
        'additional_in_basis': additional_in_basis,
        'solution': solution,
        'additional_analysis': additional_analysis,
        'is_feasible': f_value == 0 and not artificial_in_basis
    }

    # Интерпретация активных ограничений
    if result['is_feasible'] and additional_in_basis:
        active_constraints = []
        for var, value in additional_in_basis:
            if var in [str(v) for v in vs]:
                idx = [str(v) for v in vs].index(var)
                active_constraints.append({
                    'var': var,
                    'type': 'non_negativity',
                    'original_var': str(variables_order[idx]),
                    'value': value
                })
            elif var in [str(w) for w in ws]:
                idx = [str(w) for w in ws].index(var)
                active_constraints.append({
                    'var': var,
                    'type': 'lambda',
                    'original_var': f"lambda{idx + 1}",
                    'value': value
                })
        result['active_constraints'] = active_constraints

    return result


def solve_kkt_example():
    x, y = sp.symbols('x y', real=True)
    variables = [x, y]

    # Целевая функция
    # obj_func = 2 * x ** 2 + 3 * y ** 2 + 4 * x * y - 6 * x - 3 * y
    obj_func = 2 * x**2 + 2 * x * y + 2 * y**2 - 4 * x - 6 * y

    # Ограничения в форме g(x) <= 0
    constraints = [
        # x + y - 1,
        # 2 * x + 3 * y - 4
        x + 2*y - 2
    ]

    kkt_system = dynamic_kkt_system(obj_func, constraints, variables)
    artificial_system = add_artificial_variables(kkt_system, kkt_system['kkt_conditions'])
    simplex_data = build_simplex_table(kkt_system, artificial_system)
    solution_results = solve_simplex_table(kkt_system, artificial_system, simplex_data)

    # Проверка с использованием scipy.optimize.minimize
    def objective(vars):
        return sp.lambdify((x, y), obj_func, 'numpy')(vars[0], vars[1])

    # Преобразование ограничений в формат scipy
    scipy_constraints = []
    for i, constr in enumerate(constraints):
        # Ограничение g_i(x, y) <= 0
        def constraint_func(vars, constr=constr):
            return -sp.lambdify((x, y), constr, 'numpy')(vars[0], vars[1])  # -g_i(x, y) >= 0

        scipy_constraints.append({
            'type': 'ineq',
            'fun': constraint_func
        })

    # Границы: x >= 0, y >= 0
    bounds = [(0, None), (0, None)]

    # Начальное приближение
    initial_guess = [0, 0]

    # Решение задачи оптимизации
    scipy_result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=scipy_constraints
    )

    return {
        'kkt_system': kkt_system,
        'artificial_system': artificial_system,
        'simplex_data': simplex_data,
        'solution_results': solution_results,
        'scipy_result': scipy_result,
        'variables': variables,
        'obj_func': obj_func,
        'constraints': constraints
    }


def main():
    results = solve_kkt_example()

    kkt_system = results['kkt_system']
    artificial_system = results['artificial_system']
    simplex_data = results['simplex_data']
    solution_results = results['solution_results']
    scipy_result = results['scipy_result']
    variables = results['variables']
    obj_func = results['obj_func']
    constraints = results['constraints']

    x, y = variables

    # === Вывод результатов KKT системы ===
    print("### Задача оптимизации ###")
    print(f"Целевая функция: {obj_func}")
    print("Ограничения:")
    for i, g in enumerate(constraints):
        print(f"g{i + 1}(x) = {g} <= 0")

    print("\n### Переменные системы ###")
    print(f"Переменные оптимизации: {kkt_system['variables']}")
    print(f"Множители Лагранжа: {kkt_system['lambdas']}")
    print(f"Переменные неотрицательности v: {kkt_system['vs']}")
    print(f"Переменные нежесткости w: {kkt_system['ws']}")

    print("\n### Функция Лагранжа ###")
    print(sp.pretty(kkt_system['lagrangian']))

    print("\n### Частные производные ###")
    for i, var in enumerate(kkt_system['variables']):
        print(f"dL/d{var} = {kkt_system['dL_dx'][i]}")

    print("\nКоэффициенты в условиях KKT (для анализа знаков):")
    for i, eq in enumerate(kkt_system['kkt_conditions']):
        print(f"Уравнение {i + 1}: {eq} = 0")
        if i < len(kkt_system['variables']):
            print(f"  Уравнение для переменной {kkt_system['variables'][i]}")
        else:
            print(f"  Уравнение для ограничения {i - len(kkt_system['variables']) + 1}")

    for i, lam in enumerate(kkt_system['lambdas']):
        print(f"dL/d{lam} = {kkt_system['dL_dlambda'][i]}")

    print("\n### Условия KKT ###")
    for i, eq in enumerate(kkt_system['kkt_conditions']):
        print(f"Уравнение {i + 1}: {eq} = 0")

    print("\n### Условия допустимости ###")
    print("Примарные условия:")
    for cond in kkt_system['primal_feasibility']:
        print(sp.pretty(cond))

    print("\nДвойственные условия:")
    for cond in kkt_system['dual_feasibility']:
        print(sp.pretty(cond))

    print("\n### Условия дополняющей нежесткости ###")
    for var, v in kkt_system['complementary_slackness']:
        if var in kkt_system['variables']:
            i = kkt_system['variables'].index(var)
            print(f"{var} * v{i + 1} = 0, {var} >= 0, v{i + 1} >= 0")
        else:
            i = kkt_system['lambdas'].index(var)
            print(f"{var} * w{i + 1} = 0, {var} >= 0, w{i + 1} >= 0")

    # === Вывод результатов добавления искусственных переменных ===
    print("\n### Ввод искусственных переменных ###")
    for eq_info in artificial_system['artificial_info']:
        i = eq_info['equation_index']
        eq = eq_info['equation']
        free_term = eq_info['free_term']
        dop_var_name = eq_info['dop_var_name']
        dop_var_sign = eq_info['dop_var_sign']
        need_artificial = eq_info['need_artificial']

        print(f"Анализ уравнения {i}: {eq} = 0")
        print(f"  Свободный член: {free_term}")
        print(f"  Доп. переменная: {dop_var_name} входит со знаком {'+' if dop_var_sign > 0 else '-'}")
        print(f"  Совпадение знаков: {'Да' if need_artificial else 'Нет'}")

        if need_artificial:
            z_var = eq_info['z_var']
            modified_eq = eq_info['modified_eq']
            right_part = eq_info['right_part']
            print(f"В уравнение {i} введена искусственная переменная {z_var}")
            print(f"Уравнение с {z_var}: {modified_eq} = {right_part}")
        else:
            modified_eq = eq_info['modified_eq']
            right_part = eq_info['right_part']
            print(f"В уравнение {i} не требуется вводить искусственную переменную")
            print(f"Уравнение остается: {modified_eq} = {right_part}")

    if artificial_system['z_vars']:
        F_z = artificial_system['F_z']
        F_z_expanded = artificial_system['F_z_expanded']

        print("\n### Вспомогательная функция ЛП ###")
        print(f"F(z) = {F_z}")
        print(f"F(z) = {F_z_expanded}")

        print("\n### Окончательная система уравнений ###")
        for i, (eq, right) in enumerate(artificial_system['modified_equations']):
            print(f"Уравнение {i + 1}: {eq} = {right}")

        print("\n### Условия неотрицательности для искусственных переменных ###")
        for z in artificial_system['z_vars']:
            print(f"{z} >= 0")
    else:
        print("\nИскусственных переменных не требуется вводить в данной задаче.")

    # === Вывод результатов построения симплекс-таблицы ===
    print("\n### Первая симплекс-таблица ###")
    print(simplex_data['table'])

    # === Вывод результатов симплекс-метода ===
    for idx, iter_data in enumerate(solution_results['iterations']):
        iteration = iter_data['iteration']
        print(f"\n### Итерация {iteration} ###")
        print(iter_data['table'])

        # Если обнаружено зацикливание
        if 'cycle_detected' in iter_data and iter_data['cycle_detected']:
            print("Обнаружено зацикливание: текущий базис уже встречался.")
            print(f"Базис: {iter_data['current_basis']}")
            break

        # Если решение оптимально
        if iter_data['is_optimal']:
            print("Найдено оптимальное решение (все коэффициенты в строке F <= 0).")
            break

        # Коэффициенты в строке F
        print("Коэффициенты в строке F:")
        for var, coef in iter_data['f_coeffs'].items():
            print(f"{var}: {coef:.2f}")

        # Информация о выборе ведущего столбца
        if 'no_pivot_col' in iter_data and iter_data['no_pivot_col']:
            print("Нет подходящего ведущего столбца с учетом условий дополняющей нежесткости и базисных переменных.")
            break

        pivot_col = iter_data['pivot_col']
        print(f"Ведущий столбец: {pivot_col['var']} (коэффициент = {pivot_col['coef']:.2f})")

        # Информация о выборе ведущей строки
        print("Отношения для выбора ведущей строки:")
        for ratio_info in iter_data['ratio_data']:
            if 'skipped' in ratio_info:
                print(f"Строка {ratio_info['row_var']} пропущена ({ratio_info['reason']})")
            else:
                print(
                    f"Строка {ratio_info['row_var']}: {ratio_info['free_term']:.6f} / {ratio_info['pivot_col_val']:.6f} = {ratio_info['ratio']:.6f}")

        pivot_row = iter_data['pivot_row']
        print(f"Ведущая строка: {pivot_row['var']} (отношение = {pivot_row['ratio']:.6f})")
        print(f"Опорный элемент: {iter_data['pivot_element']:.6f}")

    # === Вывод финального решения ===
    final_solution = solution_results['solution']
    print("\n### Финальное решение ###")
    if final_solution and final_solution['is_feasible']:
        print("Найдено допустимое оптимальное базисное решение (F = 0, искусственные переменные выведены).")
        print("Решение:")
        for var, value in final_solution['solution'].items():
            print(f"{var} = {value:.6f}")
        print(f"Значение целевой функции: {final_solution['f_value']:.6f}")

        if final_solution['additional_in_basis']:
            print("\n### Активные ограничения ###")
            for constraint in final_solution['active_constraints']:
                if constraint['type'] == 'non_negativity':
                    print(
                        f"{constraint['var']} > 0 => {constraint['original_var']} = 0 (ограничение неотрицательности активно)")
                else:
                    print(f"{constraint['var']} > 0 => {constraint['original_var']} = 0 (ограничение не активно)")
    else:
        print(
            f"Система не имеет допустимого базисного решения (F = {final_solution['f_value']:.6f}, искусственные переменные в базисе: {final_solution['artificial_in_basis']}).")

    # === Проверка решения через scipy ===
    print("\n### Проверка решения с использованием scipy.optimize.minimize ###")
    print("Результаты от scipy.optimize.minimize:")
    print(f"x = {scipy_result.x[0]:.6f}, y = {scipy_result.x[1]:.6f}")
    print(f"Значение целевой функции: {scipy_result.fun:.6f}")
    print("Проверка ограничений:")
    for i, constr in enumerate(constraints):
        value = sp.lambdify((x, y), constr, 'numpy')(scipy_result.x[0], scipy_result.x[1])
        print(f"g{i + 1}(x, y) = {value:.6f} (должно быть <= 0)")
    print(f"Успех оптимизации: {scipy_result.success}")
    print(f"Сообщение: {scipy_result.message}")

    # === Сравнение решений ===
    if final_solution and final_solution['is_feasible']:
        x_simplex = final_solution['solution'].get('x', 0.0)
        y_simplex = final_solution['solution'].get('y', 0.0)
        print("\n### Сравнение решений ###")
        print(f"Симплекс-метод: x = {x_simplex:.6f}, y = {y_simplex:.6f}")
        print(f"Scipy minimize: x = {scipy_result.x[0]:.6f}, y = {scipy_result.x[1]:.6f}")
        print(
            f"Абсолютная разница: |x_simplex - x_scipy| = {abs(x_simplex - scipy_result.x[0]):.6f}, |y_simplex - y_scipy| = {abs(y_simplex - scipy_result.x[1]):.6f}")

        # Вычисление значения целевой функции для симплекс-метода
        f_simplex = sp.lambdify((x, y), obj_func, 'numpy')(x_simplex, y_simplex)
        print(f"Значение целевой функции (симплекс): {f_simplex:.6f}")
        print(f"Разница в значении целевой функции: {abs(f_simplex - scipy_result.fun):.6f}")
    else:
        print("\nСравнение невозможно: симплекс-метод не нашёл допустимого решения.")


if __name__ == "__main__":
    main()