import sympy as sp
from prettytable import PrettyTable
from fractions import Fraction
import copy
from scipy.optimize import minimize
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np


class MySimplexMethod(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, params_dict, log_emitter):
        """
        Initialize the Simplex Method solver with given parameters.

        Args:
            params_dict (dict): Dictionary containing optimization parameters:
                - 'function': Objective function (sympy Expr)
                - 'constraints': List of constraints (sympy Expr)
                - 'variables': List of variables (sympy Symbols)
                - 'max_iterations': Maximum number of iterations (optional)
            log_emitter: Object for emitting log messages
        """
        super().__init__()
        # Initialize parameters from dictionary
        self.function = params_dict['function']
        self.constraints = params_dict['constraints']
        self.variables = params_dict['variables']
        self.max_iterations = params_dict.get('max_iterations', 100)
        self.points = []  # Stores [x, y] coordinates during optimization

        # Validate input parameters
        if not isinstance(self.function, sp.Expr):
            raise ValueError("Function must be a sympy expression")
        if not all(isinstance(c, sp.Expr) for c in self.constraints):
            raise ValueError("All constraints must be sympy expressions")
        if not all(isinstance(v, sp.Symbol) for v in self.variables):
            raise ValueError("Variables must be sympy symbols")

        # Setup logging and execution control
        self.log_emitter = log_emitter
        self._is_running = False
        self.initial_delay = 0.05  # Initial delay between iterations (for visualization)
        self.min_delay = 0.001  # Minimum delay between iterations

        # Optimization results storage
        self.kkt_system = None  # KKT system components
        self.artificial_system = None  # Artificial variables system
        self.simplex_data = None  # Simplex table data
        self.solution_results = None  # Final solution results
        self.scipy_result = None  # Scipy verification results

        # Current state tracking
        self.current_iteration = 0  # Current iteration count
        self.current_solution = None  # Current solution during iterations

    def run(self):
        """Main execution method for the optimization process."""
        if self._is_running:
            return
        self._is_running = True
        self.log_emitter.log_signal.emit("🔹 Simplex Method optimization started...")

        try:
            # Log the optimization constraints
            if self.constraints:
                constraint_msg = ["### Optimization Constraints ###"]
                for i, constr in enumerate(self.constraints, 1):
                    constraint_msg.append(f"g{i}: {sp.pretty(constr)} <= 0")
                self.log_emitter.log_signal.emit("\n".join(constraint_msg))
            else:
                self.log_emitter.log_signal.emit(
                    "### Optimization Constraints ###\nNo explicit constraints provided (assuming x >= 0, y >= 0)")

            # Step 1: Build the KKT system
            self._build_kkt_system()

            # Step 2: Add artificial variables
            self._add_artificial_variables()

            # Step 3: Build the simplex table
            self._build_simplex_table()

            # Step 4: Solve using simplex method
            self._solve_simplex()

            # Step 5: Verify solution with scipy
            self._verify_with_scipy()

            # Completion message
            self.log_emitter.log_signal.emit("🎉 Simplex Method optimization finished successfully!")

        except Exception as e:
            self.log_emitter.log_signal.emit(f"❌ Error in Simplex Method optimization: {str(e)}")
        finally:
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        """Stop the optimization process."""
        self._is_running = False
        self.log_emitter.log_signal.emit("⏹ Simplex Method optimization stopped by user")

    def _build_kkt_system(self):
        """Construct the Karush-Kuhn-Tucker (KKT) system for the optimization problem."""
        self.log_emitter.log_signal.emit("🔧 Building KKT system...")

        n_constraints = len(self.constraints)
        # Create Lagrange multipliers (λ), non-negativity variables (v), and slackness variables (w)
        lambdas = [sp.symbols(f'lambda{i + 1}', real=True) for i in range(n_constraints)]
        vs = [sp.symbols(f'v{i + 1}', real=True) for i in range(len(self.variables))]
        ws = [sp.symbols(f'w{i + 1}', real=True) for i in range(n_constraints)]

        # Construct the Lagrangian function
        L = self.function + sum(lambdas[i] * self.constraints[i] for i in range(n_constraints))

        # Derivatives of Lagrangian
        dL_dx = [sp.diff(L, var) for var in self.variables]
        dL_dlambda = self.constraints

        # KKT conditions
        kkt_conditions = []
        for i in range(len(self.variables)):
            kkt_conditions.append(dL_dx[i] - vs[i])
        for i in range(n_constraints):
            kkt_conditions.append(dL_dlambda[i] + ws[i])

        # Feasibility conditions
        primal_feasibility = [constr <= 0 for constr in self.constraints] + [var >= 0 for var in self.variables]
        dual_feasibility = [lam >= 0 for lam in lambdas] + [v >= 0 for v in vs] + [w >= 0 for w in ws]

        # Complementary slackness conditions
        complementary_slackness = [(self.variables[i], vs[i]) for i in range(len(self.variables))]
        complementary_slackness += [(lambdas[i], ws[i]) for i in range(n_constraints)]

        # Collect all variables
        all_variables = self.variables + lambdas + vs + ws

        # Store the complete KKT system
        self.kkt_system = {
            'lagrangian': L,
            'kkt_conditions': kkt_conditions,
            'primal_feasibility': primal_feasibility,
            'dual_feasibility': dual_feasibility,
            'complementary_slackness': complementary_slackness,
            'all_variables': all_variables,
            'lambdas': lambdas,
            'vs': vs,
            'ws': ws,
            'variables': self.variables,
            'dL_dx': dL_dx,
            'dL_dlambda': dL_dlambda,
            'obj_func': self.function,
            'constraints': self.constraints
        }

        self._log_kkt_system()

    def _log_kkt_system(self):
        """Log the KKT system information in a clean, structured format."""
        sections = [
            ("KKT SYSTEM", ""),
            ("Objective function:", str(sp.pretty(self.function))),
            ("Constraints:", *[f"  g{i + 1}: {sp.pretty(c)} <= 0" for i, c in enumerate(self.constraints)]),
            ("", ""),
            ("Variables:", ""),
            (f"  Optimization: {', '.join(map(str, self.variables))}", ""),
            (f"  Lagrange multipliers (λ): {', '.join(map(str, self.kkt_system['lambdas']))}", ""),
            (f"  Non-negativity (v): {', '.join(map(str, self.kkt_system['vs']))}", ""),
            (f"  Slackness (w): {', '.join(map(str, self.kkt_system['ws']))}", ""),
            ("", ""),
            ("Lagrangian:", str(sp.pretty(self.kkt_system['lagrangian']))),
            ("", ""),
            ("KKT Conditions:",
             *[f"  {i + 1}. {sp.pretty(eq)} = 0" for i, eq in enumerate(self.kkt_system['kkt_conditions'])]),
            ("", ""),
            ("Feasibility:", ""),
            ("  Primal:", *[f"    {sp.pretty(c)}" for c in self.kkt_system['primal_feasibility']]),
            ("  Dual:", *[f"    {sp.pretty(c)}" for c in self.kkt_system['dual_feasibility']]),
            ("", ""),
            ("Complementary Slackness:",
             *[f"  {sp.pretty(var)} * {sp.pretty(v)} = 0" for var, v in self.kkt_system['complementary_slackness']])
        ]

        # Flatten the sections and filter out empty strings
        message = []
        for section in sections:
            if isinstance(section, tuple):
                message.extend(line for line in section if line)
            elif section:
                message.append(section)

        self.log_emitter.log_signal.emit("\n".join(message))

    def _add_artificial_variables(self):
        """Add artificial variables to handle negative free terms in the KKT system."""
        self.log_emitter.log_signal.emit("🔧 Adding artificial variables...")

        variables = self.kkt_system['all_variables']
        vs = self.kkt_system['vs']
        ws = self.kkt_system['ws']
        z_vars = []  # Artificial variables
        modified_equations = []
        z_expressions = {}  # Expressions for artificial variables
        artificial_info = []  # Information about each equation modification

        for i, eq in enumerate(self.kkt_system['kkt_conditions']):
            left_part = eq
            right_part = 0
            free_term = left_part.as_coeff_add(*variables)[0]

            # Determine which auxiliary variable we're working with (v or w)
            if i < len(vs):
                dop_var = vs[i]
                dop_var_name = f'v{i + 1}'
                dop_var_sign = -1  # v variables have negative sign in equations
            else:
                dop_var = ws[i - len(vs)]
                dop_var_name = f'w{i - len(vs) + 1}'
                dop_var_sign = 1  # w variables have positive sign in equations

            # Check if we need an artificial variable for this equation
            need_artificial = (dop_var_sign < 0 and free_term < 0) or (dop_var_sign > 0 and free_term > 0)

            # Store equation information
            eq_info = {
                'equation_index': i + 1,
                'equation': eq,
                'free_term': free_term,
                'dop_var_name': dop_var_name,
                'dop_var_sign': dop_var_sign,
                'need_artificial': need_artificial
            }

            # Move free term to right side
            right_part_final = -free_term
            modified_eq = left_part - free_term

            if need_artificial:
                # Add artificial variable z
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

        # Store the artificial system information
        self.artificial_system = {
            'z_vars': z_vars,
            'z_expressions': z_expressions,
            'F_z': sum(z_vars) if z_vars else None,  # Auxiliary objective function
            'F_z_expanded': sum(z_expressions[z] for z in z_vars) if z_vars else None,
            'modified_equations': modified_equations,
            'artificial_info': artificial_info
        }

        self._log_artificial_variables()

    def _log_artificial_variables(self):
        """Log information about artificial variables and modified equations."""
        msg = ["### Artificial Variables ###"]

        for eq_info in self.artificial_system['artificial_info']:
            i = eq_info['equation_index']
            info = [
                f"Equation {i}: {eq_info['equation']} = 0",
                f"  Free term: {eq_info['free_term']}",
                f"  Aux variable: {eq_info['dop_var_name']} (sign: {'+' if eq_info['dop_var_sign'] > 0 else '-'})",
                f"  Need artificial: {'Yes' if eq_info['need_artificial'] else 'No'}"
            ]

            if eq_info['need_artificial']:
                info.append(f"  Added z{i}: {eq_info['modified_eq']} = {eq_info['right_part']}")
            else:
                info.append(f"  Modified equation: {eq_info['modified_eq']} = {eq_info['right_part']}")

            msg.extend(info)

        if self.artificial_system['z_vars']:
            msg.extend([
                "",
                "### Auxiliary LP Function ###",
                f"F(z) = {self.artificial_system['F_z']}",
                f"F(z) expanded: {self.artificial_system['F_z_expanded']}",
                "",
                "### Final System ###",
                *[f"Equation {i + 1}: {eq} = {rhs}" for i, (eq, rhs) in
                  enumerate(self.artificial_system['modified_equations'])],
                "",
                "### Non-negativity Conditions ###",
                *[f"{z} >= 0" for z in self.artificial_system['z_vars']]
            ])
        else:
            msg.append("\nNo artificial variables needed.")

        self.log_emitter.log_signal.emit("\n".join(msg))

    def _build_simplex_table(self):
        """Построение начальной симплекс-таблицы из измененной системы ККТ."""
        # Определение порядка переменных в таблице
        variables_order = (
                self.kkt_system['variables'] +
                self.kkt_system['lambdas'] +
                self.kkt_system['vs'] +
                self.kkt_system['ws'] +
                self.artificial_system['z_vars']
        )

        # Создание заголовков таблицы
        headers = ['Basic var', 'Free term'] + [str(var) for var in variables_order]
        table = PrettyTable(headers)
        table.float_format = ".2f"

        # Обработка каждого измененного уравнения для создания строк таблицы
        rows_data = []
        for eq, rhs in self.artificial_system['modified_equations']:
            # Получение коэффициентов для каждой переменной
            coeffs = [eq.coeff(var) for var in variables_order]
            basis_var = None

            # Определение базисной переменной (z или w)
            for z_var in self.artificial_system['z_vars']:
                if eq.coeff(z_var) == 1:
                    basis_var = z_var
                    break

            if not basis_var:
                for w_var in self.kkt_system['ws']:
                    if eq.coeff(w_var) == 1:
                        basis_var = w_var
                        break

            # Добавление строки в таблицу
            row = [str(basis_var), float(rhs)] + [float(coef) for coef in coeffs]
            table.add_row(row)
            rows_data.append({
                'basis_var': basis_var,
                'rhs': rhs,
                'coeffs': coeffs
            })

        # Добавление строки целевой функции, если есть искусственные переменные
        f_row_data = None
        if self.artificial_system['F_z_expanded'] is not None:
            f_coeffs = [-self.artificial_system['F_z_expanded'].coeff(var) for var in variables_order]
            f_constant = self.artificial_system['F_z_expanded'].as_coeff_add(*variables_order)[0]
            row = ['F', float(f_constant)] + [float(coef) for coef in f_coeffs]
            table.add_row(row)
            f_row_data = {
                'constant': f_constant,
                'coeffs': f_coeffs
            }

        # Сохранение всех данных симплекс-таблицы
        self.simplex_data = {
            'table': table,
            'rows_data': rows_data,
            'f_row_data': f_row_data,
            'variables_order': variables_order,
            'headers': headers
        }

    def _solve_simplex(self):
        """Решение задачи оптимизации с использованием симплекс-метода."""
        # Получение необходимых переменных из сохраненных систем
        variables_order = self.simplex_data['variables_order']
        z_vars = self.artificial_system['z_vars']
        vs = self.kkt_system['vs']
        ws = self.kkt_system['ws']
        complementary_slackness = self.kkt_system['complementary_slackness']

        # Инициализация рабочих переменных
        simplex_table = self.simplex_data['table']
        table_rows = simplex_table._rows
        headers = self.simplex_data['headers']

        # Сброс счетчиков и хранилища решений
        self.current_iteration = 0
        basis_history = set()  # Отслеживание посещенных базисов для обнаружения циклов
        self.solution_results = {'iterations': []}
        self.points = []  # Очистка списка точек для новой оптимизации

        # Основной цикл итераций симплекс-метода
        while self.current_iteration < self.max_iterations and self._is_running:
            self.current_iteration += 1
            iteration_info = {
                'iteration': self.current_iteration,
                'table': copy.deepcopy(simplex_table),
                'current_solution': {},
                'objective_value': None
            }

            # Преобразование всех значений таблицы в Fraction для точных вычислений
            frac_rows = []
            for row in table_rows:
                new_row = [row[0]] + [Fraction(str(val)).limit_denominator() for val in row[1:]]
                frac_rows.append(new_row)

            # Извлечение текущих базисных переменных
            current_basis = tuple(row[0] for row in frac_rows if row[0] != 'F')
            basis_str = str(current_basis)
            iteration_info['current_basis'] = current_basis

            # Проверка на зацикливание (повторное посещение того же базиса)
            if basis_str in basis_history:
                iteration_info['cycle_detected'] = True
                self.solution_results['iterations'].append(iteration_info)
                final_solution = self._handle_final_solution(frac_rows, variables_order, z_vars, vs, ws,
                                                             complementary_slackness)
                self.solution_results['solution'] = final_solution
                # Отправка финальных точек для визуализации
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            basis_history.add(basis_str)

            # Вычисление текущего решения из базиса
            solution = {}
            for row in frac_rows:
                if row[0] != 'F':
                    basis_var = row[0]
                    free_term = float(row[1])
                    solution[basis_var] = free_term

            # Установка небазисных переменных в ноль
            for var in variables_order:
                if str(var) not in solution:
                    solution[str(var)] = 0.0

            self.current_solution = solution
            iteration_info['current_solution'] = solution

            # Вычисление значения целевой функции и сохранение точек для визуализации
            obj_value = None
            if 'x' in solution and 'y' in solution:
                try:
                    x, y = solution['x'], solution['y']
                    self.points.append([x, y])  # Сохранение текущей точки

                    # Вычисление значения целевой функции в текущей точке
                    obj_value = float(sp.lambdify(self.variables, self.function)(x, y))
                    iteration_info['objective_value'] = obj_value

                    # Отправка точек для визуализации
                    points_array = np.array(self.points, dtype=float)  # Формат: (n, 2)
                    self.update_signal.emit(points_array)

                except Exception as e:
                    pass

            # Проверка условий оптимальности (все коэффициенты в строке F <= 0)
            f_row = [row for row in frac_rows if row[0] == 'F'][0]
            f_coeffs = {headers[i]: coef for i, coef in enumerate(f_row[2:], 2)}
            iteration_info['f_coeffs'] = {str(var): float(f_coeffs[str(var)]) for var in variables_order}

            is_optimal = all(coef <= 0 for coef in f_row[2:])
            iteration_info['is_optimal'] = is_optimal

            if is_optimal:
                self.solution_results['iterations'].append(iteration_info)
                final_solution = self._handle_final_solution(frac_rows, variables_order, z_vars, vs, ws,
                                                             complementary_slackness)
                self.solution_results['solution'] = final_solution
                # Отправка финальных точек для визуализации
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            # Выбор ведущего столбца (переменная, входящая в базис)
            max_coeff = float('-inf')
            pivot_col_idx = None
            pivot_col_var = None
            basis_vars = {row[0] for row in frac_rows if row[0] != 'F'}

            # Оценка всех кандидатов на ведущий столбец
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

                # Проверка, может ли эта переменная войти в базис
                if coef > max_coeff and not is_basis:
                    can_use = True

                    # Проверка условий взаимодополняющей слабины
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
                self.solution_results['iterations'].append(iteration_info)
                # Отправка финальных точек для визуализации
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            # Выбор ведущей строки (переменная, покидающая базис)
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
                self.solution_results['iterations'].append(iteration_info)
                # Отправка финальных точек для визуализации
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            # Получение значения ведущего элемента
            pivot_element = frac_rows[pivot_row_idx][pivot_col_idx]
            iteration_info['pivot_element'] = float(pivot_element)

            # Создание новой симплекс-таблицы после операции поворота
            new_table = PrettyTable(headers)
            new_table.float_format = ".2f"
            new_rows = []

            # Выполнение исключения Гаусса для операции поворота
            for i, row in enumerate(frac_rows):
                if i == pivot_row_idx:
                    # Нормализация ведущей строки
                    new_row = [row[0]] + [val / pivot_element for val in row[1:]]
                else:
                    # Исключение ведущего столбца из других строк
                    factor = row[pivot_col_idx]
                    pivot_row = [val / pivot_element for val in frac_rows[pivot_row_idx][1:]]
                    new_row = [row[0]] + [
                        row[j] - factor * pivot_row[j - 1]
                        for j in range(1, len(row))
                    ]
                new_rows.append(new_row)

            # Обновление базисной переменной в ведущей строке
            new_rows[pivot_row_idx][0] = str(pivot_col_var)

            # Преобразование Fraction в float для отображения
            table_rows = []
            for row in new_rows:
                display_row = [row[0]] + [float(val) for val in row[1:]]
                new_table.add_row(display_row)
                table_rows.append(display_row)

            # Обновление рабочих переменных для следующей итерации
            frac_rows = new_rows
            simplex_table = new_table

            # Сохранение информации об итерации и логирование деталей
            self.solution_results['iterations'].append(iteration_info)
            self._log_simplex_iteration(iteration_info)

        # Проверка достижения максимального количества итераций без сходимости
        if self.current_iteration >= self.max_iterations:
            final_solution = self._handle_final_solution(frac_rows, variables_order, z_vars, vs, ws,
                                                         complementary_slackness)
            self.solution_results['solution'] = final_solution
            # Отправка финальных точек для визуализации
            self.update_signal.emit(np.array(self.points, dtype=float))

    def _log_simplex_iteration(self, iteration_info):
        """Log detailed information about a simplex iteration."""
        msg = [
            f"\n### Iteration {iteration_info['iteration']} ###",
            str(iteration_info['table']),
            "",
            "### F-row coefficients ###"
        ]

        # Log all coefficients in the objective row
        for var, coef in iteration_info['f_coeffs'].items():
            msg.append(f"{var}: {coef:.6f}")

        # Log pivot selection information
        msg.append("\n### Pivot Selection ###")
        msg.append(
            f"Leading column: {iteration_info['pivot_col']['var']} (coef: {iteration_info['pivot_col']['coef']:.6f})")

        # Log ratio test information
        msg.append("\n### Ratios ###")
        for ratio in iteration_info['ratio_data']:
            if 'skipped' in ratio:
                msg.append(f"Row {ratio['row_var']}: skipped ({ratio['reason']})")
            else:
                msg.append(
                    f"Row {ratio['row_var']}: {ratio['free_term']:.6f} / {ratio['pivot_col_val']:.6f} = {ratio['ratio']:.6f}")

        # Log selected pivot information
        msg.append(
            f"\nLeading row: {iteration_info['pivot_row']['var']} (ratio: {iteration_info['pivot_row']['ratio']:.6f})")
        msg.append(f"Pivot element: {iteration_info['pivot_element']:.6f}")

        # Log objective value if available
        if 'objective_value' in iteration_info:
            msg.append(f"\nCurrent objective value: {iteration_info['objective_value']:.6f}")

        self.log_emitter.log_signal.emit("\n".join(msg))

    def _handle_final_solution(self, frac_rows, variables_order, z_vars, vs, ws, complementary_slackness):
        """Analyze and interpret the final simplex solution."""
        f_row = [row for row in frac_rows if row[0] == 'F'][0]
        f_value = float(f_row[1])

        # Check if any artificial variables remain in the basis with positive values
        artificial_in_basis = any(
            row[0] in [str(z) for z in z_vars] and float(row[1]) > 0 for row in frac_rows if row[0] != 'F')

        # Check additional variables (v and w) in basis with positive values
        additional_vars = [str(v) for v in vs] + [str(w) for w in ws]
        additional_in_basis = [
            (row[0], float(row[1]))
            for row in frac_rows
            if row[0] != 'F' and row[0] in additional_vars and float(row[1]) > 0
        ]

        # Extract solution values
        solution = {}
        for row in frac_rows:
            if row[0] != 'F':
                basis_var = row[0]
                free_term = float(row[1])
                solution[basis_var] = free_term

        for var in variables_order:
            if str(var) not in solution:
                solution[str(var)] = 0.0

        # Analyze implications of additional variables in basis
        additional_analysis = []
        for var, value in additional_in_basis:
            analysis = {'var': var, 'value': value, 'implications': []}

            # Check complementary slackness conditions
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

        # Prepare final solution result
        result = {
            'f_value': f_value,
            'artificial_in_basis': artificial_in_basis,
            'additional_in_basis': additional_in_basis,
            'solution': solution,
            'additional_analysis': additional_analysis,
            'is_feasible': f_value == 0 and not artificial_in_basis
        }

        # Interpret active constraints if solution is feasible
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

        self._log_final_solution(result)
        return result

    def _log_final_solution(self, final_solution):
        """Log the final solution information."""
        msg = ["\n### Final Solution ###"]

        if final_solution['is_feasible']:
            msg.append("✅ Found feasible optimal solution")
            msg.append("Solution:")

            # Log all variable values
            for var, value in final_solution['solution'].items():
                msg.append(f"{var} = {value:.6f}")

            msg.append(f"Objective value: {final_solution['f_value']:.6f}")

            # Log active constraints information
            if final_solution['additional_in_basis']:
                msg.append("\nActive constraints:")
                for constraint in final_solution['active_constraints']:
                    if constraint['type'] == 'non_negativity':
                        msg.append(
                            f"  {constraint['var']} > 0 => {constraint['original_var']} = 0 (non-negativity constraint active)")
                    else:
                        msg.append(
                            f"  {constraint['var']} > 0 => {constraint['original_var']} = 0 (constraint inactive)")
        else:
            msg.append("❌ No feasible solution found")
            msg.append(f"F value: {final_solution['f_value']:.6f}")
            msg.append(f"Artificial variables in basis: {final_solution['artificial_in_basis']}")

        self.log_emitter.log_signal.emit("\n".join(msg))

    def _verify_with_scipy(self):
        self.log_emitter.log_signal.emit("\n🔍 Verifying solution with scipy.optimize.minimize...")

        def objective(vars):
            return sp.lambdify(self.variables, self.function, 'numpy')(vars[0], vars[1])

        scipy_constraints = []
        for constr in self.constraints:
            def constraint_func(vars, c=constr):
                return -sp.lambdify(self.variables, c, 'numpy')(vars[0], vars[1])  # -g_i(x, y) >= 0

            scipy_constraints.append({
                'type': 'ineq',
                'fun': constraint_func
            })

        bounds = [(0, None), (0, None)]

        initial_guess = [0, 0]

        self.scipy_result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints
        )

        msg = [
            "### Scipy Optimization Results ###",
            f"Solution: x = {self.scipy_result.x[0]:.6f}, y = {self.scipy_result.x[1]:.6f}",
            f"Objective value: {self.scipy_result.fun:.6f}",
            "Constraint values:",
            *[
                f"g{i + 1}(x,y) = {sp.lambdify(self.variables, c, 'numpy')(self.scipy_result.x[0], self.scipy_result.x[1]):.6f}"
                for i, c in enumerate(self.constraints)],
            f"Success: {self.scipy_result.success}",
            f"Message: {self.scipy_result.message}"
        ]

        if self.solution_results.get('solution', {}).get('is_feasible', False):
            simplex_sol = self.solution_results['solution']['solution']
            x_simplex = simplex_sol.get('x', 0)
            y_simplex = simplex_sol.get('y', 0)

            msg.extend([
                "\n### Comparison ###",
                f"Simplex solution: x = {x_simplex:.6f}, y = {y_simplex:.6f}",
                f"Scipy solution: x = {self.scipy_result.x[0]:.6f}, y = {self.scipy_result.x[1]:.6f}",
                f"Difference: Δx = {abs(x_simplex - self.scipy_result.x[0]):.6f}, Δy = {abs(y_simplex - self.scipy_result.x[1]):.6f}",
                f"Objective difference: {abs(sp.lambdify(self.variables, self.function)(x_simplex, y_simplex) - self.scipy_result.fun):.6f}"
            ])

        self.log_emitter.log_signal.emit("\n".join(msg))

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        if not isinstance(value, sp.Expr):
            raise ValueError("Function must be a sympy expression")
        self._function = value

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        if not isinstance(value, list) or not all(isinstance(c, sp.Expr) for c in value):
            raise ValueError("Constraints must be a list of sympy expressions")
        self._constraints = value

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        if not isinstance(value, list) or not all(isinstance(v, sp.Symbol) for v in value):
            raise ValueError("Variables must be a list of sympy symbols")
        self._variables = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Max iterations must be a positive integer")
        self._max_iterations = value

    @property
    def is_running(self):
        return self._is_running

    @property
    def current_solution(self):
        return self._current_solution

    @current_solution.setter
    def current_solution(self, value):
        self._current_solution = value
