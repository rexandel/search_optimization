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
        super().__init__()
        self.function = params_dict['function']
        self.constraints = params_dict['constraints']
        self.variables = params_dict['variables']
        self.max_iterations = params_dict.get('max_iterations', 100)
        self.points = []

        if not isinstance(self.function, sp.Expr):
            raise ValueError("Function must be a sympy expression")
        if not all(isinstance(c, sp.Expr) for c in self.constraints):
            raise ValueError("All constraints must be sympy expressions")
        if not all(isinstance(v, sp.Symbol) for v in self.variables):
            raise ValueError("Variables must be sympy symbols")

        self.log_emitter = log_emitter
        self._is_running = False
        self.initial_delay = 0.05
        self.min_delay = 0.001

        # Результаты оптимизации
        self.kkt_system = None
        self.artificial_system = None
        self.simplex_data = None
        self.solution_results = None
        self.scipy_result = None

        # Текущее состояние
        self.current_iteration = 0
        self.current_solution = None

    def run(self):
        self._is_running = True
        self.log_emitter.log_signal.emit("🔹 KKT optimization started...")

        try:
            if self.constraints:
                constraint_msg = ["### Optimization Constraints ###"]
                for i, constr in enumerate(self.constraints, 1):
                    constraint_msg.append(f"g{i}: {sp.pretty(constr)} <= 0")
                self.log_emitter.log_signal.emit("\n".join(constraint_msg))
            else:
                self.log_emitter.log_signal.emit("### Optimization Constraints ###\nNo explicit constraints provided (assuming x >= 0, y >= 0)")

            # Шаг 1: Построение системы KKT
            self._build_kkt_system()

            # Шаг 2: Добавление искусственных переменных
            self._add_artificial_variables()

            # Шаг 3: Построение симплекс-таблицы
            self._build_simplex_table()

            # Шаг 4: Решение симплекс-методом
            self._solve_simplex()

            # Шаг 5: Проверка решения через scipy
            self._verify_with_scipy()

            # Завершение
            self.log_emitter.log_signal.emit("🎉 KKT optimization finished successfully!")

        except Exception as e:
            self.log_emitter.log_signal.emit(f"❌ Error in KKT optimization: {str(e)}")
        finally:
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.log_emitter.log_signal.emit("⏹ KKT optimization stopped by user")

    def _build_kkt_system(self):
        self.log_emitter.log_signal.emit("🔧 Building KKT system...")

        n_constraints = len(self.constraints)
        lambdas = [sp.symbols(f'lambda{i + 1}', real=True) for i in range(n_constraints)]
        vs = [sp.symbols(f'v{i + 1}', real=True) for i in range(len(self.variables))]
        ws = [sp.symbols(f'w{i + 1}', real=True) for i in range(n_constraints)]

        L = self.function + sum(lambdas[i] * self.constraints[i] for i in range(n_constraints))
        dL_dx = [sp.diff(L, var) for var in self.variables]
        dL_dlambda = self.constraints

        kkt_conditions = []
        for i in range(len(self.variables)):
            kkt_conditions.append(dL_dx[i] - vs[i])
        for i in range(n_constraints):
            kkt_conditions.append(dL_dlambda[i] + ws[i])

        primal_feasibility = [constr <= 0 for constr in self.constraints] + [var >= 0 for var in self.variables]
        dual_feasibility = [lam >= 0 for lam in lambdas] + [v >= 0 for v in vs] + [w >= 0 for w in ws]

        complementary_slackness = [(self.variables[i], vs[i]) for i in range(len(self.variables))]
        complementary_slackness += [(lambdas[i], ws[i]) for i in range(n_constraints)]

        all_variables = self.variables + lambdas + vs + ws

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
        self.log_emitter.log_signal.emit("🔧 Adding artificial variables...")

        variables = self.kkt_system['all_variables']
        vs = self.kkt_system['vs']
        ws = self.kkt_system['ws']
        z_vars = []
        modified_equations = []
        z_expressions = {}
        artificial_info = []

        for i, eq in enumerate(self.kkt_system['kkt_conditions']):
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

        self.artificial_system = {
            'z_vars': z_vars,
            'z_expressions': z_expressions,
            'F_z': sum(z_vars) if z_vars else None,
            'F_z_expanded': sum(z_expressions[z] for z in z_vars) if z_vars else None,
            'modified_equations': modified_equations,
            'artificial_info': artificial_info
        }

        self._log_artificial_variables()

    def _log_artificial_variables(self):
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
        self.log_emitter.log_signal.emit("🔧 Building simplex table...")

        variables_order = (
                self.kkt_system['variables'] +
                self.kkt_system['lambdas'] +
                self.kkt_system['vs'] +
                self.kkt_system['ws'] +
                self.artificial_system['z_vars']
        )

        headers = ['Basis', 'RHS'] + [str(var) for var in variables_order]
        table = PrettyTable(headers)
        table.float_format = ".2f"

        rows_data = []
        for eq, rhs in self.artificial_system['modified_equations']:
            coeffs = [eq.coeff(var) for var in variables_order]
            basis_var = None

            for z_var in self.artificial_system['z_vars']:
                if eq.coeff(z_var) == 1:
                    basis_var = z_var
                    break

            if not basis_var:
                for w_var in self.kkt_system['ws']:
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
        if self.artificial_system['F_z_expanded'] is not None:
            f_coeffs = [-self.artificial_system['F_z_expanded'].coeff(var) for var in variables_order]
            f_constant = self.artificial_system['F_z_expanded'].as_coeff_add(*variables_order)[0]
            row = ['F', float(f_constant)] + [float(coef) for coef in f_coeffs]
            table.add_row(row)
            f_row_data = {
                'constant': f_constant,
                'coeffs': f_coeffs
            }

        self.simplex_data = {
            'table': table,
            'rows_data': rows_data,
            'f_row_data': f_row_data,
            'variables_order': variables_order,
            'headers': headers
        }

        self.log_emitter.log_signal.emit("### Initial Simplex Table ###\n" + str(table))

    def _solve_simplex(self):
        self.log_emitter.log_signal.emit("🔧 Solving with simplex method...")

        variables_order = self.simplex_data['variables_order']
        z_vars = self.artificial_system['z_vars']
        vs = self.kkt_system['vs']
        ws = self.kkt_system['ws']
        complementary_slackness = self.kkt_system['complementary_slackness']

        simplex_table = self.simplex_data['table']
        table_rows = simplex_table._rows
        headers = self.simplex_data['headers']

        self.current_iteration = 0
        basis_history = set()
        self.solution_results = {'iterations': []}
        self.points = []  # Initialize points list for [x, y] coordinates

        while self.current_iteration < self.max_iterations and self._is_running:
            self.current_iteration += 1
            iteration_info = {
                'iteration': self.current_iteration,
                'table': copy.deepcopy(simplex_table),
                'current_solution': {},
                'objective_value': None
            }

            # Convert rows to Fraction for precision
            frac_rows = []
            for row in table_rows:
                new_row = [row[0]] + [Fraction(str(val)).limit_denominator() for val in row[1:]]
                frac_rows.append(new_row)

            # Extract current basis
            current_basis = tuple(row[0] for row in frac_rows if row[0] != 'F')
            basis_str = str(current_basis)
            iteration_info['current_basis'] = current_basis

            if basis_str in basis_history:
                iteration_info['cycle_detected'] = True
                self.solution_results['iterations'].append(iteration_info)
                final_solution = self._handle_final_solution(frac_rows, variables_order, z_vars, vs, ws,
                                                             complementary_slackness)
                self.solution_results['solution'] = final_solution
                self.log_emitter.log_signal.emit("🔄 Cycle detected in simplex method!")
                # Emit final points
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            basis_history.add(basis_str)

            # Compute current solution
            solution = {}
            for row in frac_rows:
                if row[0] != 'F':
                    basis_var = row[0]
                    free_term = float(row[1])
                    solution[basis_var] = free_term

            for var in variables_order:
                if str(var) not in solution:
                    solution[str(var)] = 0.0

            self.current_solution = solution
            iteration_info['current_solution'] = solution

            # Store [x, y] coordinates and compute objective value
            obj_value = None
            if 'x' in solution and 'y' in solution:
                try:
                    x, y = solution['x'], solution['y']
                    self.points.append([x, y])  # Append [x, y] to points
                    obj_value = float(sp.lambdify(self.variables, self.function)(x, y))
                    iteration_info['objective_value'] = obj_value

                    # Emit points as a 2D numpy array for visualization
                    points_array = np.array(self.points, dtype=float)  # Shape: (n, 2)
                    self.update_signal.emit(points_array)

                except Exception as e:
                    self.log_emitter.log_signal.emit(f"⚠ Error calculating objective: {str(e)}")

            # Log simplex table and objective value before pivot operations
            table_msg = [
                f"\n### Simplex Table at Iteration {self.current_iteration} ###",
                str(simplex_table)
            ]
            if obj_value is not None:
                table_msg.append(f"Objective Function Value: {obj_value:.6f}")
            else:
                table_msg.append("Objective Function Value: Not computed (x or y missing)")
            self.log_emitter.log_signal.emit("\n".join(table_msg))

            # Check optimality
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
                self.log_emitter.log_signal.emit("✅ Optimal solution found!")
                # Emit final points
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            # Select pivot column
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
                                    candidate['disqualified_reason'] = f"{var2} basis and positive ({row[1]})"
                                    break
                        elif var == var2:
                            for row in frac_rows:
                                if row[0] == str(var1) and row[1] > 0:
                                    can_use = False
                                    candidate['disqualified_reason'] = f"{var1} basis and positive ({row[1]})"
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
                self.log_emitter.log_signal.emit("❌ No suitable pivot column found!")
                # Emit final points
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            # Select pivot row
            min_ratio = float('inf')
            pivot_row_idx = None
            pivot_row_var = None

            ratio_data = []
            for i, row in enumerate(frac_rows):
                if row[0] == 'F' or row[0] == str(pivot_col_var):
                    ratio_data.append({
                        'row_var': row[0],
                        'skipped': True,
                        'reason': 'F or matches pivot column'
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
                        'reason': 'Coefficient in pivot column <= 0'
                    })

            iteration_info['ratio_data'] = ratio_data
            iteration_info['pivot_row'] = {'var': pivot_row_var, 'index': pivot_row_idx, 'ratio': min_ratio}

            if pivot_row_idx is None:
                iteration_info['unbounded'] = True
                self.solution_results['iterations'].append(iteration_info)
                self.log_emitter.log_signal.emit("❌ Problem is unbounded!")
                # Emit final points
                self.update_signal.emit(np.array(self.points, dtype=float))
                break

            # Pivot element
            pivot_element = frac_rows[pivot_row_idx][pivot_col_idx]
            iteration_info['pivot_element'] = float(pivot_element)

            # Create new table
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

            # Update basis variable
            new_rows[pivot_row_idx][0] = str(pivot_col_var)

            # Convert Fraction to float for display
            table_rows = []
            for row in new_rows:
                display_row = [row[0]] + [float(val) for val in row[1:]]
                new_table.add_row(display_row)
                table_rows.append(display_row)

            frac_rows = new_rows
            simplex_table = new_table

            self.solution_results['iterations'].append(iteration_info)
            self._log_simplex_iteration(iteration_info)

        if self.current_iteration >= self.max_iterations:
            final_solution = self._handle_final_solution(frac_rows, variables_order, z_vars, vs, ws,
                                                         complementary_slackness)
            self.solution_results['solution'] = final_solution
            self.log_emitter.log_signal.emit(f"⚠ Reached maximum iterations ({self.max_iterations})")
            # Emit final points
            self.update_signal.emit(np.array(self.points, dtype=float))

    def _log_simplex_iteration(self, iteration_info):
        msg = [
            f"\n### Iteration {iteration_info['iteration']} ###",
            str(iteration_info['table']),
            "",
            "### F-row coefficients ###"
        ]

        for var, coef in iteration_info['f_coeffs'].items():
            msg.append(f"{var}: {coef:.6f}")

        msg.append("\n### Pivot Selection ###")
        msg.append(
            f"Leading column: {iteration_info['pivot_col']['var']} (coef: {iteration_info['pivot_col']['coef']:.6f})")

        msg.append("\n### Ratios ###")
        for ratio in iteration_info['ratio_data']:
            if 'skipped' in ratio:
                msg.append(f"Row {ratio['row_var']}: skipped ({ratio['reason']})")
            else:
                msg.append(
                    f"Row {ratio['row_var']}: {ratio['free_term']:.6f} / {ratio['pivot_col_val']:.6f} = {ratio['ratio']:.6f}")

        msg.append(
            f"\nLeading row: {iteration_info['pivot_row']['var']} (ratio: {iteration_info['pivot_row']['ratio']:.6f})")
        msg.append(f"Pivot element: {iteration_info['pivot_element']:.6f}")

        if 'objective_value' in iteration_info:
            msg.append(f"\nCurrent objective value: {iteration_info['objective_value']:.6f}")

        self.log_emitter.log_signal.emit("\n".join(msg))

    def _handle_final_solution(self, frac_rows, variables_order, z_vars, vs, ws, complementary_slackness):
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

        self._log_final_solution(result)
        return result

    def _log_final_solution(self, final_solution):
        msg = ["\n### Final Solution ###"]

        if final_solution['is_feasible']:
            msg.append("✅ Found feasible optimal solution")
            msg.append("Solution:")

            for var, value in final_solution['solution'].items():
                msg.append(f"{var} = {value:.6f}")

            msg.append(f"Objective value: {final_solution['f_value']:.6f}")

            if final_solution['additional_in_basis']:
                msg.append("\nActive constraints:")
                for constraint in final_solution['active_constraints']:
                    if constraint['type'] == 'non_negativity':
                        msg.append(
                            f"  {constraint['var']} > 0 => {constraint['original_var']} = 0 (non-negativity constraint active)")
                    else:
                        msg.append(f"  {constraint['var']} > 0 => {constraint['original_var']} = 0 (constraint inactive)")
        else:
            msg.append("❌ No feasible solution found")
            msg.append(f"F value: {final_solution['f_value']:.6f}")
            msg.append(f"Artificial variables in basis: {final_solution['artificial_in_basis']}")

        self.log_emitter.log_signal.emit("\n".join(msg))

    def _verify_with_scipy(self):
        self.log_emitter.log_signal.emit("\n🔍 Verifying solution with scipy.optimize.minimize...")

        # Преобразование целевой функции
        def objective(vars):
            return sp.lambdify(self.variables, self.function, 'numpy')(vars[0], vars[1])

        # Преобразование ограничений
        scipy_constraints = []
        for constr in self.constraints:
            def constraint_func(vars, c=constr):
                return -sp.lambdify(self.variables, c, 'numpy')(vars[0], vars[1])  # -g_i(x, y) >= 0

            scipy_constraints.append({
                'type': 'ineq',
                'fun': constraint_func
            })

        # Границы: x >= 0, y >= 0
        bounds = [(0, None), (0, None)]

        # Начальное приближение
        initial_guess = [0, 0]

        # Решение задачи
        self.scipy_result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints
        )

        # Логирование результатов
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

        # Сравнение с симплекс-методом
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