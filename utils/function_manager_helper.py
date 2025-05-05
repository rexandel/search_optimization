import json
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

class FunctionManagerHelper:
    def __init__(self, json_file_path=None):
        self.functions = []
        self.current_function_index = 0

        if json_file_path:
            self.load_functions_from_file(json_file_path)

    def load_functions_from_file(self, json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if 'functions' not in data:
                print("Error: JSON file does not contain 'functions' key")
                return False

            self.parse_functions(data['functions'])
            return True
        except Exception as e:
            print(f"Error loading functions from JSON: {str(e)}")
            return False

    def parse_functions(self, functions_data):
        self.functions = []

        for func_data in functions_data:
            if not all(key in func_data for key in ['name', 'formula', 'python_formula']):
                print(f"Skipping incomplete function definition: {func_data.get('name', 'Unnamed')}")
                continue

            try:
                func_object = eval(func_data['python_formula'], {'np': np}, {})

                constraints = []
                if 'constraints' in func_data and isinstance(func_data['constraints'], list):
                    for constraint in func_data['constraints']:
                        if 'python_formula' in constraint:
                            try:
                                constraint_func = eval(constraint['python_formula'], {'np': np}, {})
                                constraints.append({
                                    'formula': constraint.get('formula', ''),
                                    'function': constraint_func
                                })
                            except Exception as e:
                                print(f"Error creating constraint: {str(e)}")

                self.functions.append({
                    'name': func_data['name'],
                    'formula': func_data['formula'],
                    'python_formula': func_data['python_formula'],
                    'function': func_object,
                    'constraints': constraints
                })
            except Exception as e:
                print(f"Error creating function {func_data['name']}: {str(e)}")

    def get_function_names(self):
        return [func['name'] for func in self.functions]

    def get_function_by_index(self, index):
        if 0 <= index < len(self.functions):
            return self.functions[index]
        return None

    def get_function_by_name(self, name):
        for func in self.functions:
            if func['name'] == name:
                return func
        return None

    def evaluate_function(self, func_index, x, y):
        try:
            func = self.get_function_by_index(func_index)
            if func:
                return func['function'](x, y)
            return None
        except Exception as e:
            print(f"Error evaluating function: {str(e)}")
            return None

    def set_current_function(self, index):
        if 0 <= index < len(self.functions):
            self.current_function_index = index
            return True
        return False

    def get_current_function(self):
        if not self.functions:
            return None
        return self.functions[self.current_function_index]

    def populate_combo_box(self, combo_box):
        combo_box.clear()
        if len(self.functions) > 0:
            for func in self.functions:
                combo_box.addItem(func['name'])

    def to_symbolic_function(self, func_index):
        """
        Convert a function and its constraints at the given index to symbolic SymPy expressions
        by parsing the python_formula (lambda expression). Excludes non-negativity constraints
        (x >= 0 and y >= 0, represented as -x <= 0 and -y <= 0).

        Args:
            func_index (int): Index of the function in self.functions.

        Returns:
            dict: Dictionary containing:
                - 'variables': List of SymPy symbols [x, y].
                - 'objective': SymPy expression for the objective function.
                - 'constraints': List of SymPy expressions for constraints (excluding x >= 0, y >= 0).
                - 'success': Boolean indicating if conversion was successful.
                - 'error': Error message if conversion failed (None if successful).
        """
        if not 0 <= func_index < len(self.functions):
            return {
                'variables': None,
                'objective': None,
                'constraints': None,
                'success': False,
                'error': f"Invalid function index: {func_index}"
            }

        func_data = self.functions[func_index]
        x, y = sp.symbols('x y', real=True)
        variables = [x, y]

        # Define parsing transformations for robust formula parsing
        transformations = (standard_transformations + (implicit_multiplication_application,))

        try:
            # Extract expression from python_formula (e.g., 'lambda x, y: expr' -> 'expr')
            python_formula = func_data['python_formula'].strip()
            if not python_formula:
                raise ValueError("Empty python_formula string")

            # Match 'lambda x, y: expression'
            match = re.match(r'^\s*lambda\s+x\s*,\s*y\s*:\s*(.+)$', python_formula)
            if not match:
                raise ValueError("Invalid lambda expression format")

            expr_str = match.group(1).strip()
            if not expr_str:
                raise ValueError("Empty expression in lambda")

            # Parse the objective function expression
            objective = parse_expr(
                expr_str,
                local_dict={'x': x, 'y': y},
                transformations=transformations
            )

            # Parse constraints, excluding x >= 0 and y >= 0
            constraints = []
            for constraint in func_data.get('constraints', []):
                constraint_formula = constraint.get('python_formula', '').strip()
                if not constraint_formula:
                    print(f"Warning: Empty constraint python_formula for function {func_data['name']}")
                    continue

                try:
                    # Extract expression from constraint python_formula
                    match = re.match(r'^\s*lambda\s+x\s*,\s*y\s*:\s*(.+)$', constraint_formula)
                    if not match:
                        raise ValueError("Invalid constraint lambda expression format")

                    constraint_expr_str = match.group(1).strip()
                    if not constraint_expr_str:
                        raise ValueError("Empty expression in constraint lambda")

                    constraint_expr = parse_expr(
                        constraint_expr_str,
                        local_dict={'x': x, 'y': y},
                        transformations=transformations
                    )

                    # Skip non-negativity constraints (x >= 0, y >= 0)
                    if constraint_expr == -x or constraint_expr == -y:
                        continue

                    constraints.append(constraint_expr)
                except Exception as e:
                    print(f"Error parsing constraint python_formula '{constraint_formula}': {str(e)}")
                    continue

            return {
                'variables': variables,
                'objective': objective,
                'constraints': constraints,
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'variables': None,
                'objective': None,
                'constraints': None,
                'success': False,
                'error': f"Error parsing function '{func_data['name']}': {str(e)}"
            }