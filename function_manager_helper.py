import json
import numpy as np

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
                locals_dict = {'np': np}
                func_object = eval(func_data['python_formula'], {'np': np}, {})

                test_value = func_object(0, 0)

                self.functions.append({
                    'name': func_data['name'],
                    'formula': func_data['formula'],
                    'python_formula': func_data['python_formula'],
                    'function': func_object
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
        for func in self.functions:
            combo_box.addItem(func['name'])
