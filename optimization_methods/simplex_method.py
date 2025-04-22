import numpy as np
from scipy.optimize import linprog


class SimplexMethod:
    @staticmethod
    def piecewise_linear_approximation_2d(func, x_range, y_range, num_segments_x, num_segments_y):
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_points = np.linspace(x_min, x_max, num_segments_x + 1)
        y_points = np.linspace(y_min, y_max, num_segments_y + 1)

        linear_approx = []

        for i in range(num_segments_x):
            for j in range(num_segments_y):
                x1, x2 = x_points[i], x_points[i + 1]
                y1, y2 = y_points[j], y_points[j + 1]

                f_x1_y1 = func(x1, y1)
                f_x1_y2 = func(x1, y2)
                f_x2_y1 = func(x2, y1)
                f_x2_y2 = func(x2, y2)

                X = np.array([[1, x1, y1],
                              [1, x1, y2],
                              [1, x2, y1],
                              [1, x2, y2]])
                Y = np.array([f_x1_y1, f_x1_y2, f_x2_y1, f_x2_y2])

                coef = np.linalg.lstsq(X, Y, rcond=None)[0]

                linear_approx.append({
                    'coefs': coef,
                    'x_range': (x1, x2),
                    'y_range': (y1, y2)
                })

        return (x_points, y_points), linear_approx

    @staticmethod
    def find_segment_minimum(segment):
        a0, a1, a2 = segment['coefs']
        x1, x2 = segment['x_range']
        y1, y2 = segment['y_range']

        c = [a1, a2]

        bounds = [
            (x1, x2),
            (y1, y2)
        ]

        res = linprog(c, bounds=bounds, method='highs')

        if res.success:
            x_min, y_min = res.x
            min_value = a0 + a1 * x_min + a2 * y_min
            return {
                'x': x_min,
                'y': y_min,
                'value': min_value,
                'segment_coefs': [a0, a1, a2],
                'segment_range': segment['x_range'] + segment['y_range']
            }
        else:
            return None

    @staticmethod
    def find_global_minimum(func, x_range, y_range, num_segments_x, num_segments_y):
        _, segments = SimplexMethod.piecewise_linear_approximation_2d(
            func, x_range, y_range, num_segments_x, num_segments_y)

        minima = []
        for segment in segments:
            segment_min = SimplexMethod.find_segment_minimum(segment)
            if segment_min:
                minima.append(segment_min)

        if not minima:
            return None

        global_min = min(minima, key=lambda x: x['value'])

        return {
            'global_min': global_min,
            'all_minima': minima,
            'num_segments': len(segments)
        }
