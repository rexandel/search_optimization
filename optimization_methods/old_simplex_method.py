import numpy as np
from scipy.optimize import linprog
from PyQt5.QtCore import QObject, pyqtSignal
import time


class OldSimplexMethod(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray, bool)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self.function = params_dict['function']
        self.x_range = params_dict['x_range']
        self.y_range = params_dict['y_range']
        self.num_segments_x = params_dict['num_segments_x']
        self.num_segments_y = params_dict['num_segments_y']
        self.log_emitter = log_emitter
        self._is_running = False
        self.initial_delay = 0.1
        self.min_delay = 0.01
        self.all_points = []

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

                func_values = {
                    'f(x1,y1)': func(x1, y1),
                    'f(x1,y2)': func(x1, y2),
                    'f(x2,y1)': func(x2, y1),
                    'f(x2,y2)': func(x2, y2)
                }

                X = np.array([[1, x1, y1],
                              [1, x1, y2],
                              [1, x2, y1],
                              [1, x2, y2]])
                Y = np.array([func_values['f(x1,y1)'],
                              func_values['f(x1,y2)'],
                              func_values['f(x2,y1)'],
                              func_values['f(x2,y2)']])

                coef, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                linear_approx.append({
                    'coefs': coef,
                    'x_range': (float(x1), float(x2)),
                    'y_range': (float(y1), float(y2)),
                    'residual': float(residuals[0]) if len(residuals) > 0 else 0.0
                })

        return (x_points, y_points), linear_approx

    def find_segment_minimum(self, segment):
        a0, a1, a2 = segment['coefs']
        x1, x2 = segment['x_range']
        y1, y2 = segment['y_range']

        c = [a1, a2]
        bounds = [(x1, x2), (y1, y2)]
        res = linprog(c, bounds=bounds, method='highs')

        if res.success:
            x_min, y_min = res.x
            min_value = a0 + a1 * x_min + a2 * y_min

            self.log_emitter.log_signal.emit(
                f"Found minimum in segment x=[{x1:.2f}, {x2:.2f}], y=[{y1:.2f}, {y2:.2f}]\n"
                f"• Coordinates: ({x_min:.4f}, {y_min:.4f})\n"
                f"• Function value: {min_value:.6f}\n"
                f"• Linear approx: {a0:.2f} + {a1:.2f}x + {a2:.2f}y\n"
                f"• Approximation residual: {segment['residual']:.4e}"
            )

            return {
                'x': float(x_min),
                'y': float(y_min),
                'value': float(min_value),
                'segment_coefs': [float(a0), float(a1), float(a2)],
                'segment_range': (float(x1), float(x2), float(y1), float(y2))
            }
        else:
            self.log_emitter.log_signal.emit(
                f"⚠️ No minimum found in segment x=[{x1:.2f}, {x2:.2f}], y=[{y1:.2f}, {y2:.2f}]"
            )
            return None

    def run(self):
        self._is_running = True
        self.all_points = []
        self.log_emitter.log_signal.emit(
            "🔹 Starting simplex method optimization...\n"
            f"• Search area: x ∈ [{self.x_range[0]}, {self.x_range[1]}], "
            f"y ∈ [{self.y_range[0]}, {self.y_range[1]}]\n"
            f"• Grid size: {self.num_segments_x}x{self.num_segments_y} segments"
        )

        self.update_signal.emit(np.empty((0, 2)), False)

        try:
            self.log_emitter.log_signal.emit("\n📊 Building piecewise linear approximation...")
            start_time = time.time()

            (x_points, y_points), segments = self.piecewise_linear_approximation_2d(
                self.function, self.x_range, self.y_range,
                self.num_segments_x, self.num_segments_y)

            approx_time = time.time() - start_time
            self.log_emitter.log_signal.emit(
                f"✓ Approximation built in {approx_time:.2f} seconds\n"
                f"• Total segments: {len(segments)}\n"
                f"• X grid points: {len(x_points)}\n"
                f"• Y grid points: {len(y_points)}"
            )

            minima = []
            self.log_emitter.log_signal.emit("\n🔍 Searching for local minima in segments...")

            for i, segment in enumerate(segments):
                if not self._is_running:
                    break

                segment_info = (
                    f"\nSegment {i + 1}/{len(segments)}\n"
                    f"• X range: [{segment['x_range'][0]:.2f}, {segment['x_range'][1]:.2f}]\n"
                    f"• Y range: [{segment['y_range'][0]:.2f}, {segment['y_range'][1]:.2f}]"
                )
                self.log_emitter.log_signal.emit(segment_info)

                segment_min = self.find_segment_minimum(segment)
                if segment_min:
                    minima.append(segment_min)
                    new_point = np.array([[segment_min['x'], segment_min['y']]])
                    self.all_points.append(new_point)

                    self.update_signal.emit(np.concatenate(self.all_points), False)

                    delay = max(self.min_delay, self.initial_delay * (0.9 ** i))
                    time.sleep(delay)

            if minima:
                global_min = min(minima, key=lambda x: x['value'])
                total_time = time.time() - start_time

                report = (
                    "\n🎉 OPTIMIZATION RESULTS\n"
                    "════════════════════════\n"
                    f"• Global minimum at: ({global_min['x']:.6f}, {global_min['y']:.6f})\n"
                    f"• Function value: {global_min['value']:.6f}\n"
                    f"• Found {len(minima)} local minima from {len(segments)} segments\n"
                    f"• Total computation time: {total_time:.2f} seconds\n"
                    "════════════════════════\n"
                    "Segment with global minimum:\n"
                    f"• X range: [{global_min['segment_range'][0]:.2f}, {global_min['segment_range'][1]:.2f}]\n"
                    f"• Y range: [{global_min['segment_range'][2]:.2f}, {global_min['segment_range'][3]:.2f}]\n"
                    f"• Approximation: {global_min['segment_coefs'][0]:.2f} + "
                    f"{global_min['segment_coefs'][1]:.2f}x + "
                    f"{global_min['segment_coefs'][2]:.2f}y"
                )
                self.log_emitter.log_signal.emit(report)
            else:
                self.log_emitter.log_signal.emit("\n❌ No minima found in any segment!")

        except Exception as e:
            self.log_emitter.log_signal.emit(f"\n❌ ERROR: {str(e)}")
        finally:
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.log_emitter.log_signal.emit("\n⏹ Optimization stopped by user")
