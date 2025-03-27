from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import time


class GradientDescent(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self.pointX = params_dict['point'][0]
        self.pointY = params_dict['point'][1]
        self.firstEps = params_dict['epsilons'][0]
        self.secondEps = params_dict['epsilons'][1]
        self.thirdEps = params_dict['epsilons'][2]
        self.initial_step = params_dict['initial_step']
        self.max_iterations = params_dict['max_iterations']
        self.function = params_dict['function']

        self.log_emitter = log_emitter
        self._is_running = False
        self.initial_delay = 0.05
        self.min_delay = 0.001

    def _compute_gradient(self, x, y, h=1e-5):
        fx_plus = self.function(x + h, y)
        fx_minus = self.function(x - h, y)
        df_dx = (fx_plus - fx_minus) / (2 * h)

        fy_plus = self.function(x, y + h)
        fy_minus = self.function(x, y - h)
        df_dy = (fy_plus - fy_minus) / (2 * h)

        return df_dx, df_dy

    def run(self):
        self._is_running = True
        self.log_emitter.log_signal.emit("🔹 Optimization started...")

        try:
            x = np.array([self.pointX, self.pointY])
            t = self.initial_step
            prev_func_value = self.function(x[0], x[1])

            points = [x.copy()]
            for k in range(self.max_iterations):
                if not self._is_running:
                    break

                grad_x, grad_y = self._compute_gradient(x[0], x[1])
                current_grad = np.array([grad_x, grad_y])
                grad_norm = np.linalg.norm(current_grad)

                message = (
                    f"Iteration {k + 1}:\n"
                    f"📍 Point: ({x[0]:.6f}, {x[1]:.6f})\n"
                    f"📉 Function value: {prev_func_value:.6f}\n"
                    f"🔽 Step size: {t:.6f}\n"
                    f"------------------------------------\n"
                )
                self.log_emitter.log_signal.emit(message)

                if grad_norm < self.firstEps:  # First stop condition
                    self.log_emitter.log_signal.emit(
                        f"✅ Stopping: Gradient norm {grad_norm:.6f} < {self.firstEps}"
                    )
                    break

                nx = x - t * current_grad
                current_func_value = self.function(nx[0], nx[1])
                func_diff = current_func_value - prev_func_value

                if func_diff < 0:
                    step_diff = np.linalg.norm(nx - x)

                    if step_diff < self.secondEps and abs(func_diff) < self.secondEps:  # Second stop condition
                        self.log_emitter.log_signal.emit("✅ Stopping: Small step and function change")
                        points.append(nx.copy())
                        x = nx
                        break

                    x = nx
                    prev_func_value = current_func_value
                    points.append(x.copy())
                    self.update_signal.emit(np.array(points))
                else:
                    t /= 2
                    self.log_emitter.log_signal.emit(f"🔻 Reducing step size to: {t:.6f}")

                delay = max(self.min_delay, self.initial_delay * (0.95 ** k))
                time.sleep(delay)

            self.optimization_path = np.array(points)

            self.pointX, self.pointY = float(x[0]), float(x[1])
            final_message = (
                "🎉 Optimization finished!\n"
                f"🏁 Final point: ({x[0]:.6f}, {x[1]:.6f})\n"
                f"📊 Final value: {prev_func_value:.6f}"
            )
            self.log_emitter.log_signal.emit(final_message)

        except Exception as e:
            self.log_emitter.log_signal.emit(f"❌ Error: {str(e)}")
        finally:
            self.update_signal.emit(np.array(points))
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.log_emitter.log_signal.emit("⏹ Optimization stopped by user")

    @property
    def pointX(self):
        return self._pointX

    @pointX.setter
    def pointX(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("X coordinate must be numeric")
        self._pointX = float(value)

    @property
    def pointY(self):
        return self._pointY

    @pointY.setter
    def pointY(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Y coordinate must be numeric")
        self._pointY = float(value)

    @property
    def firstEps(self):
        return self._firstEps

    @firstEps.setter
    def firstEps(self, value):
        if not isinstance(value, float) or not (0 < value < 1):
            raise ValueError("First epsilon must be float between 0 and 1")
        self._firstEps = value

    @property
    def secondEps(self):
        return self._secondEps

    @secondEps.setter
    def secondEps(self, value):
        if not isinstance(value, float) or not (0 < value < 1):
            raise ValueError("Second epsilon must be float between 0 and 1")
        self._secondEps = value

    @property
    def thirdEps(self):
        return self._thirdEps

    @thirdEps.setter
    def thirdEps(self, value):
        if not isinstance(value, float) or not (0 < value < 1):
            raise ValueError("Third epsilon must be float between 0 and 1")
        self._thirdEps = value

    @property
    def initial_step(self):
        return self._initial_step

    @initial_step.setter
    def initial_step(self, value):
        if not isinstance(value, float) or value <= 0:
            raise ValueError("Initial step must be positive float")
        self._initial_step = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Max iterations must be positive integer")
        self._max_iterations = value

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        if value is None:
            raise ValueError("Function cannot be None")
        self._function = value