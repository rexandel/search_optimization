import numpy as np
from scipy.optimize import minimize
from PyQt5.QtCore import QObject, pyqtSignal
import time

class LibrarySimplexMethod(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray, bool)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self.function = params_dict['function']
        self.constraints = params_dict['constraints']
        self.log_emitter = log_emitter
        self._is_running = False
        self.all_points = []

    def run(self):
        self._is_running = True
        self.all_points = []
        self.log_emitter.log_signal.emit(
            "🔹 Starting optimization with scipy.optimize.minimize...\n"
            "• Objective function and constraints loaded from parameters"
        )

        self.update_signal.emit(np.empty((0, 2)), False)

        try:
            start_time = time.time()
            self.log_emitter.log_signal.emit("\n📊 Setting up optimization...")

            # Log constraints
            if self.constraints:
                constraint_msg = ["### Optimization Constraints ###"]
                for i, constr in enumerate(self.constraints, 1):
                    constraint_msg.append(f"g{i}: {constr['formula']}")
                self.log_emitter.log_signal.emit("\n".join(constraint_msg))
            else:
                self.log_emitter.log_signal.emit("### Optimization Constraints ###\nNo explicit constraints provided (assuming x >= 0, y >= 0)")

            # Define the objective function for minimize
            def objective(vars):
                return self.function(vars[0], vars[1])

            # Prepare constraints for minimize
            scipy_constraints = []
            for constr in self.constraints:
                # Constraints are in the form g(x,y) <= 0
                def constraint_func(vars, c=constr['function']):
                    return -c(vars[0], vars[1])  # Convert to -g(x,y) >= 0

                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': constraint_func
                })

            # Define bounds for non-negativity (x >= 0, y >= 0)
            bounds = [(0, None), (0, None)]

            # Initial guess
            initial_guess = np.array([0.0, 0.0])

            # Perform optimization
            self.log_emitter.log_signal.emit("\n🔍 Running optimization...")
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'disp': True, 'maxiter': 1000}
            )

            # Log detailed results
            end_time = time.time()
            total_time = end_time - start_time

            if result.success:
                self.all_points.append(np.array([[result.x[0], result.x[1]]]))
                self.update_signal.emit(np.concatenate(self.all_points), False)

                # Evaluate constraints at the solution
                constraint_values = []
                for i, constr in enumerate(self.constraints):
                    value = constr['function'](result.x[0], result.x[1])
                    constraint_values.append(f"g{i+1}(x,y) = {value:.6f}")

                report = [
                    "\n🎉 OPTIMIZATION RESULTS",
                    "════════════════════════",
                    f"• Optimal point: ({result.x[0]:.6f}, {result.x[1]:.6f})",
                    f"• Function value: {result.fun:.6f}",
                    f"• Success: {result.success}",
                    f"• Message: {result.message}",
                    f"• Iterations: {result.nit}",
                    f"• Function evaluations: {result.nfev}",
                    f"• Total computation time: {total_time:.2f} seconds",
                    "════════════════════════",
                    "Constraint values:"
                ]
                report.extend(constraint_values)
            else:
                report = [
                    "\n❌ OPTIMIZATION FAILED",
                    "════════════════════════",
                    f"• Message: {result.message}",
                    f"• Success: {result.success}",
                    f"• Total computation time: {total_time:.2f} seconds",
                    "════════════════════════"
                ]

            self.log_emitter.log_signal.emit("\n".join(report))

        except Exception as e:
            self.log_emitter.log_signal.emit(f"\n❌ ERROR: {str(e)}")
        finally:
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.log_emitter.log_signal.emit("\n⏹ Optimization stopped by user")