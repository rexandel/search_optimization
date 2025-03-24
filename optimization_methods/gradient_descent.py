class GradientDescentParams:
    def __init__(self, params_dict):
        self.pointX = params_dict['point'][0]
        self.pointY = params_dict['point'][1]
        self.firstEps = params_dict['epsilons'][0]
        self.secondEps = params_dict['epsilons'][1]
        self.thirdEps = params_dict['epsilons'][2]
        self.initial_step = params_dict['initial_step']
        self.max_iterations = params_dict['max_iterations']
        self.function = params_dict['function']

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
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Initial step must be positive integer")
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