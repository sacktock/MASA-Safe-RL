from typing import Callable, Union

Schedule = Callable[[float], float]

class ConstantSchedule():

    def __init__(self, value: float):
        assert value is float, "Expected value: float for class ConstantSchedule __init__(value)"
        self.value = value

    def __call__(self, step: int) -> float:
        return self.value

class FloatSchedule():

    def __init__(self, value_schedule: Union[Schedule, float]):

        if isinstance(value_schedule, Schedule):
            self.schedule: Schedule = value_schedule
        elif isinstance(value_schedule, float):
            self.schedule = ConstantSchedule(value_schedule)
        else:
            assert callable(value_schedule), f"value_schedule must be callable or float for class FloatSchedule __init__(value_schedule), not {value_schedule}"
            self.schedule = value_schedule

    def __call__(self, step: int) -> float:
        return self.schedule(step)
        