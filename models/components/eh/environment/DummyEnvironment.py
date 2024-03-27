from models.components.eh.environment.Environment import Environment


class DummyEnvironment(Environment):
    name = "DummyEnvironment"
    param_list = {"eh_Solar": 0}

    def __init__(self, step_size, param_list):
        super().__init__(step_size, param_list)

    def step(self, name) -> float:
        return self.param_list[name] / self.step_size
