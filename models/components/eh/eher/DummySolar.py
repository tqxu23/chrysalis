from models.components.eh.eher.EHer import EHer


class DummySolar(EHer):
    name = "DummySolar"
    param_list = {"eh_area": 1e-4, "eh_efficiency": 0.15, "eh_power": 1}

    def __init__(self, step_size, param_list):
        super().__init__(step_size, param_list)

    def harvest(self, harvest_params) -> float:
        return self.param_list["eh_area"] * self.param_list["eh_efficiency"] * \
            self.param_list["eh_power"] / self.step_size
