from models.components.insitu.Insitu import Insitu


class DummyInsitu(Insitu):
    name = "DummyInsitu (Resistor)"
    param_list = {"insitu_power": 1e-3}
    count = 0

    # step_params: out_power
    def step(self, params=None) -> float:
        self.count = self.count + 1
        return 1e-3 / self.step_size

    def shut(self):
        pass