from models.Model import Model
from models.components.eh.DummyEH import DummyEH
from models.components.insitu.DummyInsitu import DummyInsitu
import utils


class DummyModel(Model):
    name = "DummyModel"
    param_list = {"insitu_power": 1e-3, "cap_volume": 1e-3, "cap_voltage": 0,
                  "eh_area": 1e-4, "eh_efficiency": 0.15, "eh_power": 100,
                  "eh_Solar": 100/0.15}
    throughput = 0
    step_size = 100000

    # step_params: out_power
    def __init__(self, step_size, param_list=None):
        super().__init__(step_size, param_list)
        self.eh = DummyEH(step_size, param_list)
        self.insitu = DummyInsitu(step_size, param_list)
        self.throughput = 0

    def step(self, params=None):
        if self.eh.stateOn:
            out_power = self.insitu.step()
            if not self.eh.step(out_power):
                return False
            self.throughput = self.insitu.count
        else:
            self.insitu.shut()
            self.eh.step(0)
        return True
