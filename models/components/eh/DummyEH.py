from models.components.eh.EH import EH
from models.components.eh.environment.DummyEnvironment import DummyEnvironment
from models.components.eh.cap.DummyCap import DummyCap
from models.components.eh.eher.DummySolar import DummySolar
import utils


class DummyEH(EH):
    name = "DummyEH"
    # envir = DummyEnvironment()
    # cap = DummyCap()
    # eher = DummySolar(100000)
    stateOn = False

    param_list = {"cap_volume": 1e-3, "cap_voltage": 0, "eh_area": 1e-4, "eh_efficiency": 0.15, "eh_Solar": 100/0.15}
    v_on = 3.1
    v_off = 2.8
    v_max = 4.1
    step_size = 100000

    def __init__(self, step_size, param_list=None):
        super().__init__(param_list)
        self.step_size = step_size
        self.envir = DummyEnvironment(step_size, param_list)
        self.cap = DummyCap(step_size, param_list)
        self.eher = DummySolar(step_size, param_list)
        self.stateOn = False
        self.v_on = param_list["v_on"]
        self.v_off = param_list["v_off"]
        self.v_max = param_list["v_max"]

    # step_params: out_power
    def step(self, out_power, params=None) -> bool:
        if params is not None:
            self.refresh_params(params)
        power = self.envir.step("eh_Solar")
        in_power = self.eher.harvest(power)
        if self.cap.voltage > self.v_max:
            self.cap.voltage = self.v_max
        power_change = in_power - out_power
        if not self.cap.step(power_change):
            return False
        voltage = self.cap.voltage
        if voltage > self.v_on and not self.stateOn:
            self.stateOn = True
        if voltage < self.v_off and self.stateOn:
            self.stateOn = False
        return True
