from models.components.eh.cap.Capacitor import Capacitor
import utils


class DummyCap(Capacitor):
    name = "DummyCap"
    voltage = 0
    param_list = {"cap_volume": 1e-3, "cap_voltage": 0}
    leakagePower = 0
    # I = 0.03CV+40uA
    def __init__(self, step_size, param_list=None):
        super().__init__(step_size, param_list)
        self.voltage = param_list["cap_voltage"]
        self.volume = param_list["cap_volume"]
        self.leakagePower = max(0.09*self.volume,9e-6)/step_size

    def step(self, power):
        if 1 / 2 * self.volume * self.voltage * self.voltage + power < 0:
            return False
        self.voltage = max(0,((1 / 2 * self.volume * self.voltage * self.voltage + power - self.leakagePower) * 2 / self.volume)) ** 0.5
        return True
