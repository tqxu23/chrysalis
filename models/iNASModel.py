from models.Model import Model
from models.components.eh.DummyEH import DummyEH
from models.components.insitu.INASInsitu import INASInsitu
from models.components.insitu.INASCostCore import test_cnn_cost
from models.components.insitu.INASCostCore.test_cnn_cost import Mat
import utils


class iNASModel(Model):
    name = "iNASModel"
    layer = {}
    layer['type'] = "CONV"
    layer['K'] = Mat(0, 4, 3, 5, 5)
    layer['OFM'] = Mat(0, 1, 4, 28, 28)
    layer['IFM'] = Mat(0, 1, 3, 32, 32)
    maxTr = layer['OFM'].h
    maxTc = layer['OFM'].w
    maxTm = layer['OFM'].ch
    maxTn = layer['IFM'].ch
    layer['tile_size'] = [1, 1, 1, 1, 7, 28, 4, 3]
    layer['stride'] = 1
    param_list = {"insitu_power": 1e-3, "cap_volume": 1e-3, "cap_voltage": 2.79,
                  "eh_area": 1e-4, "eh_efficiency": 0.15,  "layer": [layer],
                  "eh_Solar": 100 / 0.15, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.1, "v_off": 2.8, "v_max": 4.1, "strategy": "runTime"} \
        # base, iNAS, greedy, runTime
    throughput = 0
    step_size = 100000
    latency_mode = False
    step_count = 0
    # step_params: out_power
    def __init__(self, step_size, param_list=None):
        super().__init__(step_size, param_list)
        self.eh = DummyEH(step_size, param_list)
        self.insitu = INASInsitu(step_size, param_list)
        if param_list["mode"] == "latency":
            self.latency_mode = True
        self.throughput = 0

    def step(self, params=None):
        self.step_count = self.step_count+1
        if self.eh.stateOn:
            power = self.eh.envir.step("eh_Solar")
            in_power = self.eh.eher.harvest(power)
            params = {"curVoltage": self.eh.cap.voltage, "vOff": self.eh.v_off, "capVolume": self.eh.cap.volume,
                      "inPower": in_power}
            cont, out_power = self.insitu.step(params)
            if not self.eh.step(out_power):
                print("wrong in out power")
                return True, True
                # return True
            if not cont:
                self.eh.stateOn = False
            self.throughput = self.insitu.count
            if self.throughput >= 1 and self.latency_mode:
                return False, False
        else:
            self.insitu.shut()
            self.eh.step(0)
        return True, False
