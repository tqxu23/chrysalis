import argparse

from models.Model import Model
from models.components.eh.DummyEH import DummyEH
from models.components.insitu.GammaInsitu import GammaInsitu
from models.components.insitu.GammaInsitu import test_cost
import utils
from models.components.insitu.GammaCostCore.ExecInfo import ExecInfo


class GammaModel(Model):
    name = "gammaModel"
    param_list = {"mode": "", "insitu_power": 1e-3, "cap_volume": 1e-3, "cap_voltage": 0,
                  "eh_area": 1e-4, "eh_efficiency": 0.15, "eh_power": 100,
                  "eh_Solar": 100 / 0.15, '--num_pe': 512, '--l1_size': 1024, '--l2_size': 10240, '--inter': True,
                  '--arch': "tpu_accel"}
    throughput = 0
    step_size = 100000
    step_count = 0
    mapping = []
    execInfo = []
    # step_params: out_power
    def __init__(self, step_size, param_list=None):
        super().__init__(step_size, param_list)
        layer_num = param_list["layer_num"]
        cost = []
        inter = []
        runtime = []
        energy = []
        pe = []
        if self.param_list["mode"] == "layer":
            # for i in range(layer_num):
            cost.clear()
            cost.append(test_cost(self.param_list))
            self.mapping.clear()
            self.mapping.append(cost[0]['Mapping'])
            # print(self.mapping)
            inter.append(self.mapping[0][1][1])
            runtime.append(cost[0]["Runtime"])
            energy.append(cost[0]["Energy"])
            pe.append(cost[0]["Num_PE"])
            # area = cost[i]["Area"]
            # NumPE = cost[i]["Num_PE"]
            # L1Buffer = cost[i]["L1Buffer"]
            # L2Buffer = cost[i]["L2Buffer"]
            self.execInfo.clear()
            self.execInfo.append(ExecInfo())
            if self.param_list['--inter']:
                self.execInfo[0].frequency = 3e7
                self.execInfo[0].latency = runtime[0] / inter[0] / self.execInfo[0].frequency
                self.execInfo[0].power = energy[0] / (runtime[0] / self.execInfo[0].frequency) * 1e-9
                self.execInfo[0].all_num = inter[0]
            else:
                self.execInfo[0].frequency = 3e7
                self.execInfo[0].latency = runtime[0] / self.execInfo[0].frequency
                self.execInfo[0].power = energy[0] / (runtime[0] / self.execInfo[0].frequency) * 1e-9
                self.execInfo[0].all_num = 1
        elif param_list["mode"] == "network":
            self.execInfo = param_list["ExecInfo"]

        param_list["ExecInfo"] = self.execInfo
        param_list["Num_PE"] = pe
        self.eh = DummyEH(step_size, param_list)
        self.insitu = GammaInsitu(step_size, param_list)
        self.throughput = 0

    def step(self, params=None):
        self.step_count = self.step_count + 1
        if self.eh.stateOn:
            out_power = self.insitu.step()
            if not self.eh.step(out_power):
                # print("power mistake")
                return True
            self.throughput = self.insitu.count
            if self.throughput >= 1:
                if self.step_count==1:
                    self.step_count = 50000
                return False
        else:
            self.insitu.shut()
            self.eh.step(0)
        return True
