from models.components.insitu.Insitu import Insitu
from models.components.insitu.GammaCostCore.src.GAMMA.test_cost import test_cost
from models.components.insitu.GammaCostCore.ExecInfo import ExecInfo
import numpy as np
import utils


class GammaInsitu(Insitu):
    name = "GammaCostCore"
    param_list = {"ExecInfo": None}
    execInfo = None
    count = 0
    tile_count = 0
    energy_cycle_count = 0
    # leakage = 1.11305 per 2048 byte
    def __init__(self, step_size, param_list):
        super().__init__(step_size, param_list)
        self.count = 0
        self.tile_count = 0
        self.energy_cycle_count = 0
        self.cur_layer = 0
        self.layer_num = param_list["layer_num"]
        self.execInfo = param_list["ExecInfo"]
        self.l1_size = param_list['--l1_size']
        self.pe = param_list["--num_pe"]
        if self.execInfo == None:
            utils.debug_print("No ExecInfo!")
            return

    # step_params: out_power
    def step(self, params=None) -> float:
        power = (self.execInfo[self.cur_layer].power + self.l1_size*self.pe/2048 * 1.11305*1e-3) / self.step_size
        requirement = np.ceil(self.execInfo[self.cur_layer].latency * self.step_size)

        self.energy_cycle_count = self.energy_cycle_count + 1
        if self.energy_cycle_count >= requirement:
            self.tile_count = self.tile_count + 1
            self.energy_cycle_count = 0
            if self.tile_count >= self.execInfo[self.cur_layer].all_num:
                self.cur_layer = self.cur_layer + 1
                if self.cur_layer == self.layer_num:
                    self.count = self.count + 1
                    self.cur_layer = 0
                self.tile_count = 0
        return power

    def shut(self):
        self.energy_cycle_count = 0
