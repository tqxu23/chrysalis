from models.components.insitu.Insitu import Insitu
from models.components.insitu.INASCostCore.ExecInfo import ExecInfo
import numpy as np
import utils


class INASInsitu(Insitu):
    name = "INASCostCore"
    param_list = {"ExecInfo": None}
    execInfo = None
    count = 0
    tile_count = 0
    energy_cycle_count = 0
    cur_state = 1  # 1 first cycle, 2 next cycles
    tile_per_energy_cycle = 0

    def __init__(self, step_size, param_list):
        super().__init__(step_size, param_list)
        self.strategy = param_list["strategy"]
        self.count = 0
        self.tile_count = 0
        self.current_ec_tile_count = 0
        self.energy_cycle_count = 0
        self.cur_state = 0
        self.layer = param_list["layer"]
        self.current_layer = 0
        self.layer_count = len(self.layer)
        self.execInfoFunc = param_list["ExecInfo"]
        self.execInfo = param_list["ExecInfo"](self.layer[0])
        self.param_list = param_list
        self.tile_per_energy_cycle = np.floor(0.5 * param_list["cap_volume"] \
                                              * (param_list["v_on"] * param_list["v_on"] - param_list["v_off"] *
                                                 param_list["v_off"]) \
                                              / self.execInfo.first_n_power/self.execInfo.first_n_latency)
        if self.execInfo == None:
            utils.debug_print("No ExecInfo!")
            return

    # step_params: out_power
    def step(self, params=None) -> [bool,float]:

        if self.cur_state == 1:
            power = self.execInfo.first_n_power / self.step_size
            requirement = np.ceil(self.execInfo.first_n_latency * self.step_size)
        else:
            power = self.execInfo.next_n_power / self.step_size
            requirement = np.ceil(self.execInfo.next_n_latency * self.step_size)

        self.energy_cycle_count = self.energy_cycle_count + 1
        if self.energy_cycle_count >= requirement:
            self.energy_cycle_count = 0
            self.cur_state = 2
            self.tile_count = self.tile_count + 1
            self.current_ec_tile_count = self.current_ec_tile_count + 1
            # print(self.tile_count, self.execInfo.all_num)
            if self.tile_count >= self.execInfo.all_num:
                self.current_layer = self.current_layer+1
                if self.current_layer == self.layer_count:
                    self.current_layer = 0
                    self.count = self.count+1
                self.execInfo = self.execInfoFunc(self.layer[self.current_layer])
                self.tile_per_energy_cycle = np.floor(0.5 * self.param_list["cap_volume"] \
                                                      * (self.param_list["v_on"] * self.param_list["v_on"] - self.param_list["v_off"] *
                                                         self.param_list["v_off"]) \
                                                      / self.execInfo.first_n_power / self.execInfo.first_n_latency)
                self.tile_count = 0
            if self.strategy == "iNAS" and self.current_ec_tile_count >= self.tile_per_energy_cycle:
                self.current_ec_tile_count = 0
                return False, power
            if self.strategy == "runTime":
                curEnergy = 0.5 * params["capVolume"] * (params["curVoltage"] * params["curVoltage"] - params["vOff"] * params["vOff"])
                # print(params["curVoltage"],params["vOff"],params["capVolume"],self.step_size)
                # print(curEnergy, self.execInfo.next_n_power*self.execInfo.next_n_latency - self.execInfo.next_n_latency * params["inPower"]*self.step_size)
                # print(curEnergy, self.execInfo.next_n_power)
                if curEnergy < self.execInfo.next_n_power*self.execInfo.next_n_latency - self.execInfo.next_n_latency * params["inPower"] * self.step_size:
                    return False, power
            if self.strategy == "default":
                return False, power

        return True, power

    def shut(self):
        self.cur_state = 1
        self.current_ec_tile_count = 0
        self.energy_cycle_count = 0
