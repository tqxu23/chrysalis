import utils
class Insitu:
    name = "Insitu-base"
    param_list = {}
    step_size = 100000
    def __init__(self, step_size, param_list=None):
        self.step_size = step_size
        if param_list is not None:
            self.param_list = self.check_params(param_list)
        utils.debug_print("Get new " + self.name)

    def step(self, step_params):
        self.refresh_params(step_params)
        utils.debug_print("To be implemented...")
        return 0

    def check_params(self, param_list) -> tuple:
        for name in self.param_list.keys():
            if name in param_list:
                continue
            else:
                utils.debug_print("param " + name + " not assigned! Use default ones")
                param_list[name] = self.param_list[name]
        return param_list

    def refresh_params(self, harvest_params):
        for name in harvest_params:
            self.param_list[name] = harvest_params[name]

    def shut(self):
        pass