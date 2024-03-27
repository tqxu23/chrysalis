import utils
class Component:
    name = "Component-base"
    param_list = {}
    def __init__(self, param_list):
        self.param_list = self.check_params(param_list)
        utils.debug_print("Get new " + self.name)

    def step(self, step_params):
        self.refresh_params(step_params)
        utils.debug_print("To be implemented...")
        return 0

    def check_params(self, param_list) -> tuple:
        for name in self.param_list.keys():
            if param_list.haskey(name):
                continue
            else:
                utils.debug_print("param " + name + " not assigned! Use default ones")
                param_list[name] = self.param_list[name]
        return param_list

    def refresh_params(self, harvest_params):
        for name in harvest_params:
            self.param_list[name] = harvest_params[name]


class SearchSpace:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.max_value = max_value
        self.min_value = min_value
