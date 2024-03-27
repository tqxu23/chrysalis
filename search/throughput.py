from models.DummyModel import DummyModel
from models.iNASModel import iNASModel
from models.GammaModel import GammaModel
import optuna
from models.components.insitu.INASCostCore.test_cnn_cost import Mat
from models.components.insitu.INASCostCore import test_cnn_cost


def max_throughput(trial):
    # cap_volume = trial.suggest_float('cap_volume', 2e-6, 2e-6)
    eh_area = trial.suggest_float('area', 0, 1e-4)
    # num_pe = trial.suggest_int('pe', 8)
    inter = trial.suggest_categorical('inter', [True, False])
    param_list = {"insitu_power": 1e-3, "cap_volume": 2e-6, "cap_voltage": 2.7,
                  "eh_area": eh_area, "eh_efficiency": 0.15, "eh_power": 100,
                  "eh_Solar": 50 / 0.15, '--num_pe': 8, '--l1_size': -1, '--l2_size': -1
        , '--inter': inter, '--num_pop': 10, '--epochs': 4, "strategy": "runTime"}
    model = GammaModel(10000, param_list)

    for i in range(200000):
        if not model.step():
            return -1
    trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
    trial.set_user_attr("frequency", str(model.insitu.execInfo.frequency))
    trial.set_user_attr("power", str(model.insitu.execInfo.power))
    trial.set_user_attr("num", str(model.insitu.execInfo.all_num))
    return model.throughput, eh_area


def getLists(num):
    out = []
    for i in range(num):
        if num % (i + 1) == 0:
            out.append(i + 1)
    return out


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

def get_layer_cifar():
    layer = []
    l1 = {}
    l1['name'] = "CONV1"
    l1['type'] = "CONV"
    l1['K'] = Mat(0, 8, 3, 5, 5)
    l1['OFM'] = Mat(0, 1, 8, 28, 28)
    l1['IFM'] = Mat(0, 1, 3, 32, 32)
    l1['stride'] = 1
    layer.append(l1)
    l2 = {}
    l2['name'] = "CONV2"
    l2['type'] = "CONV"
    l2['K'] = Mat(0, 8, 8, 3, 3)
    l2['OFM'] = Mat(0, 1, 8, 26, 26)
    l2['IFM'] = Mat(0, 1, 8, 28, 28)
    l2['stride'] = 1
    layer.append(l2)
    l3 = {}
    l3['name'] = "CONV3"
    l3['type'] = "CONV"
    l3['K'] = Mat(0, 16, 8, 7, 7)
    l3['OFM'] = Mat(0, 1, 16, 20, 20)
    l3['IFM'] = Mat(0, 1, 8, 26, 26)
    l3['stride'] = 1
    layer.append(l3)
    l4 = {}
    l4['name'] = "CONV4"
    l4['type'] = "CONV"
    l4['K'] = Mat(0, 8, 16, 5, 5)
    l4['OFM'] = Mat(0, 1, 8, 16, 16)
    l4['IFM'] = Mat(0, 1, 16, 20, 20)
    l4['stride'] = 1
    layer.append(l4)
    l5 = {}
    l5['name'] = "CONV5"
    l5['type'] = "CONV"
    l5['K'] = Mat(0, 24, 8, 7, 7)
    l5['OFM'] = Mat(0, 1, 24, 10, 10)
    l5['IFM'] = Mat(0, 1, 8, 16, 16)
    l5['stride'] = 1
    layer.append(l5)

    l6 = {}
    l6['name'] = "GAVGPOOL"
    l6['type'] = "GAVGPOOL"
    l6['K'] = Mat(0, 24, 24, 10, 10)
    l6['OFM'] = Mat(0, 1, 24, 1, 1)
    l6['IFM'] = Mat(0, 1, 24, 10, 10)
    l6['stride'] = 1
    layer.append(l6)

    l7 = {}
    l7['name'] = "FCEND"
    l7['type'] = "FC"
    l7['K'] = Mat(0, 10, 24, 1, 1)
    l7['OFM'] = Mat(0, 1, 10, 1, 1)
    l7['IFM'] = Mat(0, 1, 24, 1, 1)
    l7['stride'] = 1
    layer.append(l7)
    return layer


def get_layer_har():
    layer = []
    l1 = {}
    l1['name'] = "CONV1"
    l1['type'] = "CONV"
    l1['K'] = Mat(0, 16, 9, 1, 1)
    l1['OFM'] = Mat(0, 1, 16, 128, 1)
    l1['IFM'] = Mat(0, 1, 9, 128, 1)
    l1['stride'] = 1
    layer.append(l1)
    l2 = {}
    l2['name'] = "CONV2"
    l2['type'] = "CONV"
    l2['K'] = Mat(0, 16, 16, 3, 1)
    l2['OFM'] = Mat(0, 1, 16, 126, 1)
    l2['IFM'] = Mat(0, 1, 16, 128, 1)
    l2['stride'] = 1
    layer.append(l2)
    l3 = {}
    l3['name'] = "CONV3"
    l3['type'] = "CONV"
    l3['K'] = Mat(0, 8, 16, 5, 1)
    l3['OFM'] = Mat(0, 1, 8, 122, 1)
    l3['IFM'] = Mat(0, 1, 16, 126, 1)
    l3['stride'] = 1
    layer.append(l3)

    l6 = {}
    l6['name'] = "GAVGPOOL"
    l6['type'] = "GAVGPOOL"
    l6['K'] = Mat(0, 8, 8, 122, 1)
    l6['OFM'] = Mat(0, 1, 8, 1, 1)
    l6['IFM'] = Mat(0, 1, 8, 122, 1)
    l6['stride'] = 1
    layer.append(l6)

    l7 = {}
    l7['name'] = "FCEND"
    l7['type'] = "FC"
    l7['K'] = Mat(0, 6, 8, 1, 1)
    l7['OFM'] = Mat(0, 1, 6, 1, 1)
    l7['IFM'] = Mat(0, 1, 8, 1, 1)
    l7['stride'] = 1
    layer.append(l7)
    return layer


def get_layer_kws():
    layer = []
    l1 = {}
    l1['name'] = "FC1"
    l1['type'] = "FC"
    l1['K'] = Mat(0, 64, 1, 250, 1)
    l1['OFM'] = Mat(0, 1, 64, 1, 1)
    l1['IFM'] = Mat(0, 1, 1, 250, 1)
    l1['stride'] = 1
    layer.append(l1)

    l2 = {}
    l2['name'] = "FC2"
    l2['type'] = "FC"
    l2['K'] = Mat(0, 128, 64, 1, 1)
    l2['OFM'] = Mat(0, 1, 128, 1, 1)
    l2['IFM'] = Mat(0, 1, 64, 1, 1)
    l2['stride'] = 1
    layer.append(l2)

    l3 = {}
    l3['name'] = "FC3"
    l3['type'] = "FC"
    l3['K'] = Mat(0, 128, 128, 1, 1)
    l3['OFM'] = Mat(0, 1, 128, 1, 1)
    l3['IFM'] = Mat(0, 1, 128, 1, 1)
    l3['stride'] = 1
    layer.append(l3)


    l4 = {}
    l4['name'] = "FC3"
    l4['type'] = "FC"
    l4['K'] = Mat(0, 64, 128, 1, 1)
    l4['OFM'] = Mat(0, 1, 64, 1, 1)
    l4['IFM'] = Mat(0, 1, 128, 1, 1)
    l4['stride'] = 1
    layer.append(l4)

    l7 = {}
    l7['name'] = "FCEND"
    l7['type'] = "FC"
    l7['K'] = Mat(0, 12, 64, 1, 1)
    l7['OFM'] = Mat(0, 1, 12, 1, 1)
    l7['IFM'] = Mat(0, 1, 64, 1, 1)
    l7['stride'] = 1
    layer.append(l7)
    return layer


@static_vars(trial_optuna_count=0)
def max_throughput_iNAS(trial):
    strategy = "iNAS"
    eh_area = trial.suggest_float('area', 0, 30e-4)
    cap_size = trial.suggest_float('cap_size', 1e-5, 1e-2)
    layer = get_layer_cifar()
    layer_new = []
    for l in layer:
        study = optuna.create_study()
        def low_level_search(trial):
            maxTr = int(l['OFM'].h)
            maxTc = int(l['OFM'].w)
            maxTm = int(l['OFM'].ch)
            maxTn = int(l['IFM'].ch)
            Tr = trial.suggest_categorical('tr', getLists(maxTr))
            Tc = trial.suggest_categorical('tc', getLists(maxTc))
            Tm = trial.suggest_categorical('tm', getLists(maxTm))
            Tn = trial.suggest_categorical('tn', getLists(maxTn))
            l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
            param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                          "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": [l],
                          "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                          "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
            model = iNASModel(100, param_list)
            trial.set_user_attr("num", str(model.insitu.execInfo.all_num))
            for i in range(10000):
                if not model.step():
                    break
            return model.step_count
        study.optimize(low_level_search, n_trials=30)  # number of iterations

    # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
    # trial.set_user_attr("power", str(model.insitu.execInfo.power))
        Tr = study.best_params['tr']
        Tc = study.best_params['tc']
        Tm = study.best_params['tm']
        Tn = study.best_params['tn']
        l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
        layer_new.append(l)
    param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                  "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": layer_new,
                  "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
    model = iNASModel(100, param_list)
    for i in range(20000):
        if not model.step():
            break
    # Tr = maxTr
    # Tc = maxTc
    # Tm = maxTm/2
    # Tn = maxTn/2
    max_throughput_iNAS.trial_optuna_count = max_throughput_iNAS.trial_optuna_count + 1
    print(max_throughput_iNAS.trial_optuna_count, model.step_count, eh_area)
    return model.step_count, eh_area

@static_vars(trial_optuna_count=0)
def max_throughput_greedy(trial):
    strategy = "greedy"
    eh_area = trial.suggest_float('area', 0, 30e-4)
    cap_size = trial.suggest_float('cap_size', 1e-5, 1e-2)
    layer = get_layer_cifar()
    layer_new = []
    for l in layer:
        study = optuna.create_study()
        def low_level_search(trial):
            maxTr = int(l['OFM'].h)
            maxTc = int(l['OFM'].w)
            maxTm = int(l['OFM'].ch)
            maxTn = int(l['IFM'].ch)
            Tr = trial.suggest_categorical('tr', getLists(maxTr))
            Tc = trial.suggest_categorical('tc', getLists(maxTc))
            Tm = trial.suggest_categorical('tm', getLists(maxTm))
            Tn = trial.suggest_categorical('tn', getLists(maxTn))
            l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
            param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                          "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": [l],
                          "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                          "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
            model = iNASModel(100, param_list)
            trial.set_user_attr("num", str(model.insitu.execInfo.all_num))
            for i in range(10000):
                if not model.step():
                    break
            return model.step_count
        study.optimize(low_level_search, n_trials=30)  # number of iterations

    # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
    # trial.set_user_attr("power", str(model.insitu.execInfo.power))
        Tr = study.best_params['tr']
        Tc = study.best_params['tc']
        Tm = study.best_params['tm']
        Tn = study.best_params['tn']
        l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
        layer_new.append(l)
    param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                  "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": layer_new,
                  "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
    model = iNASModel(100, param_list)
    for i in range(20000):
        if not model.step():
            break
    # Tr = maxTr
    # Tc = maxTc
    # Tm = maxTm/2
    # Tn = maxTn/2
    max_throughput_iNAS.trial_optuna_count = max_throughput_iNAS.trial_optuna_count + 1
    print(max_throughput_iNAS.trial_optuna_count, model.step_count, eh_area)
    return model.step_count, eh_area

@static_vars(trial_optuna_count=0)
def max_throughput_runTime(trial):
    strategy = "runTime"
    eh_area = trial.suggest_float('area', 0, 30e-4)
    cap_size = trial.suggest_float('cap_size', 1e-5, 1e-2)
    layer = get_layer_cifar()
    layer_new = []
    for l in layer:
        study = optuna.create_study()
        def low_level_search(trial):
            maxTr = int(l['OFM'].h)
            maxTc = int(l['OFM'].w)
            maxTm = int(l['OFM'].ch)
            maxTn = int(l['IFM'].ch)
            Tr = trial.suggest_categorical('tr', getLists(maxTr))
            Tc = trial.suggest_categorical('tc', getLists(maxTc))
            Tm = trial.suggest_categorical('tm', getLists(maxTm))
            Tn = trial.suggest_categorical('tn', getLists(maxTn))
            l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
            param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                          "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": [l],
                          "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                          "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
            model = iNASModel(100, param_list)
            trial.set_user_attr("num", str(model.insitu.execInfo.all_num))
            for i in range(10000):
                if not model.step():
                    break
            return model.step_count
        study.optimize(low_level_search, n_trials=30)  # number of iterations

    # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
    # trial.set_user_attr("power", str(model.insitu.execInfo.power))
        Tr = study.best_params['tr']
        Tc = study.best_params['tc']
        Tm = study.best_params['tm']
        Tn = study.best_params['tn']
        l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
        layer_new.append(l)
    param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                  "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": layer_new,
                  "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
    model = iNASModel(100, param_list)
    for i in range(20000):
        if not model.step():
            break
    # Tr = maxTr
    # Tc = maxTc
    # Tm = maxTm/2
    # Tn = maxTn/2
    max_throughput_iNAS.trial_optuna_count = max_throughput_iNAS.trial_optuna_count + 1
    print(max_throughput_iNAS.trial_optuna_count, model.step_count, eh_area)
    return model.step_count, eh_area

@static_vars(trial_optuna_count=0)
def max_throughput_default(trial):
    strategy = "default"
    eh_area = trial.suggest_float('area', 0, 30e-4)
    cap_size = trial.suggest_float('cap_size', 1e-5, 1e-2)
    layer = get_layer_cifar()
    layer_new = []
    for l in layer:
        study = optuna.create_study()
        def low_level_search(trial):
            maxTr = int(l['OFM'].h)
            maxTc = int(l['OFM'].w)
            maxTm = int(l['OFM'].ch)
            maxTn = int(l['IFM'].ch)
            Tr = trial.suggest_categorical('tr', getLists(maxTr))
            Tc = trial.suggest_categorical('tc', getLists(maxTc))
            Tm = trial.suggest_categorical('tm', getLists(maxTm))
            Tn = trial.suggest_categorical('tn', getLists(maxTn))
            l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
            param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                          "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": [l],
                          "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                          "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
            model = iNASModel(100, param_list)
            trial.set_user_attr("num", str(model.insitu.execInfo.all_num))
            for i in range(10000):
                if not model.step():
                    break
            return model.step_count
        study.optimize(low_level_search, n_trials=30)  # number of iterations

    # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
    # trial.set_user_attr("power", str(model.insitu.execInfo.power))
        Tr = study.best_params['tr']
        Tc = study.best_params['tc']
        Tm = study.best_params['tm']
        Tn = study.best_params['tn']
        l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
        layer_new.append(l)
    param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                  "eh_area": eh_area, "eh_efficiency": 0.9, "eh_power": 1, "layer": layer_new,
                  "eh_Solar": 64.361, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": strategy, "mode": "latency"}
    model = iNASModel(100, param_list)
    for i in range(20000):
        if not model.step():
            break
    # Tr = maxTr
    # Tc = maxTc
    # Tm = maxTm/2
    # Tn = maxTn/2
    max_throughput_iNAS.trial_optuna_count = max_throughput_iNAS.trial_optuna_count + 1
    print(max_throughput_iNAS.trial_optuna_count, model.step_count, eh_area)
    return model.step_count, eh_area
