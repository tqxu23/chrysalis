import sys

sys.path.append("../")
import yaml
import optuna
import search
import models
import plotly
import sys
import logging
from models.DummyModel import DummyModel
from models.iNASModel import iNASModel
from models.GammaModel import GammaModel
import optuna
from models.components.insitu.INASCostCore.test_cnn_cost import Mat
from models.components.insitu.INASCostCore import test_cnn_cost
from models.components.insitu.GammaCostCore.ExecInfo import ExecInfo
import plotly.express as px


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def getLists(num):
    out = []
    for i in range(num):
        if num % (i + 1) == 0:
            out.append(i + 1)
    return out


def engineGamma(study, layername, solar1, solar2, layer_num):
    for i in range(1000):
        print(i)
        @static_vars(trial_optuna_count=0)
        def max_throughput(trial):
            pe_num = trial.suggest_int("pe_num", 1, 64)
            l1_size = trial.suggest_int("l1_size", 1024, 8192)
            l2_size = trial.suggest_int("l2_size", 108000, 108000)
            eh_area = trial.suggest_float('area', 0, 2e-3)
            cap_size = trial.suggest_float('cap_size', 1e-5, 1e-1)
            execInfo = []
            for l in range(layer_num):
                study = optuna.create_study()

                def low_level_search(trial):
                    param_list = {"mode": "network", "insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 2.79,
                                  "eh_area": eh_area, "eh_efficiency": 0.65, "eh_power": solar1,
                                  '--num_pe': pe_num, '--l1_size': l1_size, '--l2_size': l2_size, '--inter': True,
                                  "--num_pop": 5, "--epochs": 1, "v_on": 3.1, "v_off": 2.8, "v_max": 4.1,
                                  "--model": layername,
                                  "--num_layer": 1, "--singlelayer": l+1, "layer_num": 1}
                    model = GammaModel(500, param_list)
                    trial.set_user_attr("power", model.execInfo[0].power)
                    trial.set_user_attr("latency", model.execInfo[0].latency)
                    trial.set_user_attr("all_num", model.execInfo[0].all_num)
                    # print(model.mapping)
                    for i in range(50000):
                        if not model.step():
                            break
                    count1 = model.step_count
                    return count1

                study.optimize(low_level_search, n_trials=30)  # number of iterations

                # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
                # trial.set_user_attr("power", str(model.insitu.execInfo.power))
                power = study.best_trial.user_attrs['power']
                latency = study.best_trial.user_attrs['latency']
                all_num = study.best_trial.user_attrs['all_num']
                print(l, power, latency, all_num)
                execInfo.append(ExecInfo())
                execInfo[l].power = power
                execInfo[l].latency = latency
                execInfo[l].all_num = all_num
                execInfo[l].frequency = 3e7
            param_list = {"mode": "network", "insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 2.79,
                          "eh_area": eh_area, "eh_efficiency": 0.65, "eh_power": solar1,
                          '--num_pe': pe_num, '--l1_size': l1_size, '--l2_size': l2_size, '--inter': True,
                          "--num_pop": 5, "--epochs": 2, "v_on": 3.1, "v_off": 2.8, "v_max": 4.1, "--model": layername,
                          "--num_layer": 1, "--singlelayer": 1, "layer_num": layer_num, "ExecInfo": execInfo}
            model = GammaModel(500, param_list)
            # try:
            #
            # except:
            #     print("error20001")
            #     return 20001, 1e-2
            for i in range(50000):
                if not model.step():
                    break
            count1 = model.step_count
            # param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 2.79,
            #               "eh_area": 1e-4, "eh_efficiency": 0.65, "eh_power": solar2,
            #               '--num_pe': pe_num, '--l1_size': l1_size, '--l2_size': l2_size, '--inter': True,
            #               "--num_pop": 5, "--epochs": 3, "v_on": 3.1, "v_off": 2.8, "v_max": 4.1,
            #               "--model": "resnet18",
            #               "--num_layer": 1, "--singlelayer": 1, "layer_num": 20}
            # model = GammaModel(500, param_list)
            # for i in range(100000):
            #     if not model.step():
            #         break
            # count2 = model.step_count
            count2 = count1
            max_throughput.trial_optuna_count = max_throughput.trial_optuna_count + 1
            print("end", max_throughput.trial_optuna_count, (count1 + count2) / 2, eh_area, pe_num, l1_size, l2_size,
                  cap_size)
            return (count1 + count2) / 2, eh_area

        study.optimize(max_throughput, n_trials=10, n_jobs=1)
        fig = optuna.visualization.plot_pareto_front(study)
        plotly.offline.plot(fig)


# def getRes(name, layer_num):
#     optuna.logging.set_verbosity(optuna.logging.WARNING)
#     #     optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
#     study_name = "example-gamma0728-" + name + "-bilevel-test"  # Unique identifier of the study.
#     storage_name = "sqlite:///{}.db".format(study_name)
#     study = optuna.create_study(directions=["minimize", "minimize"], study_name=study_name,
#                                 storage=storage_name, load_if_exists=True)
#     engineGamma(study, name, 20, 60, layer_num)


def getRes(name,arch, solar1, solar2, pe_num1,pe_num2,cache1,cache2, area1,area2,cap1,cap2,step,layer_num,num):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    #     optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{name}_{arch}_sp_{area1}_{area2}_cap_{cap1}_{cap2}_envir_{solar1}_{solar2}_pe_{pe_num1}_{pe_num2}_cache_{cache1}_{cache2}_step_{step}"
        
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=["minimize", "minimize"], study_name=study_name,
                                storage=storage_name, load_if_exists=True)
    engineGammaWithEyerissSettings(study,study_name,arch, name, solar1, solar2, pe_num1,pe_num2,cache1,cache2, area1,area2,cap1,cap2,step,layer_num,num)

def drawImg(study,name,step):
    df = study.trials_dataframe()
    # print(df.columns)
    df.values_1 = df.values_1 * 10000
    df.values_0 = df.values_0 / step
    df["lat*sp"] = df.values_1 * df.values_0
    df = df[(df.values_0 > 0.2  )]
    df = df[(df.values_0 < 100)]
    df.rename(columns={"values_0": "Latency (s)", "values_1": "SP size(cm²)"}, inplace=True)
    fig = px.scatter(df,
                     x="Latency (s)",
                     y="SP size(cm²)",
                     color='lat*sp',
                     color_continuous_scale='blugrn',
                     height=600, width=800, hover_data=['params_cap_size','params_pe_num','params_l1_size']
                     )
    fig.update_layout(font=dict(family='Arial'), font_color='black', font_size=26)
    fig.write_html(f"./{name}.html")
    # plotly.offline.plot(fig)

def engineGammaWithEyerissSettings(study,name,arch, layername, solar1, solar2, pe_num1,pe_num2,cache1,cache2, area1,area2,cap1,cap2,step,layer_num,num):
    for i in range(num):
        print(i)
        @static_vars(trial_optuna_count=0)
        def max_throughput(trial):
            pe_num = trial.suggest_int("pe_num", pe_num1,pe_num2)#1,64
            l1_size = trial.suggest_int("l1_size", cache1,cache2)#1024,1024*16
            l2_size = 108000
            eh_area = trial.suggest_float('area', area1,area2)# 1e-4, 30e-4
            cap_size = trial.suggest_float('cap_size', cap1, cap2)#1e-5, 1e-2
            execInfo = []
            for l in range(layer_num):
                study = optuna.create_study()

                def low_level_search(trial):
                    param_list = {"mode": "layer", "insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 2.79,
                                  "eh_area": eh_area, "eh_efficiency": 0.5, "eh_power": solar1,
                                  '--num_pe': pe_num, '--l1_size': l1_size, '--l2_size': l2_size, '--inter': True,
                                  "--num_pop": 10, "--epochs": 2, "v_on": 3.0, "v_off": 2.8, "v_max": 4.1,
                                  "--model": layername,'--arch': arch,
                                  "--num_layer": 1, "--singlelayer": l+1, "layer_num": 1}
                    model = GammaModel(step, param_list)
                    trial.set_user_attr("power", model.execInfo[0].power)
                    trial.set_user_attr("latency", model.execInfo[0].latency)
                    trial.set_user_attr("all_num", model.execInfo[0].all_num)
                    # print(model.mapping)
                    for i in range(50000):
                        if not model.step():
                            break
                    count1 = model.step_count
                    return count1

                study.optimize(low_level_search, n_trials=5)  # number of iterations

                # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
                # trial.set_user_attr("power", str(model.insitu.execInfo.power))
                power = study.best_trial.user_attrs['power']
                latency = study.best_trial.user_attrs['latency']
                all_num = study.best_trial.user_attrs['all_num']
                print(l, power, latency, all_num)
                execInfo.append(ExecInfo())
                execInfo[l].power = power
                execInfo[l].latency = latency
                execInfo[l].all_num = all_num
                execInfo[l].frequency = 3e7
            # print("layer finished")
            param_list = {"mode": "network", "insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 2.79,
                          "eh_area": eh_area, "eh_efficiency": 0.5, "eh_power": solar1,
                          '--num_pe': pe_num, '--l1_size': l1_size, '--l2_size': l2_size, '--inter': True,
                          "--num_pop": 20, "--epochs": 5, "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "--model": layername,'--arch': arch,
                          "--num_layer": 1, "--singlelayer": 1, "layer_num": layer_num, "ExecInfo": execInfo}
            model = GammaModel(step, param_list)


            for i in range(50000):
                if not model.step():
                    break
            count1 = model.step_count
            count2 = count1
            max_throughput.trial_optuna_count = max_throughput.trial_optuna_count + 1
            print("end", max_throughput.trial_optuna_count, (count1 + count2) / 2, eh_area, pe_num, l1_size, l2_size,
                  cap_size)
            return (count1 + count2) / 2, eh_area

        study.optimize(max_throughput, n_trials=10, n_jobs=1)
        # fig = optuna.visualization.plot_pareto_front(study)
        # plotly.offline.plot(fig)
        drawImg(study,name,step)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == '__main__':

    yaml_name = sys.argv[1]
    data = read_yaml(yaml_name)
    print(data)  # {'age': 45, 'name': 'zhangsan'}
    network_layer_num = {
        "alexnet": 5,
        "vgg16": 13,
        "BERT_m": 6,
        "resnet18": 20}
    layer_num = network_layer_num[data['network']]
    # "iNAS","greedy","runTime","default"
    # getResWithEyerissSettings("alexnet", 1)
    # getResWithEyerissSettings("vgg16", 1)
    # getResWithEyerissSettings("BERT_m", 1)
    #                           name, solar1, solar2, pe_num1,pe_num2,cache1,cache2, area1,area2,cap1,cap2,step,layer_num,num
    getRes(data['network'], data['accelerator']['arch'], data['solar']['min'], data['solar']['max'], 
            data['accelerator']['pe']['from'], data['accelerator']['pe']['to'], data['accelerator']['cache']['from'], data['accelerator']['cache']['to'], data['sp']['from'],data['sp']['to'],data['cap']['from'],data['cap']['to'],data['step'],layer_num,data['epoch'])
    # resnet18 20
    # BERT_m 6
    # vgg16 13
#     getDf("example-study0719-runTime-cap")
