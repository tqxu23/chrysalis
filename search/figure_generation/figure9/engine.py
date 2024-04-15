import sys

sys.path.append("../../../")
import yaml

import numpy as np
import optuna
import pandas as pd
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
import plotly.express as px


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

def get_single_conv():
    layer = []
    l1 = {}
    l1['name'] = "CONV1"
    l1['type'] = "CONV"
    l1['K'] = Mat(0, 8, 3, 5, 5)
    l1['OFM'] = Mat(0, 1, 8, 28, 28)
    l1['IFM'] = Mat(0, 1, 3, 32, 32)
    l1['stride'] = 1
    layer.append(l1)
    return layer

def getLists(num):
    out = []
    for i in range(num):
        if num % (i + 1) == 0:
            out.append(i + 1)
    return out


def drawImg(study,name,step):
    df = study.trials_dataframe()
    df["ckpt_energy"] = df.user_attrs_Er.astype(float) + df.user_attrs_Ew.astype(float) + df.user_attrs_Eb.astype(float)
    df.values_1 = df.values_1 * 10000
    df.values_0 = df.values_0 / step
    df["lat*sp"] = df.values_1 * df.values_0
    df["system_efficiency"] = df.user_attrs_Ec.astype(float)/(step*df["lat*sp"]*0.0001*30*0.15)
    df["cap_leakage"] = df.params_cap_size.astype(float)*df.values_0.astype(float)*9*0.001
    df = df[(df.values_0 > 0.2)]
    df = df[(df.values_0 < 100)]
    df.rename(columns={"values_0": "Latency (s)", "values_1": "SP size(cm²)"}, inplace=True)
    fig = px.scatter(df,
                     x="Latency (s)",
                     y="SP size(cm²)",
                     color='lat*sp',
                     color_continuous_scale='blugrn',
                     height=600, width=800,
                     hover_data=['params_cap_size',"ckpt_energy","system_efficiency","cap_leakage"]
                     )
    fig.update_layout(font=dict(family='Arial'), font_color='black', font_size=26)
    fig.write_html(f"./{name}.html")
    # plotly.offline.plot(fig)


def engine(study, name, layer, solar1, solar2,area1,area2,cap1,cap2,step,num):
    for i in range(num):
        print(f"Current Epoch: {i}")

        @static_vars(trial_optuna_count=0)
        def max_throughput(trial):
            eh_area = trial.suggest_float('area', area1, area2)
            cap_size = trial.suggest_float('cap_size', cap1, cap2)

            layer_new = []
            Eb = 0
            Lb = 0
            Er = 0
            Lr = 0
            Ec = 0
            Lc = 0
            Ew = 0
            Lw = 0
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
                                  "eh_area": eh_area, "eh_efficiency": 0.5, "eh_power": solar1, "layer": [l],
                                  "ExecInfo": test_cnn_cost.get_conv_cost,
                                  "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": "runtime", "mode": "latency"}
                    model = iNASModel(step, param_list)
                    
                    trial.set_user_attr("num", str(model.insitu.execInfo.all_num))
                    errFlag = False
                    for i in range(5000):
                        con, error = model.step()
                        if error:
                            errFlag = True
                            break
                        if not con:
                            break
                    count1 = model.step_count
                    if errFlag:
                        return 5000

                    param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                                  "eh_area": eh_area, "eh_efficiency": 0.5, "eh_power": solar2, "layer": [l],
                                  "ExecInfo": test_cnn_cost.get_conv_cost,
                                  "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": "runtime", "mode": "latency"}
                    model = iNASModel(step, param_list)
                    trial.set_user_attr("Eb", (model.insitu.execInfo.Eb))
                    trial.set_user_attr("Lb", (model.insitu.execInfo.Lb))
                    trial.set_user_attr("Er", (model.insitu.execInfo.Er))
                    trial.set_user_attr("Lr", (model.insitu.execInfo.Lr))
                    trial.set_user_attr("Ec", (model.insitu.execInfo.Ec))
                    trial.set_user_attr("Lc", (model.insitu.execInfo.Lc))
                    trial.set_user_attr("Ew", (model.insitu.execInfo.Ew))
                    trial.set_user_attr("Lw", (model.insitu.execInfo.Lw))
                    errFlag = False
                    for i in range(5000):
                        con, error = model.step()
                        if error:
                            errFlag = True
                            break
                        if not con:
                            break
                    count2 = model.step_count
                    if errFlag:
                        return 5000
                    return (count1 + count2) / 2

                study.optimize(low_level_search, n_trials=10)  # number of iterations

                # trial.set_user_attr("latency", str(model.insitu.execInfo.latency))
                # trial.set_user_attr("power", str(model.insitu.execInfo.power))
                Tr = study.best_params['tr']
                Tc = study.best_params['tc']
                Tm = study.best_params['tm']
                Tn = study.best_params['tn']
                ua = study.best_trial.user_attrs
                Eb = Eb + (ua['Eb'])
                Lb = Lb + (ua['Lb'])
                Er = Er + (ua['Er'])
                Lr = Lr + (ua['Lr'])
                Ec = Ec + (ua['Ec'])
                Lc = Lc + (ua['Lc'])
                Ew = Ew + (ua['Ew'])
                Lw = Lw + (ua['Lw'])
                l['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
                layer_new.append(l)
            param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                          "eh_area": eh_area, "eh_efficiency": 0.5, "eh_power": solar1, "layer": layer_new,
                          "ExecInfo": test_cnn_cost.get_conv_cost,
                          "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": "runtime", "mode": "latency"}
            model = iNASModel(step, param_list)
            errFlag = False
            for i in range(10000):
                con, error = model.step()
                if error:
                    print("cap error")
                    errFlag = True
                    break
                if not con:
                    break
            count1 = model.step_count
            if errFlag:
                return 10000
            param_list = {"insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 0,
                          "eh_area": eh_area, "eh_efficiency": 0.5, "eh_power": solar2, "layer": layer_new,
                          "ExecInfo": test_cnn_cost.get_conv_cost,
                          "v_on": 3.0, "v_off": 2.8, "v_max": 4.1, "strategy": "runtime", "mode": "latency"}
            model = iNASModel(step, param_list)
            errFlag = False
            for i in range(10000):
                con, error = model.step()
                if error:
                    print("cap error")
                    errFlag = True
                    break
                if not con:
                    break
            count2 = model.step_count
            if errFlag:
                return 10000
            trial.set_user_attr("Eb", str(Eb))
            trial.set_user_attr("Lb", str(Lb))
            trial.set_user_attr("Er", str(Er))
            trial.set_user_attr("Lr", str(Lr))
            trial.set_user_attr("Ec", str(Ec))
            trial.set_user_attr("Lc", str(Lc))
            trial.set_user_attr("Ew", str(Ew))
            trial.set_user_attr("Lw", str(Lw))
            max_throughput.trial_optuna_count = max_throughput.trial_optuna_count + 1
            print(max_throughput.trial_optuna_count, (count1 + count2) / 2, eh_area)
            return (count1 + count2) / 2, eh_area

        study.optimize(max_throughput, n_trials=10, n_jobs=1)
        drawImg(study,name, step)
    return study
        # fig = optuna.visualization.plot_pareto_front(study)
        # # plotly.offline.plot(fig)
        # fig.write_image("./result.png", scale=3)


def getRes(name, layers,area1,area2,cap1,cap2,envir1, envir2, step,num,mode="sp*lat"):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"{name}_sp_{area1}_{area2}_cap_{cap1}_{cap2}_envir_{envir1}_{envir2}_step_{step}"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    # study = optuna.create_study(directions=["minimize", "minimize"], study_name=study_name,
    #                             load_if_exists=True)
    study = optuna.create_study(directions=["minimize", "minimize"], study_name=study_name,
                                storage=storage_name, load_if_exists=True)
    # study = engine(study, study_name, layers, envir1, envir2,area1,area2,cap1,cap2,step,num)
    df = study.trials_dataframe()
    if mode=="lat":
        return df[df["values_0"] == df["values_0"].min()].iloc[0]
    elif mode=="sp":
        return df[df["values_1"] == df["values_1"].min()].iloc[0]
    elif mode=="sp*lat" or mode=="lat*sp":
        df["lat*sp"] = df.values_1 * df.values_0
        return df[df["lat*sp"] == df["lat*sp"].min()].iloc[0]
    return None


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def drawBar1(df,name):
    fig = px.bar(df,
                     x="params_cap_size",
                     y="cap_leakage",
                     color='lat*sp',
                     color_continuous_scale='blugrn',
                     height=600, width=800,
                     hover_data=['Latency (s)',"ckpt_energy","system_efficiency","SP size(cm²)"]
                     )
    fig.update_layout(font=dict(family='Arial'), font_color='black', font_size=26)
    fig.write_html(f"./{name}.html")
    # plotly.offline.plot(fig)

def drawBar2(df,name):
    fig = px.bar(df,
                     x="params_cap_size",
                     y="ckpt_energy",
                     color='lat*sp',
                     color_continuous_scale='blugrn',
                     height=600, width=800,
                     hover_data=['Latency (s)',"cap_leakage","system_efficiency","SP size(cm²)"]
                     )
    fig.update_layout(font=dict(family='Arial'), font_color='black', font_size=26)
    fig.write_html(f"./{name}.html")
    # plotly.offline.plot(fig)

if __name__ == '__main__':

    yaml_name = sys.argv[1]
    data = read_yaml(yaml_name)
    print(data) 
    networks = {
        "cifar": get_layer_cifar(),
        "simple": get_single_conv(),
        "kws": get_layer_kws(),
        "har": get_layer_har()}
    layer = networks[data['network']]
    cap_size = data['cap']['from']
    df = pd.DataFrame()
    while True:
        if cap_size>data['cap']['to']:
            break
        ret = getRes(data['network'], layer,data['sp']['from'],data['sp']['to'],cap_size,cap_size,data['solar']['min'],data['solar']['max'],data['step'],data['epoch'])
        df = df.append(ret)
        cap_size = cap_size * data['cap']['mutiple_step']
    print(df)

    df["ckpt_energy"] = df.user_attrs_Er.astype(float) + df.user_attrs_Ew.astype(float) + df.user_attrs_Eb.astype(float)
    df.values_1 = df.values_1 * 10000
    df.values_0 = df.values_0 / data['step']
    df["lat*sp"] = df.values_1 * df.values_0
    df["system_efficiency"] = df.user_attrs_Ec.astype(float)/(data['step']*df["lat*sp"]*0.0001*30*0.15)

    print(df.params_cap_size)
    df["cap_leakage"] = df.params_cap_size.astype(float)*df.values_0.astype(float)*9*0.001
    df.params_cap_size = df.params_cap_size.astype(str)
    df = df[(df.values_0 > 0.2)]
    df = df[(df.values_0 < 100)]
    df.rename(columns={"values_0": "Latency (s)", "values_1": "SP size(cm²)"}, inplace=True)
    print(df)
    print(df.columns)
    drawBar1(df,"figure9-example-1")
    drawBar2(df,"figure9-example-2")

#     getDf("example-study0719-runTime-cap")



