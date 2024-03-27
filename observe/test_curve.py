from models.DummyModel import DummyModel
from models.iNASModel import iNASModel
from models.GammaModel import GammaModel
from models.components.insitu.INASCostCore.test_cnn_cost import Mat

import plotly
import optuna
import search
import models
# 折线图
import plotly.express as px
import pandas as pd
import models.components.insitu.INASCostCore as INASCostCore

from models.iNASModel import iNASModel
from models.GammaModel import GammaModel

from models.components.insitu.INASCostCore.test_cnn_cost import Mat
from models.components.insitu.INASCostCore import test_cnn_cost
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

if __name__ == '__main__':
    layers = get_layer_cifar()
    eh_area = 30e-4
    new_layers = []
    for layer in layers:
        maxTr = int(layer['OFM'].h)
        maxTc = int(layer['OFM'].w)
        maxTm = int(layer['OFM'].ch)
        maxTn = int(layer['IFM'].ch)
        Tr = maxTr/2
        Tc = maxTc/2
        Tm = maxTm/2
        Tn = maxTn/2
        layer['tile_size'] = [1, 1, 1, 1, Tr, Tc, Tm, Tn]
        new_layers.append(layer)
    param_list1 = {"insitu_power": 1e-3, "cap_volume": 1e-3, "cap_voltage": 2.8,
                  "eh_area": eh_area, "eh_efficiency": 0.85, "layer": new_layers,
                  "eh_Solar": 40, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.1, "v_off": 2.8, "v_max": 4.1, "strategy": "runTime", "mode": "throughput"}
    model1 = iNASModel(100, param_list1)

    param_list2 = {"insitu_power": 1e-3, "cap_volume": 1e-3, "cap_voltage": 2.8,
                  "eh_area": eh_area, "eh_efficiency": 0.85, "layer": new_layers,
                  "eh_Solar": 80, "ExecInfo": test_cnn_cost.get_conv_cost,
                  "v_on": 3.1, "v_off": 2.8, "v_max": 4.1, "strategy": "runTime", "mode": "throughput"}
    model2 = iNASModel(100, param_list2)


    rows = []
    x = []
    cate = []
    for i in range(10000):
        model1.step()


        rows.append(model1.eh.cap.voltage)
        x.append(i/100)
        cate.append("runtime-vol-40")


        model2.step()
        rows.append(model2.eh.cap.voltage)
        x.append(i/100)
        cate.append("runtime-vol-80")

    # cap_volume = 1e-3
    # eh_area = 1e-5
    # param_list = {"cap_volume": cap_volume, "cap_voltage": 2.7,
    #               "eh_area": eh_area, "eh_efficiency": 0.15, "eh_power": 100,
    #               "eh_Solar": 100 / 0.15, '--num_pe': 8, '--l1_size': -1, '--l2_size': -1
    #     , '--inter': False, '--num_pop': 10, '--epochs': 4}
    # model1 = GammaModel(10000, param_list)
    #
    # layer = {}
    # layer['type'] = "CONV"
    # layer['K'] = Mat(0, 4, 3, 5, 5)
    # layer['OFM'] = Mat(0, 1, 4, 28, 28)
    # layer['IFM'] = Mat(0, 1, 3, 32, 32)
    # maxTr = layer['OFM'].h
    # maxTc = layer['OFM'].w
    # maxTm = layer['OFM'].ch
    # maxTn = layer['IFM'].ch
    # layer['stride'] = 1
    # param_list = {"insitu_power": 1e-3, "cap_volume": 1e-3, "cap_voltage": 0,
    #               "eh_area": 1e-4, "eh_efficiency": 0.15, "eh_power": 100, "layer": layer,
    #               "eh_Solar": 100 / 0.15, "ExecInfo": INASCostCore.test_cnn_cost.get_conv_cost,
    #               "v_on": 3.1, "v_off": 2.8, "v_max": 4.1, "strategy": "iNAS"}
    # model2 = iNASModel(10000, param_list)
    # rows = []
    # x = []
    # cate = []
    # for i in range(100000):
    #     model1.step()
    #     model2.step()
    #     model1.step()
    #     model2.step()
    #     model1.step()
    #     model2.step()
    #
    #     rows.append(model1.eh.cap.voltage)
    #     x.append(i/10000)
    #     cate.append("gamma-vol")
    #     # rows.append(model2.eh.cap.voltage)
    #     rows.append(model2.eh.cap.voltage)
    #     x.append(i/10000)
    #     cate.append("inas-vol")

    df = pd.DataFrame(pd.DataFrame(dict(x=x, y=rows, cate=cate)))
    print(df)
    fig = px.line(df, x="x", y="y", title='Voltage Curve', color='cate')
    plotly.offline.plot(fig)
