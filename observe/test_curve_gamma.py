import plotly
# 折线图
import plotly.express as px
import pandas as pd
import models.components.insitu.INASCostCore as INASCostCore

from models.iNASModel import iNASModel
from models.GammaModel import GammaModel
# resnet18 20
if __name__ == '__main__':
    eh_area = 1e-3
    pe_num = 16
    l1_size = 2048
    l2_size = 108000
    solar1 = 20
    cap_size = 1e-2
    layername = "alexnet"
    l = 0
    param_list = {"mode": "layer", "insitu_power": 1e-3, "cap_volume": cap_size, "cap_voltage": 2.79,
                  "eh_area": eh_area, "eh_efficiency": 0.65, "eh_power": solar1,
                  '--num_pe': pe_num, '--l1_size': l1_size, '--l2_size': l2_size, '--inter': True,
                  "--num_pop": 5, "--epochs": 1, "v_on": 3.1, "v_off": 2.8, "v_max": 4.1,
                  "--model": layername,
                  "--num_layer": 1, "--singlelayer": l + 1, "layer_num": 1}
    model1 = GammaModel(1000, param_list)
    rows = []
    x = []
    cate = []
    for i in range(50000):
        model1.step()
        rows.append(model1.eh.cap.voltage)
        x.append(i / 500)
        cate.append("runtime-vol-40")
        rows.append(model1.throughput)
        x.append(i / 500)
        cate.append("throughput")
        rows.append(model1.insitu.cur_layer)
        x.append(i / 500)
        cate.append("layer")
        # rows.append(model1.insitu.)
        # x.append(i / 500)
        # cate.append("state")
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
