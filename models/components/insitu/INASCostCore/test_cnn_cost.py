import models.components.insitu.INASCostCore.cnn as cnn
import models.components.insitu.INASCostCore.plat_energy_costs as energy_costs
from models.components.insitu.INASCostCore.ExecInfo import ExecInfo


class Mat:
    def __init__(self, data, n, ch, h, w):
        self.data = data
        self.n = n
        self.ch = ch
        self.h = h
        self.w = w

    def __repr__(self):
        s = "MAT(data = []" + \
            ", n = " + str(self.n) + \
            ", ch = " + str(self.ch) + \
            ", h = " + str(self.h) + \
            ", w = " + str(self.w) + ")\n"
        return s


def get_network(NAS_SETTINGS, predic_actions, model_fn_type='MODEL_FN_CONV2D'):
    network = []
    ifm_h = NAS_SETTINGS['IMG_SIZE']
    ifm_w = NAS_SETTINGS['IMG_SIZE']
    ifm_ch = NAS_SETTINGS['IMG_CHANNEL']
    ofm_ch = 0
    layers = NAS_SETTINGS['NUM_LAYERS']
    num_classes = NAS_SETTINGS['NUM_CLASS']

    # -- get network depending on model function type
    if model_fn_type == "MODEL_FN_CONV2D":
        # CONV layers
        for i in range(layers):
            kernel_h = predic_actions[i * 2]
            kernel_w = predic_actions[i * 2]
            filter_size = predic_actions[i * 2 + 1]
            ofm_ch = filter_size
            ofm_h = ifm_h - kernel_h + 1
            ofm_w = ifm_w - kernel_w + 1
            item = {
                'name': "CONV_" + str(i), 'type': "CONV", 'stride': 1,
                'K': Mat(None, filter_size, ifm_ch, kernel_h, kernel_w),  # n, ch, h, w
                'IFM': Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM': Mat(None, 1, filter_size, ofm_h, ofm_w),
            }
            network.append(item)
            ifm_h = ofm_h;
            ifm_w = ofm_w
            ifm_ch = filter_size

        # append GLOBAL AVG POOLING layer
        item = {
            'name': "GAVGPOOL_0", 'type': "GAVGPOOL", 'stride': 1,
            'K': Mat(None, ifm_ch, ifm_ch, ifm_h, ifm_w),
            'IFM': Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM': Mat(None, 1, ifm_ch, 1, 1),
        }
        network.append(item)

    elif model_fn_type == "MODEL_FN_CONV1D":
        ifm_w = 1
        # CONV layers
        for i in range(layers):
            kernel_h = predic_actions[i * 2];
            kernel_w = 1
            filter_size = predic_actions[i * 2 + 1]
            ofm_ch = filter_size
            ofm_h = ifm_h - kernel_h + 1;
            ofm_w = 1
            item = {
                'name': "CONV_" + str(i), 'type': "CONV", 'stride': 1,
                'K': Mat(None, filter_size, ifm_ch, kernel_h, kernel_w),  # n, ch, h, w
                'IFM': Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM': Mat(None, 1, filter_size, ofm_h, ofm_w),
            }
            network.append(item)
            ifm_h = ofm_h;
            ifm_w = ofm_w
            ifm_ch = filter_size

        # append GLOBAL AVG POOLING layer
        item = {
            'name': "GAVGPOOL_0", 'type': "GAVGPOOL", 'stride': 1,
            'K': Mat(None, ifm_ch, ifm_ch, ifm_h, ifm_w),
            'IFM': Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM': Mat(None, 1, ifm_ch, 1, 1),
        }
        network.append(item)

    elif model_fn_type == "MODEL_FN_FC":
        ifm_w = 1
        # FC layers
        for i in range(layers):
            kernel_h = ifm_h;
            kernel_w = 1
            filter_size = predic_actions[i]
            ofm_ch = filter_size
            ofm_h = 1;
            ofm_w = 1
            item = {
                'name': "FC_" + str(i), 'type': "FC", 'stride': 1,
                'K': Mat(None, filter_size, ifm_ch, kernel_h, kernel_w),  # n, ch, h, w
                'IFM': Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM': Mat(None, 1, filter_size, ofm_h, ofm_w),
            }
            network.append(item)
            ifm_h = ofm_h;
            ifm_w = ofm_w
            ifm_ch = filter_size
    else:
        sys.exit("get_network:: unknown")

    # -- append last FC layer
    item = {
        'name': "FC_END", 'type': "FC", 'stride': 1,
        'K': Mat(None, num_classes, ifm_ch, 1, 1),
        'IFM': Mat(None, 1, ifm_ch, 1, 1), 'OFM': Mat(None, 1, num_classes, 1, 1),
    }
    network.append(item)

    return network

def get_conv_cost(layer) -> ExecInfo:
    # layer = {}
    params_pres = {}
    params_exec = {}
    plat_settings = {}
    # layer['type'] = "CONV"
    # # 0 n ch h w
    # R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    # H = layer['IFM'].h; W = layer['IFM'].w
    # layer['K'] = Mat(0, 4, 3, 5, 5)
    # layer['OFM'] = Mat(0, 1, 4, 28, 28)
    # layer['IFM'] = Mat(0, 1, 3, 32, 32)
    # maxTr = layer['OFM'].h
    # maxTc = layer['OFM'].w
    # maxTm = layer['OFM'].ch
    # maxTn = layer['IFM'].ch
    # layer['stride'] = 1
    params_exec['inter_lo'] = 'reuse_I'
    params_pres['backup_batch_size'] = 1
    # Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']
    params_exec['tile_size'] = layer['tile_size']
    plat_cost_profile = energy_costs.PlatformCostModel.PLAT_MSP430_EXTNVM
    # print(cnn.est_cost_layer_intpow(layer, params_exec,params_pres, plat_settings,plat_cost_profile))
    npc, npc_n0, npc_ngt0, Epc_max, Lpc_max, Epc_min, Lpc_min, Eb, Lb, Er, Lr, Ec, Lc, Ew, Lw = cnn.est_cost_layer_intpow(layer, params_exec,
                                                                                          params_pres, plat_settings,
                                                                                          plat_cost_profile)

    ans = ExecInfo()
    ans.first_n_latency = Lpc_max
    # print(ans.first_n_latency)
    ans.first_n_power = Epc_max / Lpc_max
    # print(ans.first_n_power)
    ans.next_n_latency = Lpc_min
    ans.next_n_power = Epc_min / Lpc_min
    ans.next_n_num = npc_ngt0 / npc_n0
    ans.all_num = npc
    ans.frequency = energy_costs.CPU_CLOCK_MSP430
    ans.Eb = Eb*npc_n0
    ans.Lb = Lb*npc_n0
    ans.Er = Er*npc
    ans.Lr = Lr*npc
    ans.Ec = Ec*npc
    ans.Lc = Lc*npc
    ans.Ew = Ew*npc
    ans.Lw = Lw*npc
    # , Er, Lr, Ec, Lc, Ew, Lw
    # print(str(ans.first_n_latency*ans.first_n_power))
    # print(str(ans.next_n_latency*ans.next_n_power))
    # print(str(npc))
    # print(str(npc_n0))
    # print(str(npc_ngt0))
    return ans


def get_gavgpool_cost(layer) -> ExecInfo:
    # layer = {}
    params_pres = {}
    params_exec = {}
    plat_settings = {}
    # layer['type'] = "GAVGPOOL"
    # # R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    # # H = layer['IFM'].h; W = layer['IFM'].w
    # # data, n, ch, h, w
    # layer['K'] = Mat(0, 24, 24, 8, 8)
    # layer['OFM'] = Mat(0, 1, 24, 1, 1)
    # layer['IFM'] = Mat(0, 1, 24, 8, 8)
    # layer['stride'] = 1
    params_exec['inter_lo'] = 'reuse_I'
    params_pres['backup_batch_size'] = 2
    # Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn
    params_exec['tile_size'] = [1, 1, 1, 1, 1, 8, 24, 1]
    plat_cost_profile = energy_costs.PlatformCostModel.PLAT_MSP430_EXTNVM
    npc, npc_n0, npc_ngt0, Epc_max, Lpc_max, Epc_min, Lpc_min = cnn.est_cost_layer_intpow(layer, params_exec, params_pres, plat_settings,
                                     plat_cost_profile)
    ans = ExecInfo()
    ans.first_n_latency = Lpc_max
    # print(ans.first_n_latency)
    ans.first_n_power = Epc_max / Lpc_max
    # print(ans.first_n_power)
    ans.next_n_latency = Lpc_min
    ans.next_n_power = Epc_min / Lpc_min
    ans.next_n_num = npc_ngt0 / npc_n0
    ans.all_num = npc
    ans.frequency = energy_costs.CPU_CLOCK_MSP430
    return ans

def get_fc_cost():
    layer = {}
    params_pres = {}
    params_exec = {}
    plat_settings = {}
    layer['name'] = "FC"
    layer['type'] = "FC"
    # R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    # H = layer['IFM'].h; W = layer['IFM'].w
    # data, n, ch, h, w
    layer['K'] = Mat(0, 10, 24, 1, 1)
    layer['OFM'] = Mat(0, 1, 10, 1, 1)
    layer['IFM'] = Mat(0, 1, 24, 1, 1)
    layer['stride'] = 1
    params_exec['inter_lo'] = 'reuse_I'
    params_pres['backup_batch_size'] = 2
    # Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn
    params_exec['tile_size'] = [1, 1, 1, 1, 1, 1, 10, 24]
    plat_cost_profile = energy_costs.PlatformCostModel.PLAT_MSP430_EXTNVM
    return cnn.est_cost_layer_intpow(layer, params_exec, params_pres, plat_settings,
                                     plat_cost_profile)


def get_pool_cost():
    layer = {}
    params_pres = {}
    params_exec = {}
    plat_settings = {}
    layer['type'] = "POOL"
    # R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    # H = layer['IFM'].h; W = layer['IFM'].w
    # data, n, ch, h, w
    layer['K'] = Mat(0, 24, 1, 4, 4)
    layer['OFM'] = Mat(0, 1, 24, 2, 2)
    layer['IFM'] = Mat(0, 1, 24, 8, 8)
    layer['stride'] = 4
    params_exec['inter_lo'] = 'reuse_I'
    params_pres['backup_batch_size'] = 2
    # Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn
    params_exec['tile_size'] = [1, 1, 1, 1, 1, 8, 24, 1]
    plat_cost_profile = energy_costs.PlatformCostModel.PLAT_MSP430_EXTNVM
    Epc_max, Lpc_max, Epc_min, Lpc_min = cnn.est_cost_layer_intpow(layer, params_exec, params_pres, plat_settings,
                                                                   plat_cost_profile)
    print(Epc_max / Lpc_max, Epc_min / Lpc_min)


if __name__ == '__main__':
    layer = {}
    layer['type'] = "CONV"
    # 0 n ch h w
    layer['K'] = Mat(0, 24, 12, 5, 5)
    layer['OFM'] = Mat(0, 1, 24, 28, 28)
    layer['IFM'] = Mat(0, 1, 12, 32, 32)
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    H = layer['IFM'].h; W = layer['IFM'].w
    maxTr = layer['OFM'].h
    maxTc = layer['OFM'].w
    maxTm = layer['OFM'].ch
    maxTn = layer['IFM'].ch
    layer['stride'] = 1
    layer['tile_size'] = [1, 1, 1, 1, 4, 4, 24, 12]
    print(get_conv_cost(layer))
# npc, npc_n0, npc_ngt0, Epc_max, Lpc_max, Epc_min, Lpc_min
