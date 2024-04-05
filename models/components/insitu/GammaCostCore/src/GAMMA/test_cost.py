from models.components.insitu.GammaCostCore.src.GAMMA.train import *
from models.components.insitu.GammaCostCore.src.utils import set_hw_config, Constraint, put_into_actual_cstr, check_tpu, \
    translate_to_actual_cstr, translate_to_gemm
import pandas as pd
from models.components.insitu.GammaCostCore.ExecInfo import ExecInfo
DEVELOP_MODE = False


def test_cost(default_params):
    # parser = argparse.ArgumentParser()
    parser = {}
    # parser.add_argument('--fitness1', type=str, default="energy",
    #                     choices=('latency', 'energy', 'power', 'EDP', 'area'), help='First objective')
    parser['fitness1']="energy"
    
    # parser.add_argument('--fitness2', type=str, default="latency", choices=('latency', 'energy', 'power', 'EDP', 'area'),
    #                     help='Second objective')                    
    parser['fitness2']="latency"

    # parser.add_argument('--num_pop', type=int, default=default_params['--num_pop'], help='Number of populations')
    parser['num_pop']=default_params['--num_pop']

    # parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    parser['parRS']=False
    
    # parser.add_argument('--epochs', type=int, default=default_params['--epochs'], help='Number of epochs (i.e., Numbers of generations)')
    parser['epochs']=default_params['--epochs']

    # parser.add_argument('--outdir', type=str, default="outdir", help='Output directiory')
    parser['outdir']='outdir'
    
    # parser.add_argument('--num_pe', type=int, default=default_params["--num_pe"], help='Number of PEs')
    parser['num_pe'] = default_params["--num_pe"]

    # parser.add_argument('--l1_size', type=int, default=default_params["--l1_size"], help='L1 size (local buffer size)')
    parser['l1_size'] = default_params["--l1_size"]
    
    # parser.add_argument('--l2_size', type=int, default=default_params["--l2_size"], help='L2 size (global buffer size)')
    parser['l2_size'] = default_params["--l2_size"]
    
    # parser.add_argument('--NocBW', type=int, default=-1, help='Network-on-Chip BW')
    parser['NocBW'] = -1

    # parser.add_argument('--offchipBW', type=int, default=-1, help='Off-chip BW')
    parser['offchipBW']= -1
    
    # parser.add_argument('--hwconfig', type=str, default=None, help='HW configuration file')
    parser['hwconfig']= None

    # parser.add_argument('--model', type=str, default=default_params["--model"], help='Model to run')
    parser['model'] = default_params["--model"]
    
    # parser.add_argument('--num_layer', type=int, default=default_params["--num_layer"], help='Number of layers to optimize')
    parser['num_layer'] = default_params["--num_layer"]
    
    # parser.add_argument('--singlelayer', type=int, default=default_params["--singlelayer"], help='The layer index to optimize')
    parser['singlelayer'] = default_params["--singlelayer"]
    
    # parser.add_argument('--slevel_min', type=int, default=2, help='Minimum number of parallelization level')
    parser['slevel_min'] = 2
    
    # parser.add_argument('--slevel_max', type=int, default=2, help='Maximum number of parallelization level')
    parser['slevel_max'] = 2    

    # parser.add_argument('--fixedCluster', type=int, default=0, help='Rigid cluster size')
    parser['fixedCluster'] = 0

    # parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    parser['log_level'] = 1


    # parser.add_argument('--costmodel_cstr', type=str, default='maestro_cstr', help='Constraint from Cost model')
    parser['costmodel_cstr'] = 'maestro_cstr'

    # parser.add_argument('--mapping_cstr', type=str, default=None, help='Mapping constraint')
    parser['mapping_cstr'] = None

    # parser.add_argument('--accel_cstr', type=str, default=default_params["--arch"],
    #                     help='Constraint from the HW type configuration of the accelerator under design')
                        # dla,eye,tpu
    parser['accel_cstr'] = default_params["--arch"]

    # parser.add_argument('--area_budget', type=float, default=-1,
    #                     help='The area budget (mm2). Set to -1 if no area upper-bound')
    parser['area_budget']= -1
    # parser.add_argument('--pe_limit', type=int, default=-1,
    #                     help='Number of Processing Element budget. Set to -1 if no num_PE upper-bound')
    parser['pe_limit']= -1
    
    # parser.add_argument('--use_factor', default=False, action='store_true', help='To only use factor as tile size.')
    parser['use_factor'] = False

    # parser.add_argument('--inter', default=default_params["--inter"], help='Intermittent Computing Approvement.')
    parser['inter'] = default_params["--inter"]
    class Parser:
        def __init__(self, data):
            for key, value in data.items():
                setattr(self, key, value)
    opt = Parser(parser)
    opt = set_hw_config(opt)
    if DEVELOP_MODE:
        history_path = "/usr/scratch/felix/my_code/history/gamma_flex/"
        if os.path.exists(history_path) is False:
            history_path = "../models/components/insitu/GammaCostCore/data/model/"
            if os.path.exists(history_path) is False:
                history_path = "/Users/chuchu/Documents/gt_local/history/gamma_flex/"
    else:
        history_path = '../../../../'

    m_file_path = "../models/components/insitu/GammaCostCore/data/model/"
    if os.path.exists(m_file_path) is False:
        m_file_path = "./models/components/insitu/GammaCostCore/data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    if opt.singlelayer:
        model_defs = model_defs[opt.singlelayer - 1:opt.singlelayer]
    else:
        if opt.num_layer:
            model_defs = model_defs[:opt.num_layer]
    _, dim_size = model_defs.shape
    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    outdir = opt.outdir
    outdir = os.path.join(history_path, outdir)
    cstr_name = get_cstr_name(mapping_cstr=opt.mapping_cstr)
    exp_name = f"GAMMA_{opt.model}{f'-Lay{opt.singlelayer}' if opt.singlelayer > 0 else ''}{f'-nLay{opt.num_layer}' if opt.singlelayer < 1 and opt.num_layer > 0 else ''}_SL-{opt.slevel_min}-{opt.slevel_max}" \
               f"{f'_FixCl-{opt.fixedCluster}' if opt.fixedCluster > 0 else ''}_F1-{opt.fitness1}_GEN-{opt.epochs}_POP-{opt.num_pop}_Area-{opt.area_budget}_MaxPEs-{opt.pe_limit}" \
               f"{f'_FixedPE-{opt.num_pe}' if opt.num_pe > 0 else ''}{f'_L2Size-{opt.l2_size}' if opt.l2_size > 0 else ''}" \
               f"{f'_L1Size-{opt.l1_size}' if opt.l1_size > 0 else ''}{'_factorOnly' if opt.use_factor else ''}{f'_CostModelCstr-{opt.costmodel_cstr}' if opt.costmodel_cstr else ''}"
    outdir_exp = os.path.join(outdir, exp_name)
    # os.makedirs(outdir, exist_ok=True)
    # os.makedirs(outdir_exp, exist_ok=True)
    chkpt_file_t = "{}".format("result")
    chkpt_file = os.path.join(outdir_exp, chkpt_file_t + "_c.plt")
    map_cstr = None
    if opt.accel_cstr:
        accel_file = importlib.import_module(f'data.mapping_cstr.advanced_cstr.accel_cstr.{opt.accel_cstr}')
        accelator_cstr = accel_file.accel_cstr
        map_cstr = Constraint(num_pe=opt.num_pe)
        translate_to_actual_cstr(accelator_cstr, map_cstr)

    if opt.mapping_cstr:
        mapping_file = importlib.import_module(f'data.mapping_cstr.{opt.mapping_cstr}')
        mapping_cstr = mapping_file.mapping_cstr
        map_cstr = Constraint(num_pe=opt.num_pe) if not map_cstr else map_cstr
        put_into_actual_cstr(mapping_cstr, map_cstr)

    if opt.costmodel_cstr:
        mapping_file = importlib.import_module(f'data.mapping_cstr.advanced_cstr.costmodel_cstr.{opt.costmodel_cstr}')
        costmodel_cstr = mapping_file.mapping_cstr
        map_cstr = Constraint(num_pe=opt.num_pe) if not map_cstr else map_cstr
        put_into_actual_cstr(costmodel_cstr, map_cstr)

    if check_tpu(opt.accel_cstr, opt.mapping_cstr):
        model_defs = translate_to_gemm(model_defs)

    ans = {}
    # print(model_defs)
    # try:
    ans = train_model(model_defs, input_arg=opt, map_cstr=map_cstr, chkpt_file=chkpt_file, inter=opt.inter)

    # finally:
    for f in glob.glob("*.m"):
        os.remove(f)
    for f in glob.glob("*.csv"):
        os.remove(f)
    return ans
