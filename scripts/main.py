from DIDA.config import args
from DIDA.utils.mutils import *
from DIDA.utils.data_util import *
from DIDA.utils.util import init_logger
import warnings
warnings.simplefilter("ignore")

# load data
args,data=load_data(args)

# pre-logs
log_dir=args.log_dir
init_logger(prepare_dir(log_dir) + 'log.txt')
info_dict=get_arg_dict(args)

# Runner
from DIDA.runner import Runner
from DIDA.model import DGNN
model = DGNN(args=args).to(args.device)
runner = Runner(args,model,data)
results = runner.run()

# post-logs
measure_dict=results
info_dict.update(measure_dict)
json.dump(info_dict, open(osp.join(log_dir, 'info.json'), 'w'))

