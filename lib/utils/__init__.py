from .augmentations import SSDAugmentation
from .utils import EvalVOC, EvalCOCO

eval_solver_map = {'VOC0712': EvalVOC,
                   'COCO2014': EvalCOCO}


def eval_solver_factory(loader, cfg):
    Solver = eval_solver_map[cfg.DATASET.NAME]
    eval_solver = Solver(loader, cfg)
    return eval_solver
