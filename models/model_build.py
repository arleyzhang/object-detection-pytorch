from models import ssd
from models import fssd

ssds_map = {
                'ssd': ssd.build_ssd,
                'fssd': fssd.build_fssd,
            }

from models import vgg
networks_map = {
                    'vgg16': vgg.vgg16,
                }

def creat_model(phase, cfg):
    
    base = networks_map[cfg['base_model']]   #vgg.vgg16
    model = ssds_map[cfg['ssds_type']](phase, cfg, base)
    return model
