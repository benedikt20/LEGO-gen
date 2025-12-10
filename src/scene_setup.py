import numpy as np
from mitsuba import ScalarTransform4f as T
import mitsuba as mi

def initialize_empty_scene(img_size=128):
    d = {
        "type": "scene",
        "integrator": {
            'type': 'path',
            'max_depth': 6,
        },
        
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=[0, 0, 4], # Camera origin
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16 
            },
            'film': {
                'type': 'hdrfilm',
                'width' : img_size,
                'height': img_size,
                'rfilter': { 'type': 'tent' },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },

        # -------------------- BSDFs --------------------
        # Lego-brick blue color
        'lego-blue': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.0, 0.32, 0.73],
            }
        },
        
        # -------------------- Lighting --------------------
        
        # White Environment (The Background)
        'environment_light': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [0.95, 0.95, 0.95] # Soft ambient fill
            }
        }
    }
    return d
