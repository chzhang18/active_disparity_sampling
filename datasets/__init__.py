from .kitti_dataset import KITTIDataset
from .kitti_guidedloader import KITTIGuidedLoader

__datasets__ = {
    "kitti": KITTIDataset,
    "kittiguide": KITTIGuidedLoader
}
