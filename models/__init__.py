from models.gwcnet import GwcNet_G, GwcNet_GC
from models.loss import model_loss
from models.loss import log_regression_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC
}
