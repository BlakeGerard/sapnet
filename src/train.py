from sapnet import SAPNetDQN
import state

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

sapnet_policy = SAPNetDQN(SAP_WINDOW_W, SAP_WINDOW_H)