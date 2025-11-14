# revisar.py
from .tnn_training import DataManager
from .cnn import MLP

dm = DataManager.dm(10)
X, y = dm.get_random_batch()
print("batch shape:", X.shape)

from .conversion import Converter
conv = Converter.cvtr()
conv.set_model_ctor(lambda: MLP(in_features=dm.input_dim))
net = MLP(in_features=dm.input_dim)
out = net(X)
print("out shape:", out.shape)
