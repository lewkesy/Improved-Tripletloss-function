import sys
sys.path.append("../utils")
from alexnet import alexnet_model
from batch_generator import triplessIterator

alexnet = alexnet_model((224, 224, 3), 100, 0.00001)
train_gen = triplessIterator(2)
print(train_gen.next())
