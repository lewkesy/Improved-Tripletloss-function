# Here we denote that x is the output of the model
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import adam
from keras import backend as Keras
from keras.models import Sequential,Model
import numpy as np


# data
a = 0.6
lr = 0.001
adam = optimizers.adam(lr)
batch_num = 10
class_num = 80
train_data = np.random.random((2000, 50))
train_label = keras.utils.to_categorical(np.random.randint(10, size=(2000, 1)))

# loss function
def triplet_loss(y_true, y_pred):
	# I cannot guarantee the axis, we need to make experiment to ensure the axis
	# I assume that when axis = 1, the operation will operate in the line instead of column
	# The input data should be like [cls1, cls1, cls_another]
	x = Keras.l2_normalize(y_pred, axis=1) 
	part_batch =int(batch_num/3) 
	anchor = x[:part_batch, :]
	positive = x[part_batch: 2 * part_batch, :]
	negative = x[2 * part_batch:, :]
	dis_pos = Keras.sqrt(Keras.sum(Keras.square(Keras.abs(anchor - positive)), axis=1))
	dis_neg = Keras.sqrt(Keras.sum(Keras.square(Keras.abs(anchor - negative)), axis=1))
	res = Keras.maximun(0, dis_pos - dis_neg + a)

	return Keras.mean(res)


def improved_triplet_loss(y_true, y_pred):
	# I cannot guarantee the axis, we need to make experiment to ensure the axis
	# I assume that when axis = 1, the operation will operate in the line instead of column
	# The input data should be like [cls1, cls2, ..,cls_n]
	# The number of class is class_num

	gn = int(batch_num / class_num)
	x = Keras.l2_normalize(y_pred, axis=1)
	feat_list1 = Keras.expand_dims(x, axis=0)
	feat_list2 = Keras.expand_dims(x, axis=1)
	feat1 = tf.manip.tile(feat_list1)
	feat2 = tf.manip.tile(feat_list2)

	dis = Keras.sqrt(Keras.sum(Keras.square(feat1 - feat2), axis=2) + Keras.epsilon())
	
	# initialization for positive and negative
	positive = dis[:gn, :gn]
	negative = dis[:gn, gn:]

	# batch_num = cn * gn
	for i in range(class_num - 1):
		positive = Keras.concat([positive, dis[(i+1)*gn: (i+2)*gn]], axis=0)
		if i != class_num - 2:
			neg_dis = Keras.concat(dis[(i+1)*gn:, :(i+1)*gn], dis[(i+1)*gn:, (i+2)*gn:], axis=1)
		else:
			neg_dis = Keras.concat(dis[(i+1)*gn:, :(i+1)*gn])
		negative = Keras.concat([negative, neg_dis], axis=0)

	positive = Keras.max(positive, axis=1)
	negative = Keras.min(negative, axis=1)

	res = Keras.maximun(0, positive - negative + a)

	return Keras.mean(res)

# model setting
input = Input(shape=(50,))
x = Dense(64, Activation = "relu")(input)
feat = Dense(64, Activation = "relu", name = "triplet")(input)
pred = Dense(class_num, Activation = "softmax", name = "fc")(feat)
model = Model(input= input, output = [pred, feat])

model.compile(optimizer=adam,loss=['categorical_crossentropy',triplet_loss],loss_weights=[1.0,1.0])
# model.compile(optimizer=adam,loss=['categorical_crossentropy',improved_triplet_loss],loss_weights=[1.0,1.0])

model.fit(train_data, train_label, epoch = 20, batch_size = batch_num)