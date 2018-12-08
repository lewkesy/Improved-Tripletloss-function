########### loss function ####################################################
from keras import backend as Keras
import sys
sys.path.append("../data")
from data import batch_size
a = 0.6

def triplet_loss(y_true, y_pred):

	# The input data should be like [cls1, cls1, cls_another]
	x = Keras.l2_normalize(y_pred, axis=1) 
	part_batch =int(batch_size/3) 
	anchor = x[:part_batch, :]
	positive = x[part_batch: 2 * part_batch, :]
	negative = x[2 * part_batch:, :]
	dis_pos = Keras.sqrt(Keras.sum(Keras.square(Keras.abs(anchor - positive)), axis=1, keepdims=True))
	dis_neg = Keras.sqrt(Keras.sum(Keras.square(Keras.abs(anchor - negative)), axis=1, keepdims=True))
	# embed()
	res = Keras.maximum(0.0, dis_pos - dis_neg + a)

	return Keras.mean(res)


def improved_triplet_loss(y_true, y_pred):

	# The input data should be like [cls1, cls2, ..,cls_n]

	num_per_class = int(batch_size / num_for_class) # number of features in one class
	x = Keras.l2_normalize(y_pred, axis=1)
	feat_list1 = Keras.expand_dims(x, axis=0)
	feat_list2 = Keras.expand_dims(x, axis=1)
	feat1 = Keras.tile(feat_list1, [batch_size, 1, 1])
	feat2 = Keras.tile(feat_list2, [1, batch_size, 1])

	dis = Keras.sqrt(Keras.sum(Keras.square(feat1 - feat2), axis=2) + Keras.epsilon())
	
	# initialization for positive and negative
	positive = dis[:num_per_class, :num_per_class]
	negative = dis[:num_per_class, num_per_class:]

	for i in range(1, num_for_class):
		positive = tf.concat([positive, dis[i * num_per_class: (i + 1) * num_per_class, i * num_per_class: (i + 1) * num_per_class]], 0)
		if i != num_for_class - 1:
			neg_dis = tf.concat([dis[i * num_per_class: (i + 1) * num_per_class, :i * num_per_class], dis[i * num_per_class:(i + 1) * num_per_class, (i + 1) * num_per_class:]], 1)
		else:
			neg_dis = dis[i * num_per_class: (i + 1) * num_per_class, :i * num_per_class]
		negative = tf.concat([negative, neg_dis], axis=0)

	positive = Keras.max(positive, axis=1)
	negative = Keras.min(negative, axis=1)

	res = Keras.maximum(0.0, positive - negative + a)

	return Keras.mean(res)
