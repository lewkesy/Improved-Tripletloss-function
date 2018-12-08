import sys
sys.path.append("../utils")
from alexnet import alexnet_model
from loss_function import triplet_loss, improved_triplet_loss
# from alexnet_multi_loss import alexnet_model
from batch_generator import ImprovedTripletIterator, ImprovedTripletIteratorTest, BasicTriplessIterator, BasicTriplessIteratorTest
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from IPython import embed

train_dir = '../data/train'
classes = os.listdir(train_dir)
num_classes = len(classes)
batch_size = 30
num_train_samples = 3700
num_test_samples = 1600

'''
original compile code for alexnet

'''

alexnet = alexnet_model((224, 224, 3), num_classes, 0.00001)
alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def get_gen(gen):
    while True:
        images, labels = gen.next()
        images = images.astype(np.float32) / 127. - 1.
        labels = [classes.index(x) for x in labels]
        labels = to_categorical(labels, num_classes=num_classes)
        yield images, labels

train_gen = get_gen(ImprovedTripletIterator(batch_size, 15))
test_gen = get_gen(ImprovedTripletIteratorTest(batch_size, 15))

'''
This is the example to train multi_loss model
To run this code, comment on the code before and uncomment the following code
Also remember to import alexnet_multi_loss instead of alexnet
To switch loss function between tripletloss and improved tripletloss, just change
the variable in the compiler

'''

# feat = alexnet.output
# pred = Activation('softmax')
# alexnet = Model(input = alexnet.input, output = [pred, feat])
# alexnet.compile(optimizer='adam', 
# 	loss=['categorical_crossentropy', improved_triplet_loss], 
# 	loss_weights=[1.0,0.8],
# 	metrics=['accuracy'])

# def get_gen_multi_loss(gen):
#     while True:
#         images, labels = gen.next()
#         images = images.astype(np.float32) / 127. - 1.
#         labels = [classes.index(x) for x in labels]
#         labels = to_categorical(labels, num_classes=num_classes)
#         yield images, [labels, np.ones([labels.shape[0] ,1])]

# train_gen = get_gen_multi_loss(triplessIterator(batch_size))



alexnet.fit_generator(train_gen, steps_per_epoch=int(num_train_samples/batch_size), epochs=6, validation_data=test_gen, validation_steps=int(num_test_samples/batch_size), max_queue_size=10, workers=5, shuffle=True)
quit()
for step in range(num_train_steps):
    im, labels = train_gen.__next__()
    alexnet.train_on_batch(im, labels)
