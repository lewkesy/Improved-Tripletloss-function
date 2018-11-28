import sys
sys.path.append("../utils")
from alexnet import alexnet_model
from batch_generator import triplessIterator
import os
import numpy as np
from keras.utils.np_utils import to_categorical

train_dir = '../caltech/caltech_train'
classes = os.listdir(train_dir)
num_classes = len(classes)
batch_size = 30
num_train_steps = 500

alexnet = alexnet_model((224, 224, 3), num_classes, 0.00001)
alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def get_gen(gen):
    while True:
        images, labels = gen.next()
        images = images.astype(np.float32) / 127. - 1.
        labels = [classes.index(x) for x in labels]
        labels = to_categorical(labels, num_classes=num_classes)
        yield images, labels

train_gen = get_gen(triplessIterator(batch_size))

alexnet.fit_generator(train_gen, steps_per_epoch=int(6000/batch_size), epochs=6, validation_data=None, validation_steps=None, max_queue_size=10, workers=5, shuffle=True)
quit()
for step in range(num_train_steps):
    im, labels = train_gen.__next__()
    alexnet.train_on_batch(im, labels)
