from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from data import num_classes, train_gen, test_gen, num_train_samples, num_test_samples, batch_size

# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
   #print(i, layer.name)

# add a global spatial average pooling layer
x = base_model.output
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(train_gen, steps_per_epoch=int(num_train_samples/batch_size), epochs=2, validation_data=test_gen, validation_steps=int(num_test_samples/batch_size), max_queue_size=10, workers=5, shuffle=True)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# candidiate 153, 141
for layer in model.layers[:163]:
   layer.trainable = False
for layer in model.layers[163:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_gen, steps_per_epoch=int(num_train_samples/batch_size), epochs=2, validation_data=test_gen, validation_steps=int(num_test_samples/batch_size), max_queue_size=10, workers=5, shuffle=True)
