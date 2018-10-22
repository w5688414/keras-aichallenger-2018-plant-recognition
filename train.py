#coding=utf-8
from __future__ import print_function
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout,PReLU,BatchNormalization,Activation
from keras.models import Model
from keras.layers import Input
from vgg16 import VGGNet
from inceptionv3 import Inception_v3
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from inception_resnet_v2 import InceptionResNetV2
from mobilenet_v2 import MobileNetv2
from resnet import ResNet50
import os
from xception import Xception
from shufflenetv2 import ShuffleNetV2
from keras.applications import imagenet_utils
from keras.callbacks import TensorBoard
import time
import argparse


train_data_dir = '/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_trainingset/data'
test_data_dir = '/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_validationset/data'
class_dictionary={}

# dimensions of our images.
img_width, img_height = 224, 224
charset_size = 61
nb_validation_samples = 4982
nb_train_samples = 32739
nb_epoch = 50
batch_size = 16

classes=[]
with open("labels.txt","r") as f:
    for line in f.readlines():
        classes.append(line.strip("\n").split(" ")[0])


def train(model,model_name='vgg'):
    train_datagen = ImageDataGenerator(1.0/255)
    test_datagen = ImageDataGenerator(1.0/255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        shuffle=True,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode="rgb",
        classes=classes,
        class_mode='categorical')
    # print(train_generator.class_indices)
    # imgs, labels = next(train_generator)
    # print(imgs[0])
    # print(labels[0])
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode="rgb",
        classes=classes,
        class_mode='categorical')
    # print(validation_generator.class_indices)
    save_path=os.path.join('trained_model',model_name)
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
    tensorboard = TensorBoard(log_dir='./logs/{}'.format(model_name), batch_size=batch_size)
    model_names = (os.path.join(save_path,model_name+'.{epoch:02d}-{val_acc:.4f}.hdf5'))
    model_checkpoint = ModelCheckpoint(model_names,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False)
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)
    callbacks = [model_checkpoint,reduce_learning_rate,tensorboard]

    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    model.fit_generator(train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        nb_epoch=nb_epoch,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
      )

# bn + prelu
def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def train_factory(MODEL_NAME):

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config)) 
    # model = CCR(input_shape=(img_width,img_height,1),classes=charset_size)
    # model = LeNet.build(width=img_width, height=img_height, depth=1, classes=charset_size)
    # model = ResNet.build_model(SHAPE=(img_width,img_height,1), classes=charset_size)

    # vgg net 5
    # MODEL_PATH='trained_model/vggnet5.hdf5'
    # model=VGGNet5.vgg(input_shape=(img_width,img_height,1),classes=charset_size)

    model=None
    if(MODEL_NAME=='inception_resnet_v2'):
        model=InceptionResNetV2.inception_resnet_v2(input_shape=(img_width,img_height,3),classes=charset_size)
    elif(MODEL_NAME=='xception'):
        # xeception
        model=Xception.Xception((img_width,img_height,3),classes=charset_size)
    elif(MODEL_NAME=='mobilenet_v2'):
        #mobilenet v2
        model=MobileNetv2.MobileNet_v2((img_width,img_height,3),classes=charset_size)
    elif(MODEL_NAME=='inceptionv3'):
        #mobilenet v2
        model=Inception_v3.inception((img_width,img_height,3),classes=charset_size)
    elif(MODEL_NAME=='vgg16'):
        model=VGGNet.vgg(input_shape=(img_width,img_height,3),classes=charset_size)
    elif(MODEL_NAME=='resnet'):
        model=ResNet50.resnet(input_shape=(img_width,img_height,3),classes=charset_size)


    print(model.summary())
    train(model,MODEL_NAME)

    # MODEL_NAME='ShuffleNetV2'
    # model=ShuffleNetV2.ShuffleNetV2((img_width,img_height,3),classes=charset_size)

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',"--model_name",type=str,
            help='select the model name that you want to train')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    train_factory(args.model_name)
    
# input_tensor = Input(shape=(img_width, img_height, 1))
# base_model = ResNet50(include_top=False,input_tensor=input_tensor,weights=None)
# # add a global spatial average pooling layer
# x = base_model.output
# # and a logistic layer -- let's say we have 100 classes
# predictions = Dense(100, activation='softmax')(x)
# # this is the model we will train
# model = Model(inputs=input_tensor, outputs=predictions)
# for layer in base_model.layers:
#     layer.trainable = False
# model=resnet50_100(feat_dims=1000,out_dims=100)


# model = load_model("./model.h5")



# model.save("./model.h5")

# import os
# from tqdm import tqdm
# import json
# import cv2
# import numpy as np
# from keras.preprocessing.image import img_to_array

# test_dir='/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_testA/images'
# # test_dir='/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_validationset/images'
# test_images = os.listdir(test_dir)

# result = []
# for test_image in tqdm(test_images):
#     temp_dict = {}
#     img = cv2.resize(cv2.imread(os.path.join(test_dir,test_image)), (img_width, img_height)).astype(np.float32)
#     img /= 255.0
#     image = img_to_array(img)
#     image = np.expand_dims(image, axis=0)
#     predictions=model.predict(image)
#     sorted_arr=np.argsort(predictions)
#     temp_dict['image_id'] = test_image
#     temp_dict['disease_class'] = int(sorted_arr[:,-1][0])
#     result.append(temp_dict)

# with open('submit.json', 'w') as f:
#     json.dump(result, f)
#     print('write result json, num is %d' % len(result))