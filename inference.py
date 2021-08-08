from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator  
import numpy as np
from tqdm import tqdm
import json
from mobilenet_v2 import relu6
from mobilenet_v2 import DepthwiseConv2D
from keras.layers import Lambda
import tensorflow as tf
import keras.backend as K
from shufflenetv2 import ShuffleNetV2
from resnet_attention_56 import Resnet_Attention_56



test_dir='/home/eric/data/plant/ai_challenger_pdr2018_testa_20181023/AgriculturalDisease_testA'
# test_dir='/home/eric/data/plant/ai_challenger_pdr2018_validationset_20181023/AgriculturalDisease_validationset/images'
test_datagen = ImageDataGenerator(1.0/255)
# img_width, img_height = 265, 265
# model=load_model('./trained_model/resnet50/resnet50.27-0.8720.hdf5')
# model=load_model('./trained_model/inception_v4/inception_v4.41-0.8670.hdf5')
# img_width, img_height = 229, 229
# model=load_model('./trained_model/mobilenet_v2/mobilenet_v2.43-0.8674.hdf5',custom_objects={'relu6':relu6,'DepthwiseConv2D':DepthwiseConv2D})
# from custom_layers import Scale
# model=load_model('./trained_model/densenet161/densenet161.42-0.8800.hdf5',custom_objects={'Scale':Scale})
#img_width, img_height = 229, 229
# charset_size=61
# model=ShuffleNetV2.ShuffleNetV2(input_shape=(img_width,img_height,3),classes=charset_size,weights='./trained_model/shufflenet_v2/shufflenet_v2.26-0.8676.hdf5')
# model=load_model('./trained_model/shufflenet_v2/shufflenet_v2.26-0.8676.hdf5',custom_objects={'DepthwiseConv2D':DepthwiseConv2D,'Lambda':Lambda})
# img_width, img_height = 224, 224
# model=Resnet_Attention_56.Resnet_Attention_56(input_shape=(img_width,img_height,3),classes=charset_size,weights='./trained_model/resnet_attention_56/resnet_attention_56.49-0.8762.hdf5')
img_width, img_height = 224, 224
# model=load_model('./trained_model/inception_v3/inception_v3.43-0.8747.hdf5')

# model=load_model('./trained_model/squeezenet/squeezenet.33-0.8026.hdf5')
# model=load_model('./trained_model/resnet34/resnet34.54-0.8773.hdf5')

model=load_model('./trained_model/xception/xception.38-0.8740.hdf5')
# model=load_model('./trained_model/inception_resnet_v2/inception_resnet_v2.42-0.8696.hdf5')
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
print(model.summary())
nb_validation_samples = 4514
batch_size=32
generator=test_datagen.flow_from_directory(
                         test_dir,
                         target_size=(img_width,img_height),
                         batch_size=batch_size,
                         class_mode=None,
                         shuffle=False,
                         seed=42
                    )

generator.reset()

pred=model.predict_generator(generator,steps=nb_validation_samples // batch_size+1,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)

filenames=generator.filenames
result = []
for i in tqdm(range(len(filenames))):
    temp_dict = {}
    temp_dict['image_id'] = filenames[i].split('/')[-1]
    temp_dict['disease_class'] = int(predicted_class_indices[i])
    result.append(temp_dict)
#     break
#     print('image %s is %d' % (test_image, sorted_arr[:,-1][0]))
json_name='submit.json'
with open(json_name, 'w') as f:
    json.dump(result, f)
    print('write %s, num is %d' % (json_name,len(result)))