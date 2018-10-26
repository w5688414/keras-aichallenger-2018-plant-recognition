from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator  
import numpy as np
from tqdm import tqdm
import json
from mobilenet_v2 import relu6
from mobilenet_v2 import DepthwiseConv2D




test_datagen = ImageDataGenerator(1.0/255)
# img_width, img_height = 229, 229
# model=load_model('./trained_model/inception_v4/inception_v4.30-0.8103.hdf5')
img_width, img_height = 229, 229
model=load_model('./trained_model/mobilenet_v2/mobilenet_v2.43-0.8674.hdf5',custom_objects={'relu6':relu6,'DepthwiseConv2D':DepthwiseConv2D})
# from custom_layers import Scale
# model=load_model('./trained_model/densenet/densenet.17-0.8109.hdf5',custom_objects={'Scale':Scale})
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
print(model.summary())

generator=test_datagen.flow_from_directory(
                         '/home/eric/data/plant/ai_challenger_pdr2018_testa_20181023/AgriculturalDisease_testA',
                         target_size=(img_width,img_height),
                         batch_size=1,
                         class_mode=None,
                         shuffle=False,
                         seed=42
                    )

generator.reset()

pred=model.predict_generator(generator,steps=2,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

filenames=generator.filenames
result = []
for i in tqdm(range(len(filenames))):
    temp_dict = {}
    temp_dict['image_id'] = filenames[i].split('/')[-1]
    temp_dict['disease_class'] = int(predicted_class_indices[i])
    result.append(temp_dict)
#     break
#     print('image %s is %d' % (test_image, sorted_arr[:,-1][0]))

with open('submit.json', 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result)) 