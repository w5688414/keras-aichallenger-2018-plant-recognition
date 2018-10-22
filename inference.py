from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator  
import numpy as np
from tqdm import tqdm
import json


img_width, img_height = 224, 224

test_datagen = ImageDataGenerator(1.0/255)

model=load_model('./trained_model/inception_resnet_v2/inception_resnet_v2.08-0.7746.hdf5')
model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
print(model.summary())

generator=test_datagen.flow_from_directory(
                         '/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_testA',
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