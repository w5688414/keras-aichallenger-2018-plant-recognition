from keras.models import Model, load_model
from custom_layers import Scale

img_width, img_height = 224, 224
# model=load_model('./trained_model/resnet50/resnet50.36-0.8762.hdf5')
model=load_model('./trained_model/xception/xception.38-0.8740.hdf5')
# model=load_model('./trained_model/seresnet50/seresnet50.43-0.8795.hdf5')
# model=load_model('./trained_model/densenet161/densenet161.42-0.8800.hdf5',custom_objects={'Scale':Scale})
# model=load_model('./trained_model/inception_v3/inception_v3.43-0.8747.hdf5')
# model=load_model('./trained_model/inception_resnet_v2/inception_resnet_v2.46-0.8764.hdf5')
# model=load_model('./trained_model/resnet34/resnet34.54-0.8773.hdf5')
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
print(model.summary())

from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(1.0/255)
test_dir='/home/eric/data/plant/ai_challenger_pdr2018_testa_20181023/AgriculturalDisease_testA'

classes=[]
with open("labels.txt","r") as f:
    for line in f.readlines():
        classes.append(line.strip("\n").split(" ")[0])

generator=test_datagen.flow_from_directory(
                         test_dir,
                         target_size=(img_width,img_height),
                         batch_size=1,
                         class_mode=None,
                         shuffle=False,
                         seed=42
                    )

probability=model.predict_generator(generator,steps=10,verbose=1)

data={}
data['filename']=generator.filenames
for i in range(probability.shape[1]):
    data[i]=probability[:,i]

import pandas as pd
result=pd.DataFrame(data)
result.to_csv('submit.csv',index=False)