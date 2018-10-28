from keras.layers import Input, Multiply,GlobalAveragePooling2D, Add, Dense, Activation
from keras.layers import ZeroPadding2D, BatchNormalization, Flatten
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda
import tensorflow as tf
from keras.initializers import glorot_uniform
import os
from keras.models import Model, load_model



def res_conv(X, filters, base, s):
    
    name_base = base + '_branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1_bn_1')(X)
    X= Activation('relu', name=name_base + '1_relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1_conv_1', kernel_initializer=glorot_uniform(seed=0))(X)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1_bn_2')(X)
    X = Activation('relu', name=name_base + '1_relu_2')(X)
    X = Conv2D(filters=F2, kernel_size=(3,3), strides=(s,s), padding='same', name=name_base + '1_conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1_bn_3')(X)
    X = Activation('relu', name=name_base + '1_relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1_conv_3', kernel_initializer=glorot_uniform(seed=0))(X)
    
    ##### Branch2 ####
    X_shortcut = BatchNormalization(axis=-1, name=name_base + '2_bn_1')(X_shortcut)
    X_shortcut= Activation('relu', name=name_base + '2_relu_1')(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=name_base + '2_conv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    
    # Final step: Add Branch1 and Branch2
    X = Add(name=base + '_Add')([X, X_shortcut])

    return X

def res_identity(X, filters, base):
    
    name_base = base + '_branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1_bn_1')(X)
    Shortcut= Activation('relu', name=name_base + '1_relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1_conv_1', kernel_initializer=glorot_uniform(seed=0))(Shortcut)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1_bn_2')(X)
    X = Activation('relu', name=name_base + '1_relu_2')(X)
    X = Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='same', name=name_base + '1_conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1_bn_3')(X)
    X = Activation('relu', name=name_base + '1_relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1_conv_3', kernel_initializer=glorot_uniform(seed=0))(X)    
    
    # Final step: Add Branch1 and the original Input itself
    X = Add(name=base + '_Add')([X, X_shortcut])

    return X

def Trunk_block(X, F, base):
    
    name_base = base
    
    X = res_identity(X, F, name_base + '_Residual_id_1')
    X = res_identity(X, F, name_base + '_Residual_id_2')
    
    return X

def interpolation(input_tensor, ref_tensor,name): # resizes input_tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]

    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value],name=name)

def Attention_1(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '_Pre_Residual_id')
    
    X_Trunk = Trunk_block(X, filters, name_base+ '_Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '_Mask_pool_3')(X)
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_3_Down')
    
    Residual_id_3_Down_shortcut = X
    
    Residual_id_3_Down_branched = res_identity(X, filters, name_base+ '_Mask_Residual_id_3_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '_Mask_pool_2')(X)
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_2_Down')
    
    Residual_id_2_Down_shortcut = X
    
    Residual_id_2_Down_branched = res_identity(X, filters, name_base+ '_Mask_Residual_id_2_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '_Mask_pool_1')(X)
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_1_Down')
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_1_Up')
    
    temp_name1 = name_base+ "_Mask_Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut,'name':temp_name1})(X)
                                          
    X = Add(name=base + '_Mask_Add_after_Interpool_1')([X, Residual_id_2_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_2_Up')
    
    temp_name2 = name_base+ "_Mask_Interpool_2"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_3_Down_shortcut,'name':temp_name2})(X)
                                          
    X = Add(name=base + '_Mask_Add_after_Interpool_2')([X, Residual_id_3_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_3_Up')
    
    temp_name3 = name_base+ "_Mask_Interpool_3"
    
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name3})(X)
                                          
    X = BatchNormalization(axis=-1, name=name_base + '_Mask_Interpool_3_bn_1')(X)
                                          
    X = Activation('relu', name=name_base + '_Mask_Interpool_3_relu_1')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '_Mask_Interpool_3_conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=-1, name=name_base + '_Mask_Interpool_3_bn_2')(X)
                                          
    X = Activation('relu', name=name_base + '_Mask_Interpool_3_relu_2')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '_Mask_Interpool_3_conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = Activation('sigmoid', name=name_base+'_Mask_sigmoid')(X)
      
    X = Multiply(name=name_base+'_Mutiply')([X_Trunk,X])
    
    X = Add(name=name_base+'_Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '_Post_Residual_id')
    
    return X

def Attention_2(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '_Pre_Residual_id')
    
    X_Trunk = Trunk_block(X, filters, name_base+ '_Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '_Mask_pool_2')(X)
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_2_Down')
    
    Residual_id_2_Down_shortcut = X
    
    Residual_id_2_Down_branched = res_identity(X, filters, name_base+ '_Mask_Residual_id_2_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '_Mask_pool_1')(X)
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_1_Down')
                                          
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_1_Up')
    
    temp_name1 = name_base+ "_Mask_Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut,'name':temp_name1})(X)
                                          
    X = Add(name=base + '_Mask_Add_after_Interpool_1')([X, Residual_id_2_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_2_Up')
    
    temp_name2 = name_base+ "_Mask_Interpool_2"
    
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name2})(X)
                                          
    X = BatchNormalization(axis=-1, name=name_base + '_Mask_Interpool_2_bn_1')(X)
                                          
    X = Activation('relu', name=name_base + '_Mask_Interpool_2_relu_1')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '_Mask_Interpool_2_conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=-1, name=name_base + '_Mask_Interpool_2_bn_2')(X)
                                          
    X = Activation('relu', name=name_base + '_Mask_Interpool_2_relu_2')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '_Mask_Interpool_2_conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = Activation('sigmoid', name=name_base+'_Mask_sigmoid')(X)
      
    X = Multiply(name=name_base+'_Mutiply')([X_Trunk,X])
    
    X = Add(name=name_base+'_Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '_Post_Residual_id')
    
    return X

def Attention_3(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '_Pre_Residual_id')
    
    X_Trunk = Trunk_block(X, filters, name_base+ '_Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '_Mask_pool_1')(X)
    
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_1_Down')
                                          
    X = res_identity(X, filters, name_base+ '_Mask_Residual_id_1_Up')
    
    temp_name2 = name_base+ "_Mask_Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name2})(X)
                                          
    X = BatchNormalization(axis=-1, name=name_base + '_Mask_Interpool_2_bn_1')(X)
                                          
    X = Activation('relu', name=name_base + '_Mask_Interpool_2_relu_1')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '_Mask_Interpool_2_conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=-1, name=name_base + '_Mask_Interpool_2_bn_2')(X)
                                          
    X = Activation('relu', name=name_base + '_Mask_Interpool_2_relu_2')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '_Mask_Interpool_2_conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = Activation('sigmoid', name=name_base+'_Mask_sigmoid')(X)
      
    X = Multiply(name=name_base+'_Mutiply')([X_Trunk,X])
    
    X = Add(name=name_base+'_Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '_Post_Residual_id')
    
    return X

class Resnet_Attention_56():
    @staticmethod
    def Resnet_Attention_56(input_shape,classes=100,weights="trained_model_resnet.hdf5"):

        X_input = Input(input_shape)

        X = Conv2D(64, (7,7), strides=(2,2), padding='same', name='conv_1', kernel_initializer=glorot_uniform(seed=0))(X_input)
        X = BatchNormalization(axis=-1, name='bn_1')(X)
        X = Activation('relu', name='relu_1')(X)
        X = MaxPooling2D((3,3), strides=(2,2), padding='same' ,name='pool_1')(X)
        X = res_conv(X, [64,64,256], 'Residual_conv_1', 1)

        ### Attention 1 Start
        X = Attention_1(X, [64,64,256], 'Attention_1')
        ### Attention 1 End

        X = res_conv(X, [128,128,512], 'Residual_conv_2', 2)

        ### Attention 2 Start
        X = Attention_2(X, [128,128,512], 'Attention_2')
        ### Attention 2 End

        X = res_conv(X, [256,256,1024], 'Residual_conv_3', 2)

        ### Attention 3 Start
        X = Attention_3(X, [256,256,1024], 'Attention_3')
        ### Attention 3 End

        X = res_conv(X, [512,512,2048], 'Residual_conv_4', 2)

        X = res_identity(X, [512,512,2048], 'Residual_id_1')
        X = res_identity(X, [512,512,2048], 'Residual_id_2')
        X = BatchNormalization(axis=-1, name='bn_2')(X)
        X = Activation('relu', name='relu_2')(X)
        X = AveragePooling2D((7,7), strides=(1,1), name='avg_pool')(X)
        X = Flatten()(X)
        X = Dense(classes, name='Dense_1')(X)
        X = Activation('softmax', name='classifier')(X)

        model = Model(inputs=X_input, outputs=X, name='attention_56')

        if os.path.isfile(weights):
            model.load_weights(weights)
            print("Model loaded")
        else:
            print("No model is found")

        return model


