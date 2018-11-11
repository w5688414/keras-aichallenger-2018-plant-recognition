import tensorflow as tf
import cv2
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('random_flip_up_down', True, 'If uses flip')
flags.DEFINE_boolean('random_flip_left_right', True, 'If uses flip')
flags.DEFINE_boolean('random_brightness', True, 'If uses brightness')
flags.DEFINE_boolean('random_contrast', True, 'If uses contrast')
flags.DEFINE_boolean('random_saturation', True, 'If uses saturation')
flags.DEFINE_integer('image_size', 224, 'image size.')

"""
#flags examples
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
"""
def pre_process(images): 
    if FLAGS.random_flip_up_down:
	    images = tf.image.random_flip_up_down(images) 
    if FLAGS.random_flip_left_right: 
	    images = tf.image.random_flip_left_right(images) 
    if FLAGS.random_brightness: 
        images = tf.image.random_brightness(images, max_delta=0.3) 
    if FLAGS.random_contrast: 
        images = tf.image.random_contrast(images, 0.8, 1.2)
    if FLAGS.random_saturation:
	    tf.image.random_saturation(images, 0.3, 0.5)
    new_size = tf.constant([FLAGS.image_size,FLAGS.image_size],dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images

raw_image = cv2.imread("1.JPG")
#image = tf.Variable(raw_image)
image = tf.placeholder("uint8",[None,None,3])
images = pre_process(image)
with tf.Session() as session:
    result = session.run(images, feed_dict={image: raw_image})
cv2.imwrite('1_aug.jpg',result)
cv2.imshow("image",result.astype(np.uint8))
cv2.waitKey(1000)