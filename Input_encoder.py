import tensorflow as tf
from tensorflow.keras.preprocessing import image
keras = tf.keras
from keras.models import Model

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)  
  return input_image


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

# normalizing the images to [-1, 1]

def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

def preprocess_imgs(input_image):
    size=256
    input_image=resize(input,size,size)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = keras.applications.ResNet50.preprocess_input(x)

    return input_image





def encoder_trans(img):
    IMG_SIZE=256
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

    model = Model(inputs=base_model.input, outputs=base_model.get_layer('activation_47').output)
                                               
    img=preprocess_img(img)
    feats=model.predict(img)

    return feats


img=load("img.jpg")
feats= encoder_trans(img)
