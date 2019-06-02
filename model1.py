import tensorflow as tf
from tensorflow.keras.preprocessing import image
keras = tf.keras

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
    input_image = keras.applications.ResNet50.preprocess_inputpreprocess_input(x)

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
