import face_recognition as fr
import face_recognition_models as frm
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import SGD
import argparse
import os
import sys
sys.path.append('./dlib-to-tf-keras-converter')
from converter.model import *
sys.path.append('./adversarial-robustness-toolbox')
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, UniversalPerturbation, BasicIterativeMethod
from PIL import Image

# sys.path.append('./cleverhans')
# from cleverhans import utils_keras, attacks

############## cribbed from face_recognition ##############

face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = frm.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = frm.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)


def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]

###########################################################

def input_keras(img):
	raw_landmarks = dlib.full_object_detections()
	for l in _raw_face_landmarks(img,None,model='small'):
		raw_landmarks.append(l)

	input = np.vstack(dlib.get_face_chips(img,raw_landmarks))
	if len(input.shape)==3:
		input = np.expand_dims(input,0)
	return input

def main(args):

    # #cleverhans thing
    # sess = tf.Session()
    # tf.keras.backend.set_session(sess)



    model_from_dlib = load_model(args.model, custom_objects=\
                       {"GlorotUniform": tf.keras.initializers.glorot_uniform,
					    "ScaleLayer": ScaleLayer,
					    "ReshapeLayer": ReshapeLayer}, compile=False)
    reference_embedding = model_from_dlib(
        K.cast(input_keras(fr.load_image_file(args.img)),dtype='float32'))
    wrong_embedding = model_from_dlib(
        K.cast(input_keras(fr.load_image_file('./gollum.jpg')),dtype='float32'))
    threshold = 1.0
    def l2s(x):
        return tf.stack(
            [tf.linalg.norm(x-reference_embedding)-K.constant(threshold),
             tf.linalg.norm(x-wrong_embedding)-K.constant(threshold)]
    )
    classifier = models.Sequential([
        layers.Input(shape=(128,)),
        layers.Flatten(),
        layers.Lambda(l2s,output_shape=(2,)),
        layers.Flatten(),
        # layers.Activation('relu',threshold=threshold)
        layers.Activation('softmax'),
        layers.Flatten()
    ])

    def euclidean_distance(vects):
        x, y = vects[0], vects[1]
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    l2_layer = layers.Lambda(euclidean_distance)

    def l2_classification(vects):
        d = euclidean_distance(vects)
        return K.cast(K.greater_equal(d,K.constant(threshold)), dtype='float32')

    class_layer = layers.Lambda(l2_classification)


    model = models.Model(inputs=[model_from_dlib.input],
                         outputs=[class_layer(K.stack([
                                    model_from_dlib.output,
                                    reference_embedding]))
                                ])

    model.compile(loss=binary_crossentropy, optimizer=SGD(lr=0.01), metrics=['accuracy'])
    print(model.summary())
    x_input = K.cast(input_keras(fr.load_image_file(args.img)),dtype='float32')
    # an example of a face you're never going to want to match
    x_other = K.cast(input_keras(fr.load_image_file('./gollum.jpg')),dtype='float32')
    print(model.evaluate(x=x_input,y=np.asarray([0.]),steps=1))
    print(model.evaluate(x=x_other,y=np.asarray([1.]),steps=1))

    adv_model = KerasClassifier(model,clip_values=(0.,255.), use_logits=False)

    eps = 0.1
    adv_crafter = FastGradientMethod(adv_model, eps=eps)

    x_adv = adv_crafter.generate(x=K.expand_dims(x_input,axis=0))

    # cleverhans thing
    # wrap = utils_keras.KerasModelWrapper(model)
    # spsa = attacks.SPSA(wrap,sess=sess)
    # # fgsm_params = {'eps': 0.3,
    # #              'clip_min': 0.,
    # #              'clip_max': 256.}
    # x = tf.keras.backend.constant(input_keras(fr.load_image_file(args.img)))
    # x_adv = spsa.generate(x,y=0,epsilon=.5,num_steps=100,batch_size=1,spsa_iters=1,
    #                         clip_min=0.,clip_max=255.)
    # x_adv = tf.stop_gradient(x_adv)
    # preds_adv = model(x_adv)


    print('Predicted P(you): ',adv_model.predict(x_adv))
    print('Saving as ',args.img,'_adv')
    im = Image.fromarray(x_adv)
    im.save('_adv'.join(os.path.splitext(args.img)))

def parse_arg(argv):
    """ Parse the arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--img',
        type=str,
        required=True,
        help='Path to image of you (or Bill Gates) to convert into an adversarial example'
    )
    arg_parser.add_argument(
        '--me',
        type=str,
        required=False,
        help='Path to folder of images to minimize similarity to, to help obscure you')

    arg_parser.add_argument(
        '--not-me',
        type=str,
        required=False,
        help='Path to folder of images to maximize similarity to, to help obscure you')

    arg_parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='./dlib_face_recognition_resnet_model_v1.h5',
        help='Path to the keras model h5 file')

    return arg_parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arg(sys.argv[1:]))
