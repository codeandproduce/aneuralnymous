# Aneuralnymous

## 1. Motivation

The need to secure anonymity in a world with facial recognition technology that is fully capable of invading our privacy. 
Specifically, we use adversarial examples to defend against the [Social Mapper](https://github.com/Greenwolf/social_mapper) tool created by Jacob Wilkin(Greenwolf), which searches social media platforms for a person given a name and image using facial recognition.


As a Chrome extension, this project makes it immensely convenient and more realistic for people to actually use this tool before uploading images of themselves on the web.


## 2. Screenshots

### a. First Page
<img src="https://i.ibb.co/N6DXLDy/Screen-Shot-2019-10-20-at-2-23-04-AM.png" alt="alt text" width="300"/>

### b. Second Page
<img src="https://i.ibb.co/nDcyFTZ/Screen-Shot-2019-10-20-at-2-23-21-AM.png" alt="alt text" width="300"/>

### c. Third Page
<img src="https://i.ibb.co/SXnT9kM/Screen-Shot-2019-10-20-at-2-24-40-AM.png" alt="alt text" width="300"/>


## 3. Tech

You images are secured against facial recognition in Aneuralnymous by converting them into [adversarial examples](https://openai.com/blog/adversarial-example-research/). The altered images will look very similar to our eyes, but a neural network will misclassify them as other people or fail to recognize a human entirely. This technique is very difficult to defend against by the neural net owner unless they know exactly how the image was altered, which makes it ideal for privacy. Unfortunately, it is also highly specific to the neural net being targeted, and an example that fools one network may not fool another one. That's why for our first target we chose a face tracker that uses a common open-source pretrained model, which is easiest for Aneuralnymous to defeat.


Social Mapper uses the [face_recognition](https://github.com/ageitgey/face_recognition) library, which in turn uses [dlib](https://github.com/davisking/dlib)'s dlib_face_recognition_resnet_model_v1. We ported this pretrained model from C++ to [Keras](https://github.com/keras-team/keras) and used IBM's [Adversarial Robustness Toolbox](https://github.com/IBM/adversarial-robustness-toolbox/) to turn user-supplied photos into adversarial examples.

## 4. HackUMass

This is our submission for [HackUMass VII](https://hackumass.com/). The contributors to this project are Chan Woo Kim, Matt Bernstein, and Andrew O'Brien.
