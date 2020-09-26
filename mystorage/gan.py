{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\anaconda\\envs\\JH\\lib\\site-packages\\tensorflow_core\\python\\compat\\v2_compat.py:88: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "non-resource variables are not supported in the long term\n"
     ]
    }
   ],
   "source": [
    "import tensorflow.compat.v1 as tf\n",
    "tf.disable_v2_behavior()\n",
    "import numpy as np\n",
    "import dlib\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as patches\n",
    "import numpy as np\n",
    "import sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Models\n",
    "\n",
    "detector = dlib.get_frontal_face_detector()\n",
    "sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Align Functionalize\n",
    "\n",
    "def align_faces(img):\n",
    "    dets = detector(img, 1)\n",
    "    \n",
    "    if len(dets) == 0:\n",
    "        sys.exit()\n",
    "    \n",
    "    objs = dlib.full_object_detections()\n",
    "\n",
    "    for detection in dets:\n",
    "        s = sp(img, detection)\n",
    "        objs.append(s)\n",
    "        \n",
    "    faces = dlib.get_face_chips(img, objs, size=256, padding=0.5)\n",
    "    \n",
    "    return faces"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Restoring parameters from models\\model\n"
     ]
    }
   ],
   "source": [
    "# Load BeautyGAN Pretrained\n",
    "\n",
    "sess = tf.Session()\n",
    "sess.run(tf.global_variables_initializer())\n",
    "\n",
    "saver = tf.train.import_meta_graph('models/model.meta')\n",
    "saver.restore(sess, tf.train.latest_checkpoint('models'))\n",
    "graph = tf.get_default_graph()\n",
    "\n",
    "X = graph.get_tensor_by_name('X:0') # source\n",
    "Y = graph.get_tensor_by_name('Y:0') # reference\n",
    "Xs = graph.get_tensor_by_name('generator/xs:0') # output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preprocess and Postprocess Functions\n",
    "\n",
    "def preprocess(img):\n",
    "    return img.astype(np.float32) / 127.5 - 1.\n",
    "\n",
    "def postprocess(img):\n",
    "    return ((img + 1.) * 127.5).astype(np.uint8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Images\n",
    "\n",
    "# no makeup image\n",
    "no_makeup = dlib.load_rgb_image('imgs/test.jpg')\n",
    "# no_makeup = dlib.load_rgb_image('imgs/no_face_test.png')\n",
    "no_makeup_faces = align_faces(no_makeup)\n",
    "\n",
    "# id picture image\n",
    "id_picture_1 = dlib.load_rgb_image('imgs/makeup/id_picture/1.png')\n",
    "id_picture_1_faces = align_faces(id_picture_1)\n",
    "\n",
    "id_picture_2 = dlib.load_rgb_image('imgs/makeup/id_picture/2.png')\n",
    "id_picture_2_faces = align_faces(id_picture_2)\n",
    "\n",
    "# no decorate image\n",
    "no_decorate_1 = dlib.load_rgb_image('imgs/makeup/no_decorate/1.png')\n",
    "no_decorate_1_faces = align_faces(no_decorate_1)\n",
    "\n",
    "no_decorate_2 = dlib.load_rgb_image('imgs/makeup/no_decorate/2.png')\n",
    "no_decorate_2_faces = align_faces(no_decorate_2)\n",
    "\n",
    "no_decorate_3 = dlib.load_rgb_image('imgs/makeup/no_decorate/3.png')\n",
    "no_decorate_3_faces = align_faces(no_decorate_3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "no_makeup_img = no_makeup_faces[0]\n",
    "\n",
    "X_img = preprocess(no_makeup_img)\n",
    "X_img = np.expand_dims(X_img, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# id picture image Run\n",
    "\n",
    "id_picture_1_img = id_picture_1_faces[0]\n",
    "\n",
    "Y_img = preprocess(id_picture_1_img)\n",
    "Y_img = np.expand_dims(Y_img, axis=0)\n",
    "\n",
    "output = sess.run(Xs, feed_dict={\n",
    "    X: X_img,\n",
    "    Y: Y_img\n",
    "})\n",
    "\n",
    "# Save Result Image\n",
    "plt.imsave('imgs/result_img/id_picture/1.png', postprocess(output[0]))\n",
    "    \n",
    "id_picture_2_img = id_picture_2_faces[0]\n",
    "\n",
    "Y_img = preprocess(id_picture_2_img)\n",
    "Y_img = np.expand_dims(Y_img, axis=0)\n",
    "\n",
    "output = sess.run(Xs, feed_dict={\n",
    "    X: X_img,\n",
    "    Y: Y_img\n",
    "})\n",
    "\n",
    "# Save Result Image\n",
    "plt.imsave('imgs/result_img/id_picture/2.png', postprocess(output[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# no decorate image Run\n",
    "\n",
    "no_decorate_1_img = no_decorate_1_faces[0]\n",
    "\n",
    "Y_img = preprocess(no_decorate_1_img)\n",
    "Y_img = np.expand_dims(Y_img, axis=0)\n",
    "\n",
    "output = sess.run(Xs, feed_dict={\n",
    "    X: X_img,\n",
    "    Y: Y_img\n",
    "})\n",
    "\n",
    "# Save Result Image\n",
    "plt.imsave('imgs/result_img/no_decorate/1.png', postprocess(output[0]))\n",
    "    \n",
    "no_decorate_2_img = no_decorate_2_faces[0]\n",
    "\n",
    "Y_img = preprocess(no_decorate_2_img)\n",
    "Y_img = np.expand_dims(Y_img, axis=0)\n",
    "\n",
    "output = sess.run(Xs, feed_dict={\n",
    "    X: X_img,\n",
    "    Y: Y_img\n",
    "})\n",
    "\n",
    "# Save Result Image\n",
    "plt.imsave('imgs/result_img/no_decorate/2.png', postprocess(output[0]))\n",
    "\n",
    "no_decorate_3_img = no_decorate_3_faces[0]\n",
    "\n",
    "Y_img = preprocess(no_decorate_3_img)\n",
    "Y_img = np.expand_dims(Y_img, axis=0)\n",
    "\n",
    "output = sess.run(Xs, feed_dict={\n",
    "    X: X_img,\n",
    "    Y: Y_img\n",
    "})\n",
    "\n",
    "# Save Result Image\n",
    "plt.imsave('imgs/result_img/no_decorate/3.png', postprocess(output[0]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
