import sys
import cv2
import json
import numpy as np
import tensorflow as tf
from imagenet40 import get_db
from architecture import shufflenet
from input_pipeline import resize_keeping_aspect_ratio, central_crop
from evaluation import evaluate_csv
if True:
    DEPTH_MULTIPLIER = '1.0'
    SAVE_PATH = 'run01/model.ckpt-1661328'
else:
    DEPTH_MULTIPLIER = '0.5'
    SAVE_PATH = 'run00/model.ckpt-1331064'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Prepare benchmark information
with open('data/integer_encoding.json', 'r') as f:
    encoding = json.load(f)

with open('data/wordnet_decoder.json', 'r') as f:
    wordnet_decoder = json.load(f)

decoder = {i: wordnet_decoder[n] for n, i in encoding.items()}

DB_PATH = '/data/imagenet40/val'
im_paths, labels, classes = get_db(DB_PATH)

classes_idx = [encoding[c] for c in classes]

# Prepare Network
tf.reset_default_graph()

raw_images = tf.placeholder(tf.uint8, [None, None, 3])
images = tf.to_float(raw_images)/255.0
MIN_DIMENSION = 256
IMAGE_SIZE = 224
images = (1.0 / 255.0) * tf.to_float(raw_images)
images = resize_keeping_aspect_ratio(images, MIN_DIMENSION)
images = central_crop(images, crop_height=IMAGE_SIZE, crop_width=IMAGE_SIZE)
images = tf.expand_dims(images, 0)

logits = shufflenet(images, is_training=False, depth_multiplier=DEPTH_MULTIPLIER)

ema = tf.train.ExponentialMovingAverage(decay=0.995)
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

print "Running accuracy benchmark...",
sys.stdout.flush()
pred = []
conf = []
with tf.Session() as sess:
    saver.restore(sess, SAVE_PATH)

    # Read Image
    for p in im_paths:
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feed_dict = {raw_images: image}
        result = sess.run(logits, feed_dict)[0]
        prob = result[classes_idx]
        prob = np.exp(prob)
        prob = prob / np.sum(prob)
        pred.append(np.argmax(prob))
        conf.append(np.max(prob))

csv_path = "results.csv"
with open(csv_path, "w") as f:
    for im_path, p, c in zip(im_paths, pred, conf):
        f.writelines("%s,%d,%f\n" % (im_path, p, c))

print "Done"
sys.stdout.flush()

evaluate_csv(csv_path)



