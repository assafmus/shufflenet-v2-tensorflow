import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
import time
import sys

from architecture import shufflenet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.reset_default_graph()

if True:
    DEPTH_MULTIPLIER = '1.0'
else:
    DEPTH_MULTIPLIER = '0.5'

BATCH_SIZE = 32
random_batch = tf.constant(np.random.randn(BATCH_SIZE, 224, 224, 3), dtype=tf.float32)
logits = shufflenet(random_batch, is_training=False, num_classes=40, depth_multiplier=DEPTH_MULTIPLIER)

init = tf.global_variables_initializer()

print "Running speed benchmark...",
sys.stdout.flush()
times = []
with tf.Session() as sess:
    sess.run(init)
    for _ in range(1050):
        start = time.time()
        _ = sess.run(logits)
        times.append(time.time() - start)

print "Done"
sys.stdout.flush()
times = np.array(times[50:]) / BATCH_SIZE
print 'Images per second:', 1/times.mean()

