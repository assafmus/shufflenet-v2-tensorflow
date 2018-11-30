import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
import time

from architecture import shufflenet

tf.reset_default_graph()

BATCH_SIZE = 32
random_batch = tf.constant(np.random.randn(BATCH_SIZE, 224, 224, 3), dtype=tf.float32)
logits = shufflenet(random_batch, is_training=False, num_classes=40, depth_multiplier='1.0')

init = tf.global_variables_initializer()

times = []
with tf.Session() as sess:
    sess.run(init)
    for _ in range(60):
        start = time.time()
        _ = sess.run(logits)
        times.append(time.time() - start)

times = np.array(times[10:]) / BATCH_SIZE
print('images per second:', 1/times.mean())

