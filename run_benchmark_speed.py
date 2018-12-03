import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
import time
import sys

from architecture import shufflenet

def benchmark_speed(DEPTH_MULTIPLIER, BATCH_SIZE):
    assert DEPTH_MULTIPLIER in ['1.0', '0.5']
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.reset_default_graph()

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_width', type=str, help='model_width (1.0 / 0.5)')
    parser.add_argument('--batch_size', '-b', type=int, help='batch size', default=32)
    args = parser.parse_args()

    benchmark_speed(args.model_width, args.batch_size)

