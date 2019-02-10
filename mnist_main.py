"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import model
import opt

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_images = np.asarray(mnist.train.images, dtype=np.float32)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_images = np.asarray(mnist.test.images, dtype=np.float32)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Train and evaluate the model with Eve
    print(tf.estimator.train_and_evaluate(
        estimator=tf.estimator.Estimator(
            model_fn=model.cnn_classifier,
            model_dir="mnist_eve_model",
            params=dict(
                optimizer=opt.EveOptimizer()
            )
        ),
        train_spec=tf.estimator.TrainSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"x": train_images},
                y=train_labels,
                batch_size=100,
                num_epochs=None,
                shuffle=True
            ),
            max_steps=20000,
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"probabilities": "softmax"},
                    every_n_iter=100
                )
            ]
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_images},
                y=eval_labels,
                num_epochs=1,
                shuffle=False
            ),
            steps=None
        )
    ))
    # Train and evaluate the model with Adam
    print(tf.estimator.train_and_evaluate(
        estimator=tf.estimator.Estimator(
            model_fn=model.cnn_classifier,
            model_dir="mnist_adam_model",
            params=dict(
                optimizer=tf.train.AdamOptimizer()
            )
        ),
        train_spec=tf.estimator.TrainSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"x": train_images},
                y=train_labels,
                batch_size=100,
                num_epochs=None,
                shuffle=True
            ),
            max_steps=20000,
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"probabilities": "softmax"},
                    every_n_iter=100
                )
            ]
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_images},
                y=eval_labels,
                num_epochs=1,
                shuffle=False
            ),
            steps=None
        )
    ))


if __name__ == "__main__":
    tf.app.run()
