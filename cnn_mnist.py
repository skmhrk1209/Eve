"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import eve

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    inputs = tf.reshape(
        tensor=features["x"],
        shape=[-1, 28, 28, 1]
    )
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding="same",
        use_bias=False
    )
    # Batch Normalization Layer #1
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Activation Layer #1
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    inputs = tf.nn.relu(inputs)
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding="same",
        use_bias=False
    )
    # Batch Normalization Layer #2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Activation Layer #2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    inputs = tf.nn.relu(inputs)
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )
    # Global Average Layer
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 64]
    inputs = tf.reduce_mean(
        input_tensor=inputs,
        axis=[1, 2]
    )
    # Logits layer
    # Input Tensor Shape: [batch_size, 64]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )
    # Add `softmax` to the graph. It is used by the `logging_hook`.
    probabilities = tf.nn.softmax(
        logits=logits,
        name="softmax"
    )
    # Generate predictions (for PREDICT and EVAL mode)
    classes = tf.argmax(
        input=probabilities,
        axis=-1
    )
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=params["optimizer"].minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()
                )
            )
    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                accuracy=tf.metrics.accuracy(
                    labels=labels,
                    predictions=classes
                )
            )
        )


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
            model_fn=cnn_model_fn,
            model_dir="mnist_eve_model",
            params=dict(
                optimizer=eve.EveOptimizer()
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
            model_fn=cnn_model_fn,
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
