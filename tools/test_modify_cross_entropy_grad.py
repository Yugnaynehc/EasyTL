import tensorflow as tf
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad


@tf.RegisterGradient("SoftLoss")
def _MySoftLossGrad(op, grad1, grad2):
    t = _SoftmaxCrossEntropyWithLogitsGrad(op, 10 * grad1, grad2)
    print(t)
    return t


with tf.Session() as sess:
    x = tf.constant([[0.1, 0.9]])
    y = tf.constant([[0.9, 0.1]])

    with sess.graph.gradient_override_map({"SoftmaxCrossEntropyWithLogits": "SoftLoss"}):
        z = tf.nn.softmax_cross_entropy_with_logits(x, y)
    tf.initialize_all_variables().run()

    print(tf.gradients(z, [x, y])[0].eval())
