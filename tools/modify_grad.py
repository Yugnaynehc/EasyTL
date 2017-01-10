import tensorflow as tf


# Actual gradient:
@tf.RegisterGradient("MySquare")
def _MySquareGrad(op, grad):
    x = op.inputs[0]
    return grad * 20 * x  # add a "small" error just to see the difference:


with tf.Session() as sess:
    x = tf.constant([1., 2.])
    y = tf.square(x)
    with sess.graph.gradient_override_map({"Square": "MySquare"}):
        y_ = tf.square(x)
    tf.initialize_all_variables().run()

    print('Before modity:')
    print(x.eval(), y.eval(), tf.gradients(y, x)[0].eval())

    print('After modity:')
    print(x.eval(), y_.eval(), tf.gradients(y_, x)[0].eval())
