# ===========================
#   Actor DNN
# ===========================
import tensorflow as tf

# Network Parameters - Hidden layers
n_hidden_1 = 400
n_hidden_2 = 300


# def weight_variable(shape):
#     # initial = tf.random_normal_initializer(shape, 0.3)
#     # initial = tf.truncated_normal(shape, stddev=0.15, dtype=tf.float32)
#
#     return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    within the action bound for each generator.
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network('main')

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network('target')

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def action_scale_out(self, action_bound, out):
        # # Creating a mapping from [a, b] to [c, d]
        # # y = (x-a) * (d-c) / (b-a) + c
        # # The action will be the output for each generator
        # for i in range(self.a_dim):
        #     out[:, i] = (out[:, i] - (-1)) * (action_bound[i][1] - action_bound[i][0]) / (1 - (-1)) + action_bound[i][0]
        return out

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        # batch_norm0 = tf.layers.batch_normalization(inputs, training=True, momentum=0.9)

        # Input -> Hidden Layer
        # w1 = weight_variable([self.s_dim, n_hidden_1])
        initial = tf.contrib.layers.xavier_initializer(uniform=True)
        w1 = tf.get_variable(name='{}_w1'.format(name), shape=[self.s_dim, n_hidden_1],  initializer=initial)
        b1 = bias_variable([n_hidden_1])

        # Hidden Layer -> Hidden Layer
        # w2 = weight_variable([n_hidden_1, n_hidden_2])
        w2 = tf.get_variable(name='{}_w2'.format(name), shape=[n_hidden_1, n_hidden_2], initializer=initial)
        b2 = bias_variable([n_hidden_2])

        # Hidden Layer -> Output
        # w3 = weight_variable([n_hidden_2, self.a_dim])
        w3 = tf.get_variable(name='{}_w3'.format(name), shape=[n_hidden_2, self.a_dim], initializer=initial)
        b3 = bias_variable([self.a_dim])

        # 1st Hidden layer, OPTION: Softmax, relu, tanh or sigmoid
        h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)

        # batch normalization
        # bath_norm1 = tf.layers.batch_normalization(h1, training=True, momentum=0.9)

        # 2nd Hidden layer, OPTION: Softmax, relu, tanh or sigmoid
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # batch normalization
        # bath_norm2 = tf.layers.batch_normalization(h2, training=True, momentum=0.9)

        # Run tanh on output to get -1 to 1
        out = tf.nn.tanh(tf.matmul(h2, w3) + b3)

        # Scale output to action_bound
        scaled_out = self.action_scale_out(self.action_bound, out)
        # scaled_out = out

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_test(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
