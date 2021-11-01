"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U

from .residual_block import residual_block


def logsigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, hidden_size, picture_size=(900, 1400, 3), entcoeff=0.001, scope="adversary", dtype=tf.float32):
        print('----------adversary_classifier----------')
        self.scope = scope
        self.observation_shape = picture_size
        print('observation_shape:', self.observation_shape)

        # Build placeholder
        self.generator_obs_ph = tf.placeholder(dtype, (None,) + self.observation_shape, name="observations_ph")
        self.expert_obs_ph = tf.placeholder(dtype, (None,) + self.observation_shape, name="expert_observations_ph")
        print('generator_obs_ph:', self.generator_obs_ph.shape)
        print('expert_obs_ph', self.expert_obs_ph.shape)

        # Build graph
        generator_logits = self.__build_graph(self.generator_obs_ph, hidden_size=hidden_size, dtype=dtype)
        expert_logits = self.__build_graph(self.expert_obs_ph, hidden_size=hidden_size, reuse=True, dtype=dtype)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

        # Build regression loss
        # let x = logits, z = targets. z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                              labels=tf.ones_like(expert_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss

        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.compute_grad = U.function(
            inputs=[self.generator_obs_ph, self.expert_obs_ph],
            outputs=self.losses + [U.flatgrad(self.total_loss, var_list)]
        )
        print('----------------------------------------')

    def __build_graph(self, obs_ph, hidden_size, reuse=False, is_training=True, dtype=tf.float32):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape, dtype=dtype)

            inputs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std

            with tf.variable_scope('CNN'):
                x = inputs

                # 三层卷积
                for filters, strides in [(32, 2), (64, 1), (64, 1)]:
                    x = tf.layers.conv2d(
                        x,
                        filters=filters,
                        kernel_size=(3, 3),
                        strides=strides,
                        padding='same',
                        activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
                    )
                    x = tf.layers.batch_normalization(x, training=is_training)
                    x = tf.layers.max_pooling2d(x, 2, 2)

                # 三层ResNeXt
                for i in range(3):
                    x = residual_block(x, 64, 64, 1, is_training)

                # flatten层
                x = tf.layers.flatten(x)

            with tf.variable_scope('FC'):
                x = tf.contrib.layers.fully_connected(x, hidden_size*8, activation_fn=tf.nn.relu)
                x = tf.contrib.layers.fully_connected(x, hidden_size*4, activation_fn=tf.nn.relu)
                x = tf.contrib.layers.fully_connected(x, hidden_size*2, activation_fn=tf.nn.relu)
                x = tf.contrib.layers.fully_connected(x, hidden_size, activation_fn=tf.nn.relu)
                logits = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs):
        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, 0)

        sess = tf.get_default_session()
        reward = sess.run(self.reward_op, feed_dict={self.generator_obs_ph: obs})
        print(reward.shape)
        return reward[0][0]

