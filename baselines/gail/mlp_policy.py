"""
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
"""
import tensorflow as tf
import gym

from baselines.common import tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype

from baselines.acktr.utils import dense

from .residual_block import residual_block


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True,
              is_training=True, dtype=tf.float32):
        assert isinstance(ob_space, gym.spaces.Box)

        # placeholder
        self.pd_type = make_pdtype(ac_space)
        is_stochastic = tf.placeholder(dtype=tf.bool, shape=(), name="is_stochastic")
        state_ph = U.get_placeholder(dtype=dtype, shape=[None] + list(ob_space.shape), name="state_ph")

        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape, dtype=dtype)

        state = tf.clip_by_value((state_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0) 
        
        with tf.variable_scope('CNN'):
            x = state
            
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

        # 全连接层
        last_out = x
        for i in range(num_hid_layers):
            last_out = tf.layers.dense(
                last_out,
                hid_size,
                tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="vffc%i" % (i + 1)
            )
        self.v_predict = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0), dtype=dtype)[:, 0]
            
        last_out = x
        for i in range(num_hid_layers):
            last_out = tf.layers.dense(
                last_out,
                hid_size,
                tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="polfc%i" % (i + 1)
            )

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, self.pd_type.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01), dtype=dtype)
            logstd = tf.get_variable(name="logstd",
                                     shape=[1, self.pd_type.param_shape()[0]//2],
                                     initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, self.pd_type.param_shape()[0], "polfinal", U.normc_initializer(0.01), dtype=dtype)

        self.pd = self.pd_type.pdfromflat(pdparam)
        ac = U.switch(is_stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([is_stochastic, state_ph], [ac, self.v_predict])

    def act(self, ob, stochastic=False):
        ac, v_predict = self._act(stochastic, ob[None])
        return ac, v_predict

    def get_variables(self, scope=None):
        if scope is None:
            scope = self.scope
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

    def get_trainable_variables(self, scope=None):
        if scope is None:
            scope = self.scope
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
