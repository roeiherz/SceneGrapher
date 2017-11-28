from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

__author__ = 'roeih'


class LanguageModule(object):
    """
    This Class represent the BI-RNN
    """

    def __init__(self, timesteps=3, is_train=True, num_hidden=100, num_classes=51, num_input=300,
                 learning_rate=0.1, learning_rate_steps=1000, learning_rate_decay=0.5):
        """
        Construct module:
        - create input placeholders
        - create rnn step
        - attach rnn_step rnn_steps times
        - create labels placeholders
        - create module loss and train_step

        :param timesteps: rnn length
        :param is_train: whether the module will be used to train or eval
        :param learning_rate:
        :param learning_rate_steps:
        :param learning_rate_decay:
        """

        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate = learning_rate
        self.is_train = is_train
        self.timesteps = timesteps
        self.num_input = num_input

        # Create tf Graph Inputs
        self.create_placeholders()

        # Define weights - Hidden layer weights => 2*n_hidden because of forward + backward cells
        self.create_weights()

        # Define LSTM cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        self._x = tf.unstack(self.inputs_ph, self.timesteps, 1)
        # Define BI-RNN
        self._outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, self._x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        # self.logits = tf.matmul(self._outputs[-1], self.weights) + self.bias
        self.logit_0 = tf.matmul(self._outputs[0], self.w_0)
        self.logit_1 = tf.matmul(self._outputs[1], self.w_1)
        self.logit_2 = tf.matmul(self._outputs[2], self.w_2)
        self.logits = self.logit_0 + self.logit_1 + self.logit_2 + self.bias

        # loss
        if self.is_train:
            # Learning rate
            self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")
            # Calculates loss via self._outputs
            self.train_op, self.gradients, self.grad_placeholder, self.train_step = self.module_loss()

        self.accuracy = self.get_output()

    def create_placeholders(self, scope_name="placeholders"):
        """
        This function creates the place holders for input and labels
        """
        with tf.variable_scope(scope_name):
            self.inputs_ph = tf.placeholder(shape=[None, self.timesteps, self.num_input], dtype=tf.float32,
                                            name="relationships_inputs")
            self.labels_ph = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32,
                                            name="relationships_outputs")
            self.coeff_loss_ph = tf.placeholder(shape=(None), dtype=tf.float32, name="coeff_inputs")

    def create_weights(self, scope_name="weights"):
        """
        This function creates weights and biases
        """
        with tf.variable_scope(scope_name):
            #  2*n_hidden because of forward + backward cells
            # self.weights = tf.Variable(tf.random_normal([2 * self.num_hidden, self.num_classes]))
            self.w_0 = tf.Variable(tf.random_normal([2 * self.num_hidden, self.num_classes]))
            self.w_1 = tf.Variable(tf.random_normal([2 * self.num_hidden, self.num_classes]))
            self.w_2 = tf.Variable(tf.random_normal([2 * self.num_hidden, self.num_classes]))
            self.bias = tf.Variable(tf.random_normal([self.num_classes]))
            # self.b_0 = tf.Variable(tf.random_normal([self.num_classes]))
            # self.b_1 = tf.Variable(tf.random_normal([self.num_classes]))
            # self.b_2 = tf.Variable(tf.random_normal([self.num_classes]))

    def module_loss(self, scope_name="loss"):
        """
        This function defines the loss
        :param scope_name: "loss"
        :return:
        """

        with tf.variable_scope(scope_name):
            # Calc loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_ph, name="categorical_ce_loss")
            loss_op = tf.reduce_mean(loss)

            # With Coeff
            # loss_op = tf.reduce_mean(tf.multiply(loss, self.coeff_loss_ph))
            # Optimization
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_ph)
            gradients = optimizer.compute_gradients(loss_op)
            # create placeholder to minimize in a batch
            grad_placeholder = [(tf.placeholder("float", shape=grad[0].get_shape()), grad[1]) for grad in gradients]
            # Calc Gradients
            train_step = optimizer.apply_gradients(grad_placeholder)
            return loss_op, gradients, grad_placeholder, train_step

    def get_output(self, scope_name="get_output"):
        """
        This function will returns the outputs from the model
        :return:
        """
        with tf.variable_scope(scope_name):
            predictions = tf.nn.softmax(self.logits)
            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(self.labels_ph, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            return accuracy

    def get_train_op(self):
        """
        This function returns the train_op
        :return:
        """
        return self.train_op

    def get_labels_placeholders(self):
        """
        get module outputs place holders
        :return: tf.placeholder
        """
        return self.labels_ph

    def get_coeff_placeholders(self):
        """
        get module outputs place holders
        :return: tf.placeholder
        """
        return self.coeff_loss_ph

    def get_lr_placeholder(self):
        """
        get module outputs place holders
        :return: tf.placeholder
        """
        return self.lr_ph

    def get_inputs_placeholders(self):
        """
        get module input place holders
        :return: tf.placeholder
        """
        return self.inputs_ph

    def get_logits(self):
        """
        get module logits - likelihood of the last layer
        :return: tf.placeholder
        """
        return self.logits

