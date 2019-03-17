import numpy as np
import tensorflow as tf
import tempfile
import urllib
import os
import math
import pandas as pd
import argparse
import sys

DIR_SUMMARY = './ml_algorithm/summary'
DIR_MODEL = './ml_algorithm/tmp'


class ml:
    def __init__(self, learning_rate=0.05, epochs=10, batch_size=100,
                 data_path='./data/test_cases.csv', test_train_p=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_path = data_path
        self.g1_dfs, self.g2_dfs, self.label_dfs = self.data_process(data_path, test_train_p)

    def to_dtype(self, df, dtype):
        for i in xrange(df.shape[1]):
            df[i] = df[i].astype(dtype=dtype)
        return df

    def data_process(self, path, test_train_p=0.1):
        df = pd.read_csv(path, header=0)
        self.batch_number = int(df.shape[0] * (1 - test_train_p) / self.batch_size + 1)

        # g1_df = self.to_dtype(df['g1'].str.split(', ', expand=True), np.float32)
        # g2_df = self.to_dtype(df['g2'].str.split(', ', expand=True), np.float32)
        # self.tupe_length = g1_df.shape[1]
        # label_df =  self.to_dtype(pd.get_dummies(df['bis']), np.float32)

        g1_df = df['g1'].str.split(', ', expand=True)
        g2_df = df['g2'].str.split(', ', expand=True)
        self.tupe_length = g1_df.shape[1]
        label_df = pd.get_dummies(df['bis'])

        test_g1 = g1_df.sample(frac=test_train_p)
        train_g1 = np.array_split(g1_df.drop(test_g1.index), self.batch_number)
        test_g2 = g2_df.loc[test_g1.index]
        train_g2 = np.array_split(g2_df.drop(test_g1.index), self.batch_number)
        test_label = label_df.loc[test_g1.index]
        train_label = np.array_split(label_df.drop(test_g1.index), self.batch_number)

        return [train_g1, test_g1], [train_g2, test_g2], [train_label, test_label]

    def clean_old_record(self):
        for dir in [DIR_MODEL, DIR_SUMMARY]:
            if tf.gfile.Exists(dir):
                tf.gfile.DeleteRecursively(dir)
            tf.gfile.MakeDirs(dir)

    def fc(self, continue_train=False):
        with tf.name_scope('input'):
            g1 = tf.placeholder(tf.float32, [None, self.tupe_length])
            g2 = tf.placeholder(tf.float32, [None, self.tupe_length])
            y = tf.placeholder(tf.float32, [None, 2])

        with tf.name_scope('g1_p'):
            with tf.variable_scope('graph_pross'):
                g1_dence1 = tf.layers.dense(g1, self.tupe_length, activation=tf.nn.relu,
                                            kernel_initializer=tf.random_normal_initializer(),
                                            bias_initializer=tf.random_normal_initializer(),
                                            name='dence1')

                g1_s_dence1 = tf.layers.dense(g1_dence1, int(math.log(self.tupe_length) * 5), activation=tf.nn.relu,
                                              kernel_initializer=tf.random_normal_initializer(),
                                              bias_initializer=tf.random_normal_initializer(),
                                              name='s_dence1')

        with tf.name_scope('g2_p'):
            with tf.variable_scope('graph_pross', reuse=True):
                g2_dence1 = tf.layers.dense(g2, self.tupe_length, activation=tf.nn.relu,
                                            name='dence1',
                                            reuse=True)

                g2_s_dence1 = tf.layers.dense(g2_dence1, int(math.log(self.tupe_length) * 5), activation=tf.nn.relu,
                                              name='s_dence1',
                                              reuse=True)
        with tf.name_scope('merge'):
            two_graphs = tf.concat([g1_s_dence1, g2_s_dence1], 1)
            merge_layer = tf.layers.dense(two_graphs, int(math.log(self.tupe_length) * 5), activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(),
                                          bias_initializer=tf.random_normal_initializer())
        with tf.name_scope('logits'):
            logits = tf.layers.dense(merge_layer, 2, activation=tf.nn.softmax,
                                     kernel_initializer=tf.random_normal_initializer(),
                                     bias_initializer=tf.random_normal_initializer())
        with tf.name_scope('loss'):
            loss = tf.losses.softmax_cross_entropy(y, logits)
            tf.summary.scalar('loss', loss)

        global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss, global_step=global_step)

        tf_metric, tf_metric_update = tf.metrics.accuracy(tf.argmax(logits, 1), tf.argmax(y, 1), name='metric_acc')
        tf.summary.scalar('accuracy', tf_metric)




        metric_acc_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metric_acc")
        acc_initializer = tf.variables_initializer(var_list=metric_acc_var)

        merged_summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if continue_train:
                saver.restore(sess, DIR_MODEL + '/model.ckpt')
            else:
                self.clean_old_record()
                # initial
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(DIR_SUMMARY + '/ls_s2_1', sess.graph)
            test_writer = tf.summary.FileWriter(DIR_SUMMARY + '/ls_s2_1_t')

            for epoch in range(self.epochs):
                sess.run(acc_initializer)
                loss_p = None
                g_step = None
                for i in range(self.batch_number):
                    _, loss_p, summary, g_step = sess.run([train, loss, merged_summary, global_step],
                                                  feed_dict={g1: self.g1_dfs[0][i], g2: self.g2_dfs[0][i],
                                                             y: self.label_dfs[0][i]})
                    sess.run(tf_metric_update, feed_dict={g1: self.g1_dfs[1], g2: self.g2_dfs[1], y: self.label_dfs[1]})
                    test_writer.add_summary(summary, g_step)

                # if epoch % 5 == 0:
                # summary,acc = sess.run([merged_summary ,tf_metric])
                acc = sess.run(tf_metric)
                log_str = "Epoch %d \t Loss=%.4g \t Accuracy=%.4g \t G_step %d"
                print(log_str % (epoch, loss_p, acc, g_step))
                saver.save(sess, DIR_MODEL + '/model.ckpt')

        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # a = ml(epochs=10)
    # a.fc()

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=10, dest="epoch",
                        help="Number of training epochs")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.01, dest="learning_rate",
                        help='Initial learning rate')
    parser.add_argument("-b", "--batch_size", type=int, default=100, dest="batch_size",
                        help="Number of data for one batch")
    parser.add_argument("-p", "--data_path", default="./data/test_cases.csv", dest="data_path",
                        help="Path to input data")
    parser.add_argument("-r", "--test_train_rate", type=float, default=0.1, dest="test_train_rate",
                        help="The rate of test cases and train cases")
    parser.add_argument("-c", "--continue", type=bool, default=False, dest="continue_train",
                        help="Continue last training")
    args = parser.parse_args()
    # print args

    trainer = ml(learning_rate=args.learning_rate,
                 epochs=args.epoch,
                 batch_size=args.batch_size,
                 data_path=args.data_path)

    trainer.fc(args.continue_train)

    # self, learning_rate=0.05, epochs=10, batch_size=100, data_path = 'random_pairs.csv',test_train_p=0.1
