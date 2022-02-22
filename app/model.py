import os
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys

from app import app
from app.utils_laj import *
from app.data_processing import get_CMAPSSData, get_PHM08Data, data_augmentation, analyse_Data, RESCALE
today = datetime.date.today()

class CNNLSTMClass():

    def __init__(self, dataset, file_test, Train=False, trj_wise=True, plot=True):
        self.dataset = "cmapss"
        self.file_test = str(file_test)
        self.file_no = int(list(file)[9])

    def CNNLSTM(dataset, file_no, file_test, Train=False, trj_wise=False, plot=True):
        '''
        The architecture is a Meny-to-meny model combining CNN and LSTM models
        :param dataset: select the specific dataset between PHM08 or CMAPSS
        :param Train: select between training and testing
        :param trj_wise: Trajectorywise calculate RMSE and scores
        '''
        #### checkpoint saving path ####
        if file_no == 1:
            path_checkpoint = os.getcwd()+'/app/Save/Save_CNNLSTM/CNNLSTM_ML130_GRAD1_kinkRUL_FD001_seq200/CNN1D_3_lstm_2_layers'
        elif file_no == 2:
            path_checkpoint = os.getcwd()+'/app/Save/Save_CNNLSTM/CNNLSTM_ML130_GRAD1_kinkRUL_FD002/CNN1D_3_lstm_2_layers'
        elif file_no == 3:
            path_checkpoint = os.getcwd()+'/app/Save/Save_CNNLSTM/CNNLSTM_ML130_GRAD1_kinkRUL_FD003/CNN1D_3_lstm_2_layers'
        elif file_no == 4:
            path_checkpoint = os.getcwd()+'/app/Save/Save_CNNLSTM/CNNLSTM_ML130_GRAD1_kinkRUL_FD004_seq200/CNN1D_3_lstm_2_layers'
        else:
            raise ValueError("Save path not defined")
        ##################################

        if dataset == "cmapss":
            training_data, testing_data, training_pd, testing_pd, test_engine_id = get_CMAPSSData(save=False, file_test=file_test, files=[file_no])
            x_train = training_data[:, :training_data.shape[1] - 1]
            y_train = training_data[:, training_data.shape[1] - 1]
            print("training data CNN-LSTM: ", x_train.shape, y_train.shape)

            x_test = testing_data[:, :testing_data.shape[1] - 1]
            y_test = testing_data[:, testing_data.shape[1] - 1]
            print("testing data CNN-LSTM: ", x_test.shape, y_test.shape)

        elif dataset == "phm":
            training_data, testing_data, phm_testing_data = get_PHM08Data(save=False)
            x_validation = phm_testing_data[:, :phm_testing_data.shape[1] - 1]
            y_validation = phm_testing_data[:, phm_testing_data.shape[1] - 1]
            print("testing data: ", x_validation.shape, y_validation.shape)


        #----------------
        # hyperparameters
        #----------------
        batch_size = 1024           # Batch size

        # if Train == False: batch_size = 5
        if Train == False: batch_size = 10

        # sequence_length = 100       # Number of steps
        sequence_length = 200

        learning_rate = 0.001       # 0.0001

        # epochs = 5000
        epochs = 21

        ann_hidden = 50

        n_channels = 24

        lstm_size = n_channels * 3      # 3 times the amount of channels
        num_layers = 2                  # 2  # Number of layers


        X = tf.placeholder(tf.float32, [None, sequence_length, n_channels], name='inputs')
        Y = tf.placeholder(tf.float32, [None, sequence_length], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        is_train = tf.placeholder(dtype=tf.bool, shape=None, name="is_train")

        conv1 = conv_layer(X, filters=18, kernel_size=2, strides=1, padding='same', batch_norm=False, is_train=is_train,
                        scope='conv_1')
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same', name='maxpool_1')

        conv2 = conv_layer(max_pool_1, filters=36, kernel_size=2, strides=1, padding='same', batch_norm=False,
                        is_train=is_train, scope='conv_2')
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same', name='maxpool_2')

        conv3 = conv_layer(max_pool_2, filters=72, kernel_size=2, strides=1, padding='same', batch_norm=False,
                        is_train=is_train, scope='conv_3')
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same', name='maxpool_3')

        conv_last_layer = max_pool_3

        shape = conv_last_layer.get_shape().as_list()
        CNN_flat = tf.reshape(conv_last_layer, [-1, shape[1] * shape[2]])

        dence_layer_1 = dense_layer(CNN_flat, size=sequence_length * n_channels, activation_fn=tf.nn.relu, batch_norm=False,
                                    phase=is_train, drop_out=True, keep_prob=keep_prob,
                                    scope="fc_1")
        lstm_input = tf.reshape(dence_layer_1, [-1, sequence_length, n_channels])

        cell = get_RNNCell(['LSTM'] * num_layers, keep_prob=keep_prob, state_size=lstm_size)
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_output, states = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32, initial_state=init_state)
        stacked_rnn_output = tf.reshape(rnn_output, [-1, lstm_size])  # change the form into a tensor

        dence_layer_2 = dense_layer(stacked_rnn_output, size=ann_hidden, activation_fn=tf.nn.relu, batch_norm=False,
                                    phase=is_train, drop_out=True, keep_prob=keep_prob,
                                    scope="fc_2")

        output = dense_layer(dence_layer_2, size=1, activation_fn=None, batch_norm=False, phase=is_train, drop_out=False,
                            keep_prob=keep_prob,
                            scope="fc_3_output")

        prediction = tf.reshape(output, [-1])
        y_flat = tf.reshape(Y, [-1])

        h = prediction - y_flat

        cost_function = tf.reduce_sum(tf.square(h))
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(h)))
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)

        saver = tf.train.Saver()
        training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
        testing_generator = batch_generator(x_test, y_test, batch_size, sequence_length, online=False)

        if Train: model_summary(learning_rate=learning_rate, batch_size=batch_size, lstm_layers=num_layers,
                                lstm_layer_size=lstm_size, fc_layer_size=ann_hidden, sequence_length=sequence_length,
                                n_channels=n_channels, path_checkpoint=path_checkpoint, spacial_note='')

        with tf.Session() as session:
            tf.global_variables_initializer().run()

            if Train == True:
                cost = []
                iteration = int(x_train.shape[0] / batch_size)
                print("Training set MSE")
                print("No epoches: ", epochs, "No itr: ", iteration)
                __start = time.time()
                for ep in range(epochs):

                    for itr in range(iteration):
                        ## training ##
                        batch_x, batch_y = next(training_generator)
                        session.run(optimizer,
                                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8, learning_rate_: learning_rate})
                        cost.append(
                            RMSE.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0, learning_rate_: learning_rate}))

                    x_test_batch, y_test_batch = next(testing_generator)
                    mse_train, rmse_train = session.run([cost_function, RMSE],
                                                        feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0,
                                                                learning_rate_: learning_rate})
                    mse_test, rmse_test = session.run([cost_function, RMSE],
                                                    feed_dict={X: x_test_batch, Y: y_test_batch, keep_prob: 1.0,
                                                                learning_rate_: learning_rate})

                    time_per_ep = (time.time() - __start)
                    time_remaining = ((epochs - ep) * time_per_ep) / 3600
                    print("CNN-LSTM", "epoch:", ep, "\tTrainig-",
                        "MSE:", mse_train, "RMSE:", rmse_train, "\tTesting-", "MSE", mse_test, "RMSE", rmse_test,
                        "\ttime/epoch:", round(time_per_ep, 2), "\ttime_remaining: ",
                        int(time_remaining), " hr:", round((time_remaining % 1) * 60, 1), " min", "\ttime_stamp: ",
                        datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
                    __start = time.time()

                    # output checkpoint files
                    # if ep % 10 == 0 and ep != 0:
                    # if ep % 5 == 0 and ep != 0:
                    if ep % 1 == 0 and ep != 0:
                        save_path = saver.save(session, path_checkpoint)
                        if os.path.exists(path_checkpoint + '.meta'):
                            print("Model saved to file: %s" % path_checkpoint)
                        else:
                            print("NOT SAVED!!!", path_checkpoint)

                    if ep % 1000 == 0 and ep != 0: learning_rate = learning_rate / 10


                    #---------------
                    # added by Yoshi
                    #---------------
                    # if ep % 2 == 0 and ep != 0:
                    if plot:
                        trj_iteration = len(np.unique(test_engine_id))
                        # print("total trajectories: ", trj_iteration)
                        error_list = []
                        pred_list = []
                        expected_list = []
                        lower_bound = -0.01
                        test_trjectory_generator = trjectory_generator(x_test, y_test, test_engine_id, sequence_length,
                                                                        batch_size, lower_bound)
                        for itr in range(trj_iteration):
                            trj_x, trj_y = next(test_trjectory_generator)

                            __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                                feed_dict={X: trj_x, Y: trj_y, keep_prob: 1.0})

                            RUL_predict, RUL_expected = get_predicted_expected_RUL(__y, __y_pred, lower_bound)

                            error_list.append(RUL_predict - RUL_expected)
                            pred_list.append(RUL_predict)
                            expected_list.append(RUL_expected)

                            # print("id: ", itr + 1, "expected: ", RUL_expected, "\t", "predict: ", RUL_predict, "\t", "error: ",
                            #         RUL_predict - RUL_expected)

                        fig = plt.figure(figsize = (10, 10))
                        plt.rc('font', size = 18)
                        ax = fig.add_subplot()

                        plt.scatter(expected_list, pred_list, alpha=0.7, s=100)
                        plt.plot([0, 1000], [0, 1000],'k--')

                        plt.xlabel('true RUL')
                        plt.ylabel('predicted RUL')
                        plt.title('RUL correlation (CNN+LSTM)')
                        # plt.legend(loc = 'lower right')
                        plt.xlim([0, 140])
                        plt.ylim([0, 140])
                        ax.set_aspect('equal', adjustable='box')
                        plt.grid(True)
                        plt.savefig('lstm_data'+str(file_no)+'_'+f"{ep:02d}"+'.png')
                        # plt.show()
                    #---------------


                save_path = saver.save(session, path_checkpoint)
                if os.path.exists(path_checkpoint + '.meta'):
                    print("Model saved to file: %s" % path_checkpoint)
                else:
                    print("NOT SAVED!!!", path_checkpoint)

                fig = plt.figure(figsize = (10, 10))
                plt.plot(cost)
                plt.xlabel('iterations')
                plt.ylabel('cost')
                plt.savefig('lstm_data'+str(file_no)+'_cost.png')
                # plt.show()

            else:
                saver.restore(session, path_checkpoint)
                print("---")
                print("Model restored from file: %s" % path_checkpoint)
                print("---")

                if trj_wise:
                    trj_iteration = len(np.unique(test_engine_id))
                    # print("total trajectories: ", trj_iteration)
                    # print("engine id: ", test_engine_id.unique())

                    error_list = []
                    pred_list = []
                    expected_list = []
                    lower_bound = -0.01
                    test_trjectory_generator = trjectory_generator(x_test, y_test, test_engine_id, sequence_length,
                                                                batch_size, lower_bound)
                    for itr in range(trj_iteration):
                        trj_x, trj_y = next(test_trjectory_generator)

                        __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                        feed_dict={X: trj_x, Y: trj_y, keep_prob: 1.0})

                        RUL_predict, RUL_expected = get_predicted_expected_RUL(__y, __y_pred, lower_bound)

                        error_list.append(RUL_predict - RUL_expected)
                        pred_list.append(RUL_predict)
                        expected_list.append(RUL_expected)

                        print("id: ", itr + 1, "expected: ", RUL_expected, "\t", "predict: ", RUL_predict, "\t", "error: ",
                            RUL_predict - RUL_expected)
                        # plt.plot(__y_pred* RESCALE, label="prediction")
                        # plt.plot(__y* RESCALE, label="expected")
                        # plt.show()

                        #---
                        # create a plot
                        #---
                        if trj_iteration == 1:
                            trj_end = np.argmax(__y == lower_bound) - 1
                            trj_pred = __y_pred[:trj_end]
                            trj_pred[trj_pred < 0] = 0
                            trj_rul = __y[:trj_end]
                            engine_id = np.unique(test_engine_id)

                            drul = np.array(trj_pred - trj_rul)
                            drul = drul.ravel()
                            RMSE_cycle = np.sqrt(np.sum(np.square(drul)) / len(drul))
                            print("RMSE_cycles:", RMSE_cycle)
                            print("score_cycles: ", scoring_func(RMSE_cycle))

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
                            plt.rc('font', size = 14)
                            plt.rc('axes', titlesize=14)
                            plt.rc('legend', fontsize=14)

                            if file_no == 1:
                                ax1.plot(0, 0, 'og', markersize=8, label='fight conditions')
                            elif file_no == 4:
                                ax1.plot((x_test[:,1] + 1.98)*0.84/2.816, (x_test[:,0] + 1.734)*42/2.887, 'og', markersize=8, label='fight conditions')
                            ax1.fill_between([0, 0.2, 0.4, 0.6, 0.8, 0.9], [0, 0, 0, 0, 10, 15], [17, 17, 40, 40, 40, 40], alpha=0.3, linewidth=0)
                            ax1.set_xlabel('Mach number', fontsize=14)
                            ax1.set_ylabel('flight altitude (kft)', fontsize=14)
                            ax1.set_title('flight envelope')
                            ax1.legend(loc='upper left')
                            ax1.set_xlim([-0.1, 1])
                            ax1.set_ylim([-5, 50])
                            ax1.grid(True)

                            ax2.plot(np.arange(1,len(trj_pred)+1, 1), trj_pred, 'bo',  label="prediction")
                            ax2.plot(np.arange(1,len(trj_pred)+1, 1), trj_rul,  'k--', label="expected")
                            ax2.set_xlabel('cycles', fontsize=14)
                            ax2.set_ylabel('remaining useful life (RUL)', fontsize=14)
                            ax2.set_title('Engine ID: '+str(engine_id[0])+', RMSE: '+f"{RMSE_cycle:3.1f}")
                            ax2.legend(loc='lower left')
                            ax2.grid(True)

                            plt.savefig(os.getcwd()+"/app/static/images/rul.png")
                            plt.show()
                            # plt.savefig('rul_e'+f"{itr + 1:03d}"+'.png')

                    error_list = np.array(error_list)
                    error_list = error_list.ravel()
                    rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE
                    print(rmse, scoring_func(error_list))

                    # if plot:
                    #     plt.figure()
                    #     # plt.plot(expected_list, 'o', color='black', label="expected")
                    #     # plt.plot(pred_list, 'o', color='red', label="predicted")
                    #     # plt.figure()
                    #     plt.plot(np.sort(error_list), 'o', color='red', label="error")
                    #     plt.legend()
                    #     # plt.show()

                    # fig, ax = plt.subplots()
                    # ax.stem(expected_list, linefmt='b-', label="expected")
                    # ax.stem(pred_list, linefmt='r-', label="predicted")
                    # plt.legend()
                    # # plt.show()
                else:
                    x_validation = x_test
                    y_validation = y_test

                    validation_generator = batch_generator(x_validation, y_validation, batch_size, sequence_length,
                                                        online=True,
                                                        online_shift=sequence_length)

                    full_prediction = []
                    actual_rul = []
                    error_list = []

                    iteration = int(x_validation.shape[0] / (batch_size * sequence_length))
                    print("#of validation points:", x_validation.shape[0], "#datapoints covers from minibatch:",
                        batch_size * sequence_length, "iterations/epoch", iteration)

                    for itr in range(iteration):
                        x_validate_batch, y_validate_batch = next(validation_generator)
                        __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                        feed_dict={X: x_validate_batch, Y: y_validate_batch,
                                                                    keep_prob: 1.0})
                        full_prediction.append(__y_pred * RESCALE)
                        actual_rul.append(__y * RESCALE)
                        error_list.append(error * RESCALE)

                    full_prediction = np.array(full_prediction)
                    full_prediction = full_prediction.ravel()
                    actual_rul = np.array(actual_rul)
                    actual_rul = actual_rul.ravel()
                    error_list = np.array(error_list)
                    error_list = error_list.ravel()
                    rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE

                    print(y_validation.shape, full_prediction.shape, "RMSE:", rmse, "Score:", scoring_func(error_list))
                    if plot:
                        plt.plot(full_prediction, label="prediction")
                        plt.plot(actual_rul, label="expected")
                        plt.legend()
                        plt.show()


    def run(self): 
        # analyse_Data(dataset=dataset, files=[file], file_test=FILE_TEST, plot=False, min_max=False)
        self.CNNLSTM(dataset=self.dataset, file_no=self.file_no, file_test=self.file_test, Train=False, trj_wise=True, plot=True)
        # response = 
        # return response 

# if __name__ == "__main__":

#     dataset     = "cmapss"
#     file        = 1            # represent the sub-dataset for cmapss
#     # TRAIN       = True
#     TRAIN       = False
#     TRJ_WISE    = True
#     PLOT        = True

#     # file name of test data
#     if len(sys.argv) == 1:
#         FILE_TEST = None
#     else:
#         FILE_TEST = sys.argv[1]
#         file      = int(FILE_TEST[9])
#         print("test data file name:", FILE_TEST)

#     analyse_Data(dataset=dataset, files=[file], file_test=FILE_TEST, plot=False, min_max=False)

#     if TRAIN: data_augmentation(files=file,
#                                 low=[10, 35, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330],
#                                 high=[35, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350],
#                                 plot=False,
#                                 combine=False)

#     from data_processing import RESCALE, test_engine_id

#     CNNLSTM(dataset=dataset, file_no=file, Train=TRAIN, trj_wise=TRJ_WISE, plot=PLOT, file_test=FILE_TEST)