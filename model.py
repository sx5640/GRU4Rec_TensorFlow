# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import os
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import pandas as pd
import numpy as np

class GRU4Rec:
    
    def __init__(self, sess, args):
        '''
        Parameters
        -----------
        arg.loss : 'top1', 'bpr', 'cross-entropy', 'xe_logit', top1-max, bpr-max-<X>
            selects the loss function, <X> is the parameter of the loss
        arg.final_act : 'softmax', 'linear', 'relu', 'tanh', 'softmax_logit', 'leaky-<X>', elu-<X>
            selects the activation function of the final layer, <X> is the parameter of the activation function
        arg.hidden_act : 'tanh', 'relu' or 'linear'
            selects the activation function on the hidden states
        arg.layers : int
            number of GRU layers
        arg.rnn_size : int
            number of GRU units in the layers
        arg.n_epochs : int
            number of training epochs (default: 10)
        args.n_items : int
            number of unique items in dataset
        arg.batch_size : int
            size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 50)
        arg.learning_rate : float
            learning rate (default: 0.05)
        arg.decay : float
            decay parameter for RMSProp, has no effect in other modes (default: 0.9)
        arg.decay_steps : int
            number of steps in each learning rate decay staircase
        arg.grad_cap : float
            clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
        arg.sigma : float
            "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0)
        arg.init_as_normal : boolean
            False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
        arg.reset_after_session : boolean
            whether the hidden state is set to zero after a session finished (default: True)
        arg.session_key : string
            header of the session ID column in the input file (default: 'SessionId')
        arg.item_key : string
            header of the item ID column in the input file (default: 'ItemId')
        arg.time_key : string
            header of the timestamp column in the input file (default: 'Time')
        arg.dropout_p_hidden : float
            probability of dropout of hidden units (default: 0.5)
        arg.checkpoint_dir : str
            directory for saving model (default: './checkpoint')
        '''

        self.sess = sess        # set tensorflow session

        ######## set args ########
        self.is_training = args.is_training

        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.grad_cap = args.grad_cap
        self.n_items = args.n_items 
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        ######## set args end ########

        self.build_model()      # build model
        self.sess.run(tf.global_variables_initializer())        # initialize all TF variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)      # initialize TF saver

        if self.is_training:
            return

        # use self.predict_state to hold hidden states during prediction. 
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)+1e-24))
    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))
    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def build_model(self):
        # initialize placeholders
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')      # single float for each session
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')     # single float for each session
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in xrange(self.layers)]      # single float for each session and each neuron
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            #### define initializer ####
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            #### ####

            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)        # define embedding layer, where maps n_item features to a lower dimension

            #### output layer params ####
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)        # output layer params, maps results from GRU layer to probabilistic distribution of all output
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))      # output layer bias
            #### ####

            #### gru layer ####
            cell = rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)      # define single GRU layer
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)       # apply dropout
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.layers)     # merge multiple GRU layers
            #### ####

            inputs = tf.nn.embedding_lookup(embedding, self.X)      # apply embedding before gru
            output, state = stacked_cell(inputs, tuple(self.state))     # apply GRU layer
            self.final_state = state        # record state of GRU layer

        #### output layer for training ####
        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            # here we only train on the [Y] subset of softmax_W, and the rest will be ignored. therefore the dimension of sample_W should be [len(Y), rnn_size]
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            # output layer output = final_act(sample_W*X.T) + sampled_b
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            # apply cost
            self.cost = self.loss_function(self.yhat)
        #### ####

        #### output layer for prediction ####
        else:
            # for prediction, output = final_act(W*X) + b where W and b are the complete params matrix.
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = self.final_activation(logits)

        if not self.is_training:
            return
        #### ####

        #### forward and backward propagation for training ####
        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True)) 
        
        '''
        Try different optimizers.
        '''
        #optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.AdadeltaOptimizer(self.lr)
        #optimizer = tf.train.RMSPropOptimizer(self.lr)

        #### forward propagation ####
        tvars = tf.trainable_variables()        # [embedding, softmax_W, sofmax_b, gru [[Wz, Wr],[Uz, Ur]], gru [bz, br], gru [W, U], gru b
        gvs = optimizer.compute_gradients(self.cost, tvars)     # this return ([gradient, all variables]...)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]     # apply gradient cap
        else:
            capped_gvs = gvs
        #### ####

        #### backward propagation ####
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def init(self, data):
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_sessions
    
    def fit(self, data):
        # initializing function
        self.error_during_train = False
        itemids = data[self.item_key].unique()      # unique items
        self.n_items = len(itemids)     # number of unique items

        # map item ids to a set of numbers that start from 0
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        offset_sessions = self.init(data)       # maps to starting points of each unique session in dataframe

        print('fitting model...')

        # start epoch
        for epoch in xrange(self.n_epochs):
            # initialize each epoch with placeholder vars
            epoch_cost = []     # list of cost from each run
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]     # keep record of state that send in to each round of training
            session_idx_arr = np.arange(len(offset_sessions)-1)     # unique session ids that starts from 0
            iters = np.arange(self.batch_size)      # session ids that goes into current round of training
            maxiter = iters.max()   # id of the last session in current training
            start = offset_sessions[session_idx_arr[iters]]     # starting position in dataframe of each session in current training round, which may or may not be the position of the first action
            end = offset_sessions[session_idx_arr[iters]+1]     # ending position in dataframe of each session in current training round
            finished = False
            while not finished:     # start current round
                minlen = (end-start).min()      # min session length in current round == number of actions that goes in to current round
                out_idx = data.ItemIdx.values[start]        # for each session in current round, initialize Y to be the item id of the first action in current round
                for i in range(minlen-1):       # iterate through the minimum session length in current round
                    in_idx = out_idx        # set X to be Y from previous action, which == data.ItemIdx.values[start+i]
                    out_idx = data.ItemIdx.values[start+i+1]        # set Y to be the item id of next action
                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in xrange(self.layers): 
                        feed_dict[self.state[j]] = state[j]     # update state for all session that didn't start fresh this round
                    
                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)        # run tensorflow
                    epoch_cost.append(cost)     # record cost
                    if np.isnan(cost):      # error if no cost
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:       # log at start of each epoch or start of each learning rate decay staircase
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                start = start+minlen-1      # increment starting position by minlen-1 == the end position of current round
                mask = np.arange(len(iters))[(end-start)<=1]        # batch id of the sessions ended in current round
                for idx in mask:        # iterate through ended sessions
                    maxiter += 1        # moves to the next session
                    if maxiter >= len(offset_sessions)-1:       # if there is no session to pick, end current epoch
                        finished = True
                        break
                    iters[idx] = maxiter        # change ended session to next session
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]      # set starting position of the newly added session
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]      # set ending position of the newly added session
                if len(mask) and self.reset_after_session:      # reset state for new session
                    for i in xrange(self.layers):
                        state[i][mask] = 0
            ######## epoch end ########
            
            avgc = np.mean(epoch_cost)      # average cost
            if np.isnan(avgc):      # error if no average cost
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)       # save model
    
    def predict_next_batch(self, session_ids, input_item_ids, itemidmap, batch=50):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1 
            self.predict = True
        
        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0: # change internal states with session changes
            for i in xrange(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session=session_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs}
        for i in xrange(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds, index=itemidmap.index)

