#coding:utf-8
import tensorflow as tf
from utils import *
from tensorflow.python.ops.rnn_cell import GRUCell
import mimn as mimn
import rum as rum
from rnn import dynamic_rnn 
# import mann_simple_cell as mann_cell
class Model(object):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=False, Flag="DNN"):
        self.model_flag = Flag
        self.reg = False
        self.use_negsample= use_negsample
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], trainable=True)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cate_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_batch_ph)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_his_batch_ph)            


        with tf.name_scope('init_operation'):    
            self.mid_embedding_placeholder = tf.placeholder(tf.float32,[n_mid, EMBEDDING_DIM], name="mid_emb_ph")
            self.mid_embedding_init = self.mid_embeddings_var.assign(self.mid_embedding_placeholder)

        if self.use_negsample:
            self.mid_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_his_batch_ph')
            self.cate_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_cate_his_batch_ph')
            self.neg_item_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_neg_batch_ph)
            self.neg_cate_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_neg_batch_ph)
            self.neg_his_eb = tf.concat([self.neg_item_his_eb,self.neg_cate_his_eb], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))   
            
        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], axis=1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded,self.cate_his_batch_embedded], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, scope='prelu_1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, scope='prelu_2')

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsample:
                self.loss += self.aux_loss
            if self.reg:
                self.loss += self.reg_loss

            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.000001
        return y_hat

    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init,feed_dict={self.uid_embedding_placeholder: uid_weight})
    
    def init_mid_weight(self, sess, mid_weight):
        sess.run([self.mid_embedding_init],feed_dict={self.mid_embedding_placeholder: mid_weight})

    def save_mid_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding                                 
    
    def train(self, sess, inps):
        if self.use_negsample:
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.aux_loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mid_neg_batch_ph: inps[5],
                self.cate_neg_batch_ph: inps[6],
                self.mask: inps[7],
                self.target_ph: inps[8],
                self.lr: inps[9]
            })
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[7],
                self.target_ph: inps[8],
                self.lr: inps[9]
            })
            aux_loss = 0
        return loss, accuracy, aux_loss            

    def calculate(self, sess, inps):
        if self.use_negsample:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mid_neg_batch_ph: inps[5],
                self.cate_neg_batch_ph: inps[6],
                self.mask: inps[7],
                self.target_ph: inps[8]
            })
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[7],
                self.target_ph: inps[8]
            })
            aux_loss = 0
        return probs, loss, accuracy, aux_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)
        

class Model_PNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_PNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="PNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)


class Model_GRU4REC(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_GRU4REC, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, final_state1 = dynamic_rnn(GRUCell(2*EMBEDDING_DIM), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
                    
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state1], 1)
        self.build_fcn_net(inp, use_dice=False)
        

class Model_DIN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DIN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DIN")
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, HIDDEN_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, att_fea], -1)
        self.build_fcn_net(inp, use_dice=False)


class Model_ARNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_ARNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="ARNN")
        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, final_state1 = dynamic_rnn(GRUCell(2*EMBEDDING_DIM), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_gru = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, self.mask)
            att_gru = tf.reduce_sum(att_gru, 1)

        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state1, att_gru], -1)
        self.build_fcn_net(inp, use_dice=False)        

class Model_RUM(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, MEMORY_SIZE, SEQ_LEN=400, mask_flag=True):
        super(Model_RUM, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="RUM")

        def clear_mask_state(state, begin_state, mask, t):
            state["controller_state"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1))) * begin_state["controller_state"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1)) * state["controller_state"]
            state["M"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["M"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["M"]
            return state
      
        cell = rum.RUMCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE, memory_vector_dim=2*EMBEDDING_DIM,read_head_num=1, write_head_num=1,
            reuse=False, output_dim=HIDDEN_SIZE, clip_value=20, batch_size=BATCH_SIZE)
        
        state = cell.zero_state(BATCH_SIZE, tf.float32)
        begin_state = state
        for t in range(SEQ_LEN):
            output, state = cell(self.item_his_eb[:, t, :], state)
            if mask_flag:
                state = clear_mask_state(state, begin_state, self.mask, t)
        
        final_state = output
        before_memory = state['M']
        rum_att_hist = din_attention(self.item_eb, before_memory, HIDDEN_SIZE, None)

        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state, tf.squeeze(rum_att_hist)], 1)

        self.build_fcn_net(inp, use_dice=False) 

class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=400, use_negsample=False):
        super(Model_DIEN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, use_negsample, Flag="DIEN")

        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, _ = dynamic_rnn(GRUCell(2*EMBEDDING_DIM), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)        
        
        if use_negsample:
            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.neg_his_eb[:, 1:, :], self.mask[:, 1:], stag = "bigru_0")
            self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, mask=self.mask, mode="LIST", return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)

       
        
class Model_MIMN(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, MEMORY_SIZE, SEQ_LEN=400, Mem_Induction=0, Util_Reg=0, use_negsample=False, mask_flag=False):
        super(Model_MIMN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, use_negsample, Flag="MIMN")
        self.reg = Util_Reg

        def clear_mask_state(state, begin_state, begin_channel_rnn_state, mask, cell, t):
            state["controller_state"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1))) * begin_state["controller_state"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1)) * state["controller_state"]
            state["M"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["M"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["M"]
            state["key_M"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["key_M"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["key_M"]
            state["sum_aggre"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["sum_aggre"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["sum_aggre"]
            if Mem_Induction > 0:
                temp_channel_rnn_state = []
                for i in range(MEMORY_SIZE):
                    temp_channel_rnn_state.append(cell.channel_rnn_state[i] * tf.expand_dims(mask[:,t], axis=1) + begin_channel_rnn_state[i]*(1- tf.expand_dims(mask[:,t], axis=1)))
                cell.channel_rnn_state = temp_channel_rnn_state
                temp_channel_rnn_output = []
                for i in range(MEMORY_SIZE):
                    temp_output = cell.channel_rnn_output[i] * tf.expand_dims(mask[:,t], axis=1) + begin_channel_rnn_output[i]*(1- tf.expand_dims(self.mask[:,t], axis=1))
                    temp_channel_rnn_output.append(temp_output)
                cell.channel_rnn_output = temp_channel_rnn_output

            return state
      
        cell = mimn.MIMNCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE, memory_vector_dim=2*EMBEDDING_DIM,read_head_num=1, write_head_num=1,
            reuse=False, output_dim=HIDDEN_SIZE, clip_value=20, batch_size=BATCH_SIZE, mem_induction=Mem_Induction, util_reg=Util_Reg)
        
        state = cell.zero_state(BATCH_SIZE, tf.float32)
        if Mem_Induction > 0:
            begin_channel_rnn_output = cell.channel_rnn_output
        else:
            begin_channel_rnn_output = 0.0
        
        begin_state = state
        self.state_list = [state]
        self.mimn_o = []
        for t in range(SEQ_LEN):
            output, state, temp_output_list = cell(self.item_his_eb[:, t, :], state)
            if mask_flag:
                state = clear_mask_state(state, begin_state, begin_channel_rnn_output, self.mask, cell, t)
            self.mimn_o.append(output)
            self.state_list.append(state)
                
        self.mimn_o = tf.stack(self.mimn_o, axis=1)
        self.state_list.append(state)
        mean_memory = tf.reduce_mean(state['sum_aggre'], axis=-2)

        before_aggre = state['w_aggre']
        read_out, _, _ = cell(self.item_eb, state)
        
        if use_negsample:
            aux_loss_1 = self.auxiliary_loss(self.mimn_o[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.neg_his_eb[:, 1:, :], self.mask[:, 1:], stag = "bigru_0")
            self.aux_loss = aux_loss_1  

        if self.reg:
            self.reg_loss = cell.capacity_loss(before_aggre)
        else:
            self.reg_loss = tf.zeros(1)

        if Mem_Induction == 1:
            channel_memory_tensor = tf.concat(temp_output_list, 1)            
            multi_channel_hist = din_attention(self.item_eb, channel_memory_tensor, HIDDEN_SIZE, None, stag='pal')
            inp = tf.concat([self.item_eb, self.item_his_eb_sum, read_out, tf.squeeze(multi_channel_hist), mean_memory*self.item_eb], 1)
        else:
            inp = tf.concat([self.item_eb, self.item_his_eb_sum, read_out, mean_memory*self.item_eb], 1)

        self.build_fcn_net(inp, use_dice=False) 
