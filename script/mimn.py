import numpy as np
import tensorflow as tf

def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

class MIMNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num, reuse=False, 
                 output_dim=None, clip_value=20, shift_range=1, batch_size=128, mem_induction=0, util_reg=0, sharp_value=2.):
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.mem_induction = mem_induction
        self.util_reg = util_reg
        self.reuse = reuse
        self.clip_value = clip_value
        self.sharp_value = sharp_value
        self.shift_range = shift_range
        self.batch_size = batch_size

        def single_cell(num_units):
            return tf.nn.rnn_cell.GRUCell(num_units)

        if self.mem_induction > 0:
            self.channel_rnn = single_cell(self.memory_vector_dim)
            self.channel_rnn_state = [self.channel_rnn.zero_state(batch_size, tf.float32) for i in range(memory_size)]
            self.channel_rnn_output = [tf.zeros(((batch_size, self.memory_vector_dim))) for i in range(memory_size)]        

        self.controller = single_cell(self.controller_units)
        self.step = 0
        self.output_dim = output_dim

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_state["controller_state"])

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        
        if self.util_reg:
            max_q = 400.0
            prev_w_aggre = prev_state["w_aggre"] / max_q
            controller_par = tf.concat([controller_output, tf.stop_gradient(prev_w_aggre)], axis=1)
        else:
            controller_par = controller_output
        
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(
                controller_par, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer)

            parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)
            
        # prev_w_list = prev_state["w_list"]
        prev_M = prev_state["M"]
        key_M = prev_state["key_M"]
        w_list = []
        write_weight = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = (tf.nn.softplus(head_parameter[:, self.memory_vector_dim]) + 1)*self.sharp_value
            with tf.variable_scope('addressing_head_%d' % i):     
                w = self.addressing(k, beta, key_M, prev_M)
                
                if self.util_reg and i==1: 
                    s = tf.nn.softmax(
                        head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
                    )
                    gamma = 2*(tf.nn.softplus(head_parameter[:, -1]) + 1)*self.sharp_value
                    w = self.capacity_overflow(w, s, gamma)
                    write_weight.append(self.capacity_overflow(tf.stop_gradient(w), s, gamma))

            w_list.append(w)


        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]
            
        channel_weight = read_w_list[0]

        if self.mem_induction == 0:
            output_list = []

        elif self.mem_induction == 1:            
            _, ind = tf.nn.top_k(channel_weight, k=1)
            mask_weight = tf.reduce_sum(tf.one_hot(ind, depth=self.memory_size), axis=-2)
            output_list = []
            for i in range(self.memory_size):
                temp_output, temp_new_state = self.channel_rnn(tf.concat([x, tf.stop_gradient(prev_M[:,i]) * tf.expand_dims(mask_weight[:,i], axis=1)],axis=1), self.channel_rnn_state[i])
                self.channel_rnn_state[i] = temp_new_state * tf.expand_dims(mask_weight[:,i], axis=1) + self.channel_rnn_state[i]*(1- tf.expand_dims(mask_weight[:,i], axis=1))
                temp_output = temp_output * tf.expand_dims(mask_weight[:,i], axis=1) + self.channel_rnn_output[i]*(1- tf.expand_dims(mask_weight[:,i], axis=1))
                output_list.append(tf.expand_dims(temp_output,axis=1))        

        M = prev_M
        sum_aggre = prev_state["sum_aggre"]

        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)
            sum_aggre += tf.matmul(tf.stop_gradient(w), add_vector)

        w_aggre = prev_state["w_aggre"]
        if self.util_reg:
            w_aggre += tf.add_n(write_weight)
        else:
            w_aggre += tf.add_n(write_w_list)


        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            read_output = tf.contrib.layers.fully_connected(
                tf.concat([controller_output] + read_vector_list, axis=1), output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer)
            read_output = tf.clip_by_value(read_output, -self.clip_value, self.clip_value)

        self.step += 1
        return read_output, {
                "controller_state" : controller_state,
                "read_vector_list" : read_vector_list,
                "w_list" : w_list,
                "M" : M,
                "key_M": key_M,
                "w_aggre": w_aggre,
                "sum_aggre": sum_aggre
            }, output_list


    def addressing(self, k, beta, key_M, prev_M):
        # Cosine Similarity
        def cosine_similarity(key, M):
            key = tf.expand_dims(key, axis=2)
            inner_product = tf.matmul(M, key)
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=1, keep_dims=True))
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), axis=2, keep_dims=True))
            norm_product = M_norm * k_norm
            K = tf.squeeze(inner_product / (norm_product + 1e-8))                   
            return K

        K = 0.5*(cosine_similarity(k,key_M) + cosine_similarity(k,prev_M))
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  

        return w_c

    def capacity_overflow(self, w_g, s, gamma):
        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)     

        return w

    def capacity_loss(self, w_aggre):
        loss = 0.001 * tf.reduce_mean((w_aggre - tf.reduce_mean(w_aggre, axis=-1, keep_dims=True))**2 / self.memory_size / self.batch_size)
        return loss

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope('init', reuse=self.reuse):
            read_vector_list = [expand(tf.tanh(learned_init(self.memory_vector_dim)), dim=0, N=batch_size)
                for i in range(self.read_head_num)]

            w_list = [expand(tf.nn.softmax(learned_init(self.memory_size)), dim=0, N=batch_size)
                for i in range(self.read_head_num + self.write_head_num)]

            controller_init_state = self.controller.zero_state(batch_size, dtype)           

            M = expand(
                tf.tanh(tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-5), trainable=False)),
                dim=0, N=batch_size)
            
            key_M = expand(
                tf.tanh(tf.get_variable('key_M', [self.memory_size, self.memory_vector_dim],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                dim=0, N=batch_size)

            sum_aggre = tf.constant(np.zeros([batch_size, self.memory_size, self.memory_vector_dim]), dtype=tf.float32)
            zero_vector = np.zeros([batch_size, self.memory_size])
            zero_weight_vector = tf.constant(zero_vector, dtype=tf.float32)

            state = {
                "controller_state" : controller_init_state,
                "read_vector_list" : read_vector_list,
                "w_list" : w_list,
                "M" : M,
                "w_aggre" : zero_weight_vector,
                "key_M" : key_M,
                "sum_aggre" : sum_aggre

            }
            return state


