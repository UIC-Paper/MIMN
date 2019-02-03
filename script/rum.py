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

class RUMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num, reuse=False, 
                 output_dim=None, clip_value=20, batch_size=128):
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.reuse = reuse
        self.clip_value = clip_value
        self.batch_size = batch_size

        def single_cell(num_units):
            return tf.nn.rnn_cell.GRUCell(num_units)


        self.controller = single_cell(self.controller_units)
        self.step = 0
        self.output_dim = output_dim

        self.o2p_initializer = create_linear_initializer(self.controller_units)

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_state["controller_state"])

        num_parameters_per_head = self.memory_vector_dim + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        

        controller_par = controller_output
        
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(
                controller_par, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer)

            parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)
            
        prev_M = prev_state["M"]
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim]) + 1
            with tf.variable_scope('addressing_head_%d' % i):     
                w = self.addressing(k, beta, prev_M)
            
            w_list.append(w)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]
            

        M = prev_M

        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)


        read_output = controller_output

        self.step += 1
        return read_output, {
                "controller_state" : controller_state,
                "read_vector_list" : read_vector_list,
                "w_list" : w_list,
                "M" : M
            }


    def addressing(self, k, beta, prev_M):
        # Cosine Similarity
        def cosine_similarity(key, M):
            key = tf.expand_dims(key, axis=2)
            inner_product = tf.matmul(M, key)
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=1, keep_dims=True))
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), axis=2, keep_dims=True))
            norm_product = M_norm * k_norm
            K = tf.squeeze(inner_product / (norm_product + 1e-8))                   
            return K

        K = cosine_similarity(k,prev_M)
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  

        return w_c



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

            state = {
                "controller_state" : controller_init_state,
                "read_vector_list" : read_vector_list,
                "w_list" : w_list,
                "M" : M
            }
            return state


