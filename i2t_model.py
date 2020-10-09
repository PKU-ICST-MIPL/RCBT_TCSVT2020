# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:01:52 2017

@author: Administrator
"""
import tensorflow as tf
import pickle
import pdb
import os
from tensorflow.contrib import rnn
class I2t:#transfer image feature to text feature
	def __init__(self,image_dim,image_lstm_len,image_lstm_dim,text_dim,text_lstm_len,text_lstm_dim,attention_dim,hidden1_dim,hidden2_dim,hidden3_dim,class_dim,weight_decay,learning_rate,reward_base,beta,gamma,w2v_w,param=None):
		self.image_dim = image_dim
		self.text_dim = text_dim
		self.image_lstm_len = image_lstm_len
		self.image_lstm_dim = image_lstm_dim
		self.text_lstm_len = text_lstm_len
		self.text_lstm_dim = text_lstm_dim
		self.attention_dim = attention_dim
		self.hidden1_dim = hidden1_dim
		self.hidden2_dim = hidden2_dim
		self.hidden3_dim = hidden3_dim
		self.class_dim = class_dim
		self.weight_decay = weight_decay
		self.learning_rate = learning_rate
		self.params = []
		self.h = 0
		self.image_whole_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_whole_data")
		self.image_lstm_data = tf.placeholder(tf.float32, shape=[None, self.image_lstm_len, self.image_lstm_dim], name="image_lstm_data")
		self.image_label = tf.placeholder(tf.float32, shape=[None, self.class_dim], name="image_label")
		self.text_whole_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_whole_data")
		self.text_lstm_data = tf.placeholder(tf.int32, shape=[None, self.text_lstm_len], name="text_lstm_data")
		self.text_label = tf.placeholder(tf.float32, shape=[None, self.class_dim], name="text_label")
		self.image_dim = self.image_dim + self.image_lstm_dim
		self.text_dim = self.text_dim + self.text_lstm_dim
		self.batch_size = tf.placeholder(tf.int32,[])
		self.keep_prob = tf.placeholder(tf.float32)
		lstm_cell_i = rnn.BasicLSTMCell(num_units=self.image_lstm_dim, forget_bias=1.0, state_is_tuple=True)
		lstm_cell_i = rnn.DropoutWrapper(cell=lstm_cell_i, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
		lstm_cell_t = rnn.BasicLSTMCell(num_units=self.text_lstm_dim, forget_bias=1.0, state_is_tuple=True)
		lstm_cell_t = rnn.DropoutWrapper(cell=lstm_cell_t, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
		mlstm_cell_i = rnn.MultiRNNCell([lstm_cell_i for i in range(2)], state_is_tuple=True)
		mlstm_cell_t = rnn.MultiRNNCell([lstm_cell_t for i in range(2)], state_is_tuple=True)

		with tf.variable_scope('train'):
			if param == None:
		
				self.w2v_w = tf.constant(w2v_w, tf.float32)
				
				self.filter_shape_i = [5, self.image_lstm_dim, 1, self.image_lstm_dim]
				self.filter_shape_t = [5, self.text_lstm_dim, 1, self.text_lstm_dim]
				self.W_i_cnn = tf.Variable(tf.truncated_normal(self.filter_shape_i, stddev=0.1), name="W_i_cnn")
				self.b_i_cnn = tf.Variable(tf.constant(0.1, shape=[self.image_lstm_dim]), name="b_i_cnn")
				self.W_t_cnn = tf.Variable(tf.truncated_normal(self.filter_shape_t, stddev=0.1), name="W_t_cnn")
				self.b_t_cnn = tf.Variable(tf.constant(0.1, shape=[self.text_lstm_dim]), name="b_t_cnn")

				
				self.Ua_i = tf.Variable(tf.random_normal([self.image_lstm_dim], stddev=0.1))
				self.Ua_t = tf.Variable(tf.random_normal([self.text_lstm_dim], stddev=0.1))
				

				
				self.W_lstm_i = tf.get_variable('W_lstm_i', [self.image_lstm_dim, self.image_dim],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.B_lstm_i = tf.get_variable('B_lstm_i', [self.image_dim], initializer=tf.constant_initializer(0.0))
				self.W_whole_i = tf.get_variable('W_whole_i', [self.image_dim, self.image_dim],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.B_whole_i = tf.get_variable('B_whole_i', [self.image_dim], initializer=tf.constant_initializer(0.0))
				self.W_lstm_t = tf.get_variable('W_lstm_t', [self.text_lstm_dim, self.text_dim],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.B_lstm_t = tf.get_variable('B_lstm_t', [self.text_dim], initializer=tf.constant_initializer(0.0))
				self.W_whole_t = tf.get_variable('W_whole_t', [self.text_dim, self.text_dim],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.B_whole_t = tf.get_variable('B_whole_t', [self.text_dim], initializer=tf.constant_initializer(0.0))
				
				self.Wq_1 = tf.get_variable('Wq_1', [self.image_dim, self.hidden1_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wq_2 = tf.get_variable('Wq_2', [self.hidden1_dim, self.hidden2_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wq_3 = tf.get_variable('Wq_3', [self.hidden2_dim, self.hidden3_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wq_4 = tf.get_variable('Wq_4', [self.hidden3_dim, self.text_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				#self.Wq_5 = tf.get_variable('Wq_5', [self.output_dim, self.class_dim],
										#initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Bq_1 = tf.get_variable('Bq_1', [self.hidden1_dim], initializer=tf.constant_initializer(0.0))
				self.Bq_2 = tf.get_variable('Bq_2', [self.hidden2_dim], initializer=tf.constant_initializer(0.0))
				self.Bq_3 = tf.get_variable('Bq_3', [self.hidden3_dim], initializer=tf.constant_initializer(0.0))
				self.Bq_4 = tf.get_variable('Bq_4', [self.text_dim], initializer=tf.constant_initializer(0.0))
				#self.Bq_5 = tf.get_variable('Bq_5', [self.class_dim], initializer=tf.constant_initializer(0.0))
				
				self.Wc_1 = tf.get_variable('Wc_1', [self.text_dim, self.hidden3_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wc_2 = tf.get_variable('Wc_2', [self.hidden3_dim, self.hidden2_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wc_3 = tf.get_variable('Wc_3', [self.hidden2_dim, self.hidden1_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wc_4 = tf.get_variable('Wc_4', [self.hidden1_dim, self.image_dim],
										initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Bc_1 = tf.get_variable('Bc_1', [self.hidden3_dim], initializer=tf.constant_initializer(0.0))
				self.Bc_2 = tf.get_variable('Bc_2', [self.hidden2_dim], initializer=tf.constant_initializer(0.0))
				self.Bc_3 = tf.get_variable('Bc_3', [self.hidden1_dim], initializer=tf.constant_initializer(0.0))
				self.Bc_4 = tf.get_variable('Bc_4', [self.image_dim], initializer=tf.constant_initializer(0.0))
		
				self.Ws = tf.get_variable('Ws', [self.hidden2_dim, self.hidden2_dim],
																						initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Bs = tf.get_variable('Bs', [self.hidden2_dim], initializer=tf.constant_initializer(0.0))
		
				self.W = tf.get_variable('W', [self.hidden2_dim,self.class_dim])
				self.b = tf.get_variable('B', [self.class_dim])
			else:
				self.Wq_1 = tf.Variable(param[0])
				self.Wq_2 = tf.Variable(param[1])
				self.Wq_3 = tf.Variable(param[2])
				self.Wq_4 = tf.Variable(param[3])
				self.Bq_1 = tf.Variable(param[4])
				self.Bq_2 = tf.Variable(param[5])
				self.Bq_3 = tf.Variable(param[6])
				self.Bq_4 = tf.Variable(param[7])
				self.Wc_1 = tf.Variable(param[8])
				self.Wc_2 = tf.Variable(param[9])
				self.Wc_3 = tf.Variable(param[10])
				self.Wc_4 = tf.Variable(param[11])
				self.Bc_1 = tf.Variable(param[12])
				self.Bc_2 = tf.Variable(param[13])
				self.Bc_3 = tf.Variable(param[14])
				self.Bc_4 = tf.Variable(param[15])
				self.W = tf.Variable(param[16])	
				self.b = tf.Variable(param[17])
		
				self.params.append(self.Wq_1)
				self.params.append(self.Wq_2)
				self.params.append(self.Wq_3)
				self.params.append(self.Wq_4)
				self.params.append(self.Bq_1)
				self.params.append(self.Bq_2)
				self.params.append(self.Bq_3)
				self.params.append(self.Bq_4)
				self.params.append(self.Wc_1)
				self.params.append(self.Wc_2)
				self.params.append(self.Wc_3)
				self.params.append(self.Wc_4)
				self.params.append(self.Bc_1)
				self.params.append(self.Bc_2)
				self.params.append(self.Bc_3)
				self.params.append(self.Bc_4)
				self.params.append(self.W)
				self.params.append(self.b)

	#LSTM一路

		self.text_lstm_data_w2v = tf.nn.embedding_lookup(self.w2v_w, self.text_lstm_data)
		self.text_lstm_data_expand = tf.expand_dims(self.text_lstm_data_w2v, -1)
		self.conv_t = tf.nn.conv2d(
						self.text_lstm_data_expand,
						self.W_t_cnn,
						strides=[1, 1, 1, 1],
						padding="VALID",
						name="conv_t")
		self.h_t = tf.nn.relu(tf.nn.bias_add(self.conv_t, self.b_t_cnn), name="relu_t")
		self.pooled_t = tf.nn.max_pool(
						self.h_t,
						ksize=[1, 3, 1, 1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="pool_t")
		self.text_conv_data = tf.squeeze(self.pooled_t, [2])
		init_state_i = mlstm_cell_i.zero_state(self.batch_size, dtype=tf.float32)
		init_state_t = mlstm_cell_t.zero_state(self.batch_size, dtype=tf.float32)
		with tf.variable_scope('lstm_i'):
			self.lstm_output_i, state_i = tf.nn.dynamic_rnn(mlstm_cell_i, inputs=self.image_lstm_data, initial_state=init_state_i, time_major=False)
		with tf.variable_scope('lstm_t'):
			self.lstm_output_t, state_t = tf.nn.dynamic_rnn(mlstm_cell_t, inputs=self.text_conv_data, initial_state=init_state_t, time_major=False)
		#self.iv = tf.tanh(tf.tensordot(self.lstm_output_i, self.Wa_i, axes=1) + self.Ba_i)
		#self.iu = tf.tensordot(self.iv, self.Ua_i, axes=1, name='iu')
		self.iu = tf.tensordot(tf.tanh(self.lstm_output_i), self.Ua_i, axes=1, name='iu')
		self.ialpha = tf.nn.softmax(self.iu, name='ialphas')
		self.image_atten = tf.reduce_sum(self.lstm_output_i * tf.expand_dims(self.ialpha, -1), 1)
		self.image_residual = tf.reduce_mean(self.lstm_output_i,1)
		self.image_atten = tf.add(self.image_atten, self.image_residual)
		#self.tv = tf.tanh(tf.tensordot(self.lstm_output_t, self.Wa_t, axes=1) + self.Ba_t)
		#self.tu = tf.tensordot(self.tv, self.Ua_t, axes=1, name='tu')
		self.tu = tf.tensordot(tf.tanh(self.lstm_output_t), self.Ua_t, axes=1, name='tu')
		self.talpha = tf.nn.softmax(self.tu, name='talphas')
		self.text_atten = tf.reduce_sum(self.lstm_output_t * tf.expand_dims(self.talpha, -1), 1)
		self.text_residual = tf.reduce_mean(self.lstm_output_t, 1)
		self.text_atten = tf.add(self.text_atten, self.text_residual)

	#全连接一路

		self.image_data = tf.concat([self.image_whole_data, self.image_atten],1)
		self.text_data = tf.concat([self.text_whole_data, self.text_atten],1)


		self.image_rep1 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_data, self.Wq_1, self.Bq_1))#4096  to  3000
		self.image_rep2 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep1, self.Wq_2, self.Bq_2))#3000  to  2000
		self.image_reps = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep2, self.Ws, self.Bs))#2000  to  2000
		self.image_rep3 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_reps, self.Wq_3, self.Bq_3))#2000  to  1000
		self.image_rep4 = tf.nn.xw_plus_b(self.image_rep3,self.Wq_4,self.Bq_4)#1000 to 300
		self.image_sig  = tf.nn.sigmoid(self.image_rep4)
		self.image_norm = tf.sqrt(tf.reduce_sum(tf.square(self.image_sig),axis=1)+0.1)
		self.text_norm = tf.sqrt(tf.reduce_sum(tf.square(self.text_data),axis=1)+0.1)
		self.image_text = tf.reduce_sum(tf.multiply(self.image_sig, self.text_data), axis=1)
		self.image_distance = tf.log(0.5*(self.image_text/(self.image_norm*self.text_norm))+0.5)
		self.reward_image_prob = tf.nn.softmax(-self.image_distance)
		#self.image_distance = tf.reduce_sum(tf.square(self.image_sig-self.text_data),1)# 相同的语义的数据经转换后距离应该很短。
		#self.y_conv_img = tf.nn .xw_plus_b(self.image_rep2,self.W,self.b)   
		self.y_conv_img = tf.nn .xw_plus_b(self.image_reps,self.W,self.b)   
		self.img_prob = tf.nn.softmax(self.y_conv_img)
		self.text_rep1 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_data, self.Wc_1, self.Bc_1))#300  to  1000
		self.text_rep2 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep1, self.Wc_2, self.Bc_2))#1000  to  2000
		self.text_reps = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep2, self.Ws, self.Bs))#2000  to  2000
		self.text_rep3 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_reps, self.Wc_3, self.Bc_3))#2000  to  3000
		self.text_rep4 = tf.nn.xw_plus_b(self.text_rep3,self.Wc_4,self.Bc_4)#3000 to 4086
		self.text_sig  = tf.nn.sigmoid(self.text_rep4)
		self.image_norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.image_data),axis=1)+0.1)
		self.text_norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.text_sig),axis=1)+0.1)
		self.image_text2 = tf.reduce_sum(tf.multiply(self.image_data, self.text_sig), axis=1)
		self.text_distance = tf.log(0.5*(self.image_text2/(self.image_norm2*self.text_norm2))+0.5)
		self.reward_text_prob = tf.nn.softmax(-self.text_distance)
		#self.text_distance = tf.reduce_sum(tf.square(self.text_sig-self.image_data),1)# 相同的语义的数据经转换后距离应该很短。
		#self.y_conv_txt = tf.nn .xw_plus_b(self.text_rep2,self.W,self.b)
		self.y_conv_txt = tf.nn .xw_plus_b(self.text_reps,self.W,self.b)
		self.txt_prob = tf.nn.softmax(self.y_conv_txt)
		#图片转换为文本的特征表示后，再转换回去
		self.image_rep_3 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep4,self.Wc_1,self.Bc_1)) #300 to 1000
		self.image_rep_2 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep_3,self.Wc_2,self.Bc_2))#1000 to 2000
		self.image_rep_s = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep_2,self.Ws,self.Bs))#2000 to 2000
		self.image_rep_1 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep_s,self.Wc_3,self.Bc_3))#2000 to 3000
		self.image_rep_0 = tf.nn.relu(tf.nn.xw_plus_b(self.image_rep_1,self.Wc_4,self.Bc_4))
		#文本转换为图像的特征表示后，再转换回去
		self.text_rep_3 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep4,self.Wq_1,self.Bq_1)) #4096 to 3000
		self.text_rep_2 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep_3,self.Wq_2,self.Bq_2))#3000 to 2000
		self.text_rep_s = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep_2,self.Ws,self.Bs))#2000 to 2000
		self.text_rep_1 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep_s,self.Wq_3,self.Bq_3))#2000 to 1000
		self.text_rep_0 = tf.nn.relu(tf.nn.xw_plus_b(self.text_rep_1,self.Wq_4,self.Bq_4))#1000 to 300
		self.reward_image = tf.reduce_sum(tf.multiply(self.image_data,self.image_rep_0),axis=1)/(tf.sqrt(tf.reduce_sum(tf.square(self.image_data),axis=1)+0.1)*tf.sqrt(tf.reduce_sum(tf.square(self.image_rep_0),axis=1)))
		self.reward_image = tf.log(1.0/2*(1+self.reward_image))
		self.image_total_reward = self.h*self.image_distance+(1-self.h)*self.reward_image
		
		self.reward_text = tf.reduce_sum(tf.multiply(self.text_data,self.text_rep_0),axis=1)/(tf.sqrt(tf.reduce_sum(tf.square(self.text_data),axis=1)+0.1)*tf.sqrt(tf.reduce_sum(tf.square(self.text_rep_0),axis=1)))
		self.reward_text = tf.log(1.0/2*(1+self.reward_text))
		self.text_total_reward = self.h*self.text_distance + (1-self.h)*self.reward_text
		
		self.image_total_reward2 = tf.placeholder(tf.float32,shape=[None])
		self.text_total_reward2 = tf.placeholder(tf.float32,shape=[None])
		self.image_rep4_c = tf.placeholder(tf.float32,shape=[None,self.text_dim])
		self.text_rep4_c = tf.placeholder(tf.float32,shape=[None,self.image_dim])
		self.image_rep_3_c = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep4_c,self.Wc_1,self.Bc_1)) #300 to 1000
		self.image_rep_2_c = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep_3_c,self.Wc_2,self.Bc_2))#1000 to 2000
		self.image_rep_s_c = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep_2_c,self.Ws,self.Bs))#2000 to 2000
		self.image_rep_1_c = tf.nn.tanh(tf.nn.xw_plus_b(self.image_rep_s_c,self.Wc_3,self.Bc_3))#2000 to 3000
		self.image_rep_0_c = tf.nn.relu(tf.nn.xw_plus_b(self.image_rep_1_c,self.Wc_4,self.Bc_4))
		#文本转换为图像的特征表示后，再转换回去
		self.text_rep_3_c = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep4_c,self.Wq_1,self.Bq_1)) #4096 to 3000
		self.text_rep_2_c = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep_3_c,self.Wq_2,self.Bq_2))#3000 to 2000
		self.text_rep_s_c = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep_2_c,self.Ws,self.Bs))#2000 to 2000
		self.text_rep_1_c = tf.nn.tanh(tf.nn.xw_plus_b(self.text_rep_s_c,self.Wq_3,self.Bq_3))#2000 to 1000
		self.text_rep_0_c = tf.nn.relu(tf.nn.xw_plus_b(self.text_rep_1_c,self.Wq_4,self.Bq_4))#1000 to 300
		self.reward_image_c = tf.reduce_sum(tf.multiply(self.image_data,self.image_rep_0_c),axis=1)/(tf.sqrt(tf.reduce_sum(tf.square(self.image_data),axis=1)+0.1)*tf.sqrt(tf.reduce_sum(tf.square(self.image_rep_0_c),axis=1)))
		self.reward_image_c = tf.log(1.0/2*(1+self.reward_image_c))
		self.reward_text_c = tf.reduce_sum(tf.multiply(self.text_data,self.text_rep_0_c),axis=1)/(tf.sqrt(tf.reduce_sum(tf.square(self.text_data),axis=1)+0.1)*tf.sqrt(tf.reduce_sum(tf.square(self.text_rep_0_c),axis=1)))
		self.reward_text_c = tf.log(1.0/2*(1+self.reward_text_c))
		self.h_distance = tf.reduce_sum(tf.square(self.image_rep2-self.text_rep2),1)
		self.i2t_loss = 0.1*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv_img,labels=self.image_label))+\
						(1-self.h)*tf.reduce_mean(-self.reward_image_c)+0*tf.reduce_mean(tf.reduce_sum(tf.square(self.img_prob-self.txt_prob)))+\
						0*(tf.nn.l2_loss(self.W)+tf.nn.l2_loss(self.b))+0*tf.reduce_mean(self.image_distance)+self.h*tf.reduce_mean(-self.image_total_reward2*self.reward_image_prob)  
		self.t2i_loss = 0.1*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv_txt,labels=self.text_label))+\
						(1-self.h)*tf.reduce_mean(-self.reward_text_c)+0*tf.reduce_mean(tf.reduce_sum(tf.square(self.img_prob-self.txt_prob)))+\
						0*(tf.nn.l2_loss(self.W)+tf.nn.l2_loss(self.b))+0*tf.reduce_mean(self.text_distance)+self.h*tf.reduce_mean(-self.text_total_reward2*self.reward_text_prob)
		self.global_step = tf.Variable(0,trainable=False)
		self.lr_step = tf.train.exponential_decay(self.learning_rate, self.global_step, 20000, 0.96, staircase=True)
		self.optimizer = tf.train.GradientDescentOptimizer(0.1)
		self.global_step2 = tf.Variable(0,trainable=False)
		self.lr_step2 = tf.train.exponential_decay(self.learning_rate, self.global_step2, 20000, 0.96, staircase=True)
		self.optimizer2 = tf.train.GradientDescentOptimizer(0.1)
		#self.i2t_updates = self.optimizer.minimize(self.i2t_loss, var_list=self.params)
		#self.t2i_updates = self.optimizer2.minimize(self.t2i_loss, var_list=self.params)
		self.i2t_updates = self.optimizer.minimize(self.i2t_loss, var_list=tf.trainable_variables())
		self.t2i_updates = self.optimizer2.minimize(self.t2i_loss, var_list=tf.trainable_variables())

	def save_model(self,sess,output_path):
		if not os.path.exists(output_path):
			os.mkdir(output_path)    
		saver = tf.train.Saver()
		saver.save(sess, os.path.join(output_path, "model"+str(self.h)))
