# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:12:40 2017

@author: Administrator
"""

import pickle, random, pdb, time #pdb用于调试
import tensorflow as tf
import numpy as np
import utils as ut
from MAP import *
from i2t_model import *
import os
import pandas as pd
import h5py
'''
import xlsxwriter
filename = xlsxwriter.Workbook("result_wiki.xlsx")
sheet = filename.add_worksheet()
sheet.write(0,0,"reward1_text")
sheet.write(0,1,"reward2_text")
FLAG = 1 
FLAG2 = 1 
sheet2 = filename.add_worksheet()
sheet2.write(0,0,"reward1_image")
sheet2.write(0,1,"reward2_image")
'''

GPU_ID = 2
OUTPUT_DIM = 64

SELECTNUM = 2
SAMPLERATIO = 20

#WHOLE_EPOCH = 1
WHOLE_EPOCH = 10
D_EPOCH = 1
G_EPOCH = 2
GS_EPOCH = 30
#D_DISPLAY = 1
D_DISPLAY = 1
G_DISPLAY = 1

#IMAGE_DIM = 4352
#TEXT_DIM = 556
#IMAGE_DIM = 1024
#TEXT_DIM = 1024
IMAGE_DIM = 4096
TEXT_DIM = 300
IMAGE_LSTM_LEN = 49
TEXT_LSTM_LEN = 857
IMAGE_LSTM_DIM = 512
TEXT_LSTM_DIM = 300
ATTENTION_DIM = 512
HIDDEN1_DIM = 3000
HIDDEN2_DIM = 2000
HIDDEN3_DIM = 1000
CLASS_DIM = 200
BATCH_SIZE = 200
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.1
#LEARNING_RATE = 0.0001
#BETA = 1
BETA = 0
GAMMA = 0.1
REWARD_BASE = 10.0
PARAM_FILE = "param.pkl"

WORKDIR = '../XMN/'

train_img, test_img, train_txt, test_txt, train_i2t_pair, train_t2i_pair, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos = ut.load_all_query_url(WORKDIR + 'list/', CLASS_DIM)
feature_dict = ut.load_all_feature_for_train(WORKDIR + 'list/', WORKDIR + 'feature/')
feature_dict_test = ut.load_all_feature_for_test(WORKDIR + 'list/', WORKDIR + 'feature/')
feature_lstm_dict = ut.load_all_lstm_feature_for_train(WORKDIR + 'list/', WORKDIR + 'feature/')
feature_lstm_dict_test = ut.load_all_lstm_feature_for_test(WORKDIR + 'list/', WORKDIR + 'feature/')
label_dict = ut.load_all_label(WORKDIR + 'list/')#{'422030830.txt': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
record_file = open('record.txt', 'w')
record_file.close()

w2v_f = h5py.File('./data/XMN_txt.hdf5','r')
w2v_w = w2v_f['w2v'][:]

'''
def generate_samples(train_pos,train_neg):
	data = []
	for query in train_pos:
		pos_list = train_pos[query]
		neg_list = train_neg[query]	
		random.shuffle(pos_list)
		random.shuffle(neg_list)
		#sample_size = int(len(pos_list) / SAMPLERATIO)
		sample_size = 1
		pos_list = pos_list[0 : sample_size]#表示选取候选的1/20
		neg_list = neg_list[0 : sample_size]#表示选取候选的1/20  
		for i in range(sample_size):
			data.append((query,pos_list[i],neg_list[i]))		
	random.shuffle(data)
	return data
'''

def generate_samples(train_pair):
	data = []
	for query in train_pair:
		pair_list = train_pair[query]
		#sample_size = int(len(pos_list) / SAMPLERATIO)
		sample_size = 1
		pair_list = pair_list[0 : sample_size]#表示选取候选的1/20
		for i in range(sample_size):
			data.append((query,pair_list[i]))		
	random.shuffle(data)
	return data

def train_model(sess, model, train_list, flag):
	train_size = len(train_list)
	index = 0
	while index < train_size:
		input_query = []
		input_lstm_query = []
		input_pos = []
		input_lstm_pos = []
		input_neg = []
		input_lstm_neg = []
		input_label = []
		accuracys = []
		distances = []
		h_distances = []
		if index + BATCH_SIZE <= train_size:
			for i in range(index, index + BATCH_SIZE):
				query, pos = train_list[i]
				input_query.append(feature_dict[query])
				input_pos.append(feature_dict[pos])
				input_lstm_query.append(feature_lstm_dict[query])
				input_lstm_pos.append(feature_lstm_dict[pos])
				input_label.append(label_dict[query])
			real_batchsize = BATCH_SIZE
		else:
			for i in range(index, train_size):
				query, pos = train_list[i]
				input_query.append(feature_dict[query])
				input_pos.append(feature_dict[pos])
				input_lstm_query.append(feature_lstm_dict[query])
				input_lstm_pos.append(feature_lstm_dict[pos])
				input_label.append(label_dict[query])
			real_batchsize = train_size-index
		index += BATCH_SIZE
		query_data = np.asarray(input_query)
		input_pos = np.asarray(input_pos)
		query_lstm_data = np.asarray(input_lstm_query)
		input_lstm_pos = np.asarray(input_lstm_pos)
		input_label = np.asarray(input_label)
		if flag == 'i2t':
			image_total_reward,image_rep4,distance,h_distance= sess.run([model.image_total_reward,model.image_rep4,model.image_distance,model.h_distance],
						 feed_dict={model.image_lstm_data:query_lstm_data,
									model.image_whole_data:query_data,
									model.text_lstm_data: input_lstm_pos,
									model.text_whole_data: input_pos,
									model.image_label: input_label.reshape(-1,200),
									model.batch_size: real_batchsize,
									model.keep_prob: 0.5})
			#sess.run(model.image_update,feed_dict={model.image_data:query_data,model.text_data:input_pos})
			global FLAG2
			#sheet2.write(FLAG2,0,sess.run(tf.reduce_mean(model.image_distance),feed_dict = {model.image_lstm_data:query_lstm_data,model.image_whole_data:query_data,model.text_lstm_data:input_lstm_pos,model.text_whole_data:input_pos,model.batch_size: real_batchsize}))
			#sheet2.write(FLAG2,1,sess.run(tf.reduce_mean(model.reward_image),feed_dict = {model.image_lstm_data:query_lstm_data,model.image_whole_data:query_data,model.text_lstm_data:input_lstm_pos,model.text_whole_data:input_pos,model.batch_size: real_batchsize}))
			#FLAG2+=1
			#if FLAG2%100==0:
			#	print "image"
			#	print sess.run(tf.reduce_mean(model.reward_image),feed_dict = {model.image_whole_data:query_data,model.image_lstm_data:query_lstm_data,model.text_whole_data:input_pos,model.text_lstm_data:input_lstm_pos})
			sess.run(model.i2t_updates,feed_dict={model.image_lstm_data:query_lstm_data,
									model.image_whole_data:query_data,
									model.text_lstm_data: input_lstm_pos,
									model.text_whole_data: input_pos,
									model.image_label: input_label.reshape(-1,200),
									model.image_total_reward2:image_total_reward,
									model.image_rep4_c:image_rep4,
									model.batch_size: real_batchsize,
									model.keep_prob: 0.5})
			#print sess.run(model.image_distanc)
			distances.append(np.mean(distance))
			h_distances.append(np.mean(h_distance))
		elif flag == 't2i':
			text_total_reward,text_rep4,distance,h_distance= sess.run([model.text_total_reward,model.text_rep4,model.text_distance,model.h_distance],
						 feed_dict={model.text_lstm_data: query_lstm_data,
									model.text_whole_data: query_data,
									model.image_lstm_data: input_lstm_pos,
									model.image_whole_data: input_pos,
									model.text_label: input_label.reshape(-1,200),
									model.batch_size: real_batchsize,
									model.keep_prob: 0.5})
			#sess.run(model.text_update,feed_dict={model.text_data:query_data,model.image_data:input_pos})	
			global FLAG
			#sheet.write(FLAG,0,sess.run(tf.reduce_mean(model.text_distance),feed_dict = {model.text_lstm_data:query_lstm_data,model.text_whole_data:query_data,model.image_lstm_data:input_lstm_pos,model.image_whole_data:input_pos,model.batch_size: real_batchsize}))
			#sheet.write(FLAG,1,sess.run(tf.reduce_mean(model.reward_text),feed_dict = {model.text_lstm_data:query_lstm_data,model.text_whole_data:query_data,model.image_lstm_data:input_lstm_pos,model.image_whole_data:input_pos,model.batch_size: real_batchsize}))
			#FLAG+=1
			#if FLAG%100==0:
			#	print "text"
			#	print sess.run(tf.reduce_mean(model.reward_text),feed_dict = {model.text_whole_data:query_data,model.text_lstm_data:query_lstm_data,model.image_whole_data:input_pos,model.image_lstm_data:input_lstm_pos})
			sess.run(model.t2i_updates,feed_dict={model.text_lstm_data: query_lstm_data,
									model.text_whole_data: query_data,
									model.image_lstm_data: input_lstm_pos,
									model.image_whole_data: input_pos,
									model.text_label: input_label.reshape(-1,200),
									model.text_total_reward2:text_total_reward,
									model.text_rep4_c:text_rep4,
									model.batch_size: real_batchsize,
									model.keep_prob: 0.5})
			distances.append(np.mean(distance))
			#print sess.run(model.text_distance,feed_dict={model.text_data:query_data,model.image_data:input_pos})
			#pdb.set_trace()
			if pd.isnull(np.mean(distance)):
				print (query_data.shape )
				print (query_data.dtype )   
				print (input_pos.shape  )
				pdb.set_trace()
			h_distances.append(np.mean(h_distance))
			#print np.mean(h_distance)
	print ("distances:%s"%distances        )
	print ("h_distances:%s"%h_distances    )
	if flag == 'i2t':
		print ("i2t_distance: %f"%np.mean(distances) )
		print ("h_distance: %f"%np.mean(h_distances) )
	elif flag == 't2i':
		print ("t2i_distance: %f"%np.mean(distances) )
		print ("h_distance: %f"%np.mean(h_distances) )
	return model
def main():
	with tf.device('/gpu:' + str(GPU_ID)):
		param = None
		output_path = "./"
		if os.path.exists(output_path+"ha"):
			model = I2t(IMAGE_DIM, IMAGE_LSTM_LEN, IMAGE_LSTM_DIM, TEXT_DIM, TEXT_LSTM_LEN, TEXT_LSTM_DIM, ATTENTION_DIM, HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM,CLASS_DIM, WEIGHT_DECAY, LEARNING_RATE, REWARD_BASE, BETA, GAMMA, w2v_w, param)
			init = tf.global_variables_initializer()
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			sess = tf.Session(config=config)
			saver = tf.train.Saver()
			sess.run(init)
			saver.restore(sess, output_path + "/model")
		else:
			model = I2t(IMAGE_DIM, IMAGE_LSTM_LEN, IMAGE_LSTM_DIM, TEXT_DIM, TEXT_LSTM_LEN, TEXT_LSTM_DIM, ATTENTION_DIM, HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM,CLASS_DIM, WEIGHT_DECAY, LEARNING_RATE, REWARD_BASE, BETA, GAMMA, w2v_w, param)
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			sess = tf.Session(config=config)
			sess.run(tf.global_variables_initializer())
		print ('start training')
		#map_best_val = (MAP(sess, model, test_i2t_pos, test_i2t, feature_dict_test, feature_lstm_dict_test, label_dict,'i2t') + MAP(sess, model, test_t2i_pos, test_t2i, feature_dict_test, feature_lstm_dict_test, label_dict,'t2i'))/2
		#print "current best accuracy:%s"%map_best_val
		map_best_val = MAP_new(sess, model, test_img, test_txt, feature_dict_test, feature_lstm_dict_test, label_dict,0)
		map_best_val = map_best_val[2]
		print ("current best accuracy:%s"%map_best_val)
		#save_mat(sess , model, feature_dict_test, label_dict)
		result = [[0,0],[0,0],[0,0,0]]#第一个元素表示i2t最大值，第二个元素表示t2i最大值，第三个元素表示平均值最大值。
		for epoch in range(WHOLE_EPOCH):
			print ('epoch: ' + str(epoch))
			#train_list_img = generate_samples(train_i2t_pair, train_i2t_neg)
			#train_list_txt = generate_samples(train_t2i_pair, train_t2i_neg)
			train_list_img = generate_samples(train_i2t_pair)
			train_list_txt = generate_samples(train_t2i_pair)
			model = train_model(sess, model, train_list_txt, 't2i')
			model = train_model(sess, model, train_list_img, 'i2t')
			if (epoch + 1) % (D_DISPLAY) == 0:
				'''
				i2t_test_map = MAP(sess, model, test_i2t_pos, test_i2t, feature_dict_test, feature_lstm_dict_test, label_dict,'i2t')
				print 'E%d I2T_Test_MAP: %.4f' % (epoch,  i2t_test_map)
				t2i_test_map = MAP(sess, model, test_t2i_pos, test_t2i, feature_dict_test, feature_lstm_dict_test, label_dict,'t2i')
				print 'E%d T2I_Test_MAP: %.4f' % (epoch,  t2i_test_map)
				'''
				test_map =  MAP_new(sess, model, test_img, test_txt, feature_dict_test, feature_lstm_dict_test, label_dict,map_best_val)
				i2t_test_map = test_map[0]
				t2i_test_map = test_map[1]
				print ('E%d I2T_Test_MAP: %.4f, %.4f' % (epoch,  i2t_test_map, t2i_test_map))
				if i2t_test_map>result[0][0]:
					result[0]=[i2t_test_map,t2i_test_map]
				if t2i_test_map>result[1][1]:
					result[1]=[i2t_test_map,t2i_test_map]
				if (t2i_test_map+i2t_test_map)/2>result[2][0]:
					result[2][0]=(t2i_test_map+i2t_test_map)/2
					result[2][1]=i2t_test_map
					result[2][2]=t2i_test_map
				with open('record.txt', 'a') as record_file:
					record_file.write('E%d I2T_Test_MAP: %.4f\n' % (epoch, i2t_test_map))
					record_file.write('E%d T2I_Test_MAP: %.4f\n' % (epoch, t2i_test_map))
					
				average_map = 0.5 * (i2t_test_map + t2i_test_map)
				if average_map > map_best_val:
					map_best_val = average_map
					#model.save_model(sess,output_path)
					print ("current best accuracy:%s"%map_best_val)
					#if epoch>10:
						#save_mat_new(sess, model, feature_dict_test, feature_lstm_dict_test, label_dict, test_img)

			'''
			img_feature = []
			txt_feature = []
			for i in range(1801,1901):
				name_img = str(i) + '.jpg'
				name_txt = str(i) + '.txt'
				img_feature.append(sess.run(model.img_prob,feed_dict={model.image_data: np.array(feature_dict_test[name_img]).reshape(-1,4096),
																		model.image_label: np.asarray(label_dict[name_img]).reshape(-1,20)})[0])
				txt_feature.append(sess.run(model.txt_prob,feed_dict={model.text_data: np.array(feature_dict_test[name_txt]).reshape(-1,300),
																		model.text_label: np.asarray(label_dict[name_txt]).reshape(-1,20)})[0])
			pd.DataFrame(img_feature).to_excel('img_feature'+str(epoch)+'.xlsx')
			pd.DataFrame(txt_feature).to_excel('txt_feature'+str(epoch)+'.xlsx')
			'''
		#save_mat_new(sess, model, feature_dict_test, feature_lstm_dict_test, label_dict, test_img)
		print (result)
	sess.close()
if __name__ == '__main__':
	main()
