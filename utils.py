# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:59:11 2017

@author: Administrator
"""
import pdb
import scipy.io as sio
import numpy as np
import h5py

def load_train_img():
	train_img_File1 = './data/train_img_1.mat'
	train_data1 = sio.loadmat(train_img_File1)
	train_img1=train_data1['train_img_sub']
	train_img_File2 = './data/train_img_2.mat'
	train_data2 = sio.loadmat(train_img_File2)
	train_img2=train_data2['train_img_sub']
	train_img_File3 = './data/train_img_3.mat'
	train_data3 = sio.loadmat(train_img_File3)
	train_img3=train_data3['train_img_sub']
	train_img_File4 = './data/train_img_4.mat'
	train_data4 = sio.loadmat(train_img_File4)
	train_img4=train_data4['train_img_sub']
	train_img= np.concatenate((train_img1, train_img2,train_img3,train_img4), axis = 0)
	train_img = train_img.reshape(-1,49,512)
	print(train_img.shape)
	return train_img
def load_test_img():
	test_img_File = './data/test_img.mat'
	test_data = sio.loadmat(test_img_File)
	test_img=test_data['test_img']
	test_img = test_img.reshape(-1,49,512)
	print(test_img.shape)
	return test_img

def load_train_txt():
	h = h5py.File('./data/XMN_txt.hdf5', 'r')
	train_txt = h['train'][:]-1
	return train_txt
def load_test_txt():
	h = h5py.File('./data/XMN_txt.hdf5', 'r')
	test_txt = h['test'][:]-1
	return test_txt

def is_same_cate(strA, strB, label_dim):
	labelA = strA.split()
	labelB = strB.split()
	
	if len(labelA) != label_dim  or len(labelB) != label_dim:
		print(strA)
		print(strB)
		pdb.set_trace()
	
	for i in range(label_dim):
		if (labelA[i] == '1' or labelA[i] == '1.0') and labelA[i] == labelB[i]:
			return True
	return False	

def push_query(query, url, dict):
	if query in dict:
		dict[query].append(url)
	else:
		dict[query] = [url]
	return dict
	


def make_train_dict(query_list, url_list,  query_label, url_label, label_dim):
	query_pair = {}
	query_num = len(query_list) - 1
	url_num = len(url_list) - 1
	
	for i in range(query_num):
		#if i%1000==0:
		#	print "make_train_dict: %d"%i
		query = query_list[i]
		push_query(query,url_list[i],query_pair)	
	return query_pair



def make_test_dict(query_list, url_list, query_label, url_label, label_dim):
	query_url = {}
	query_pos = {}
	query_num = len(query_list) - 1
	url_num = len(url_list) - 1
	
	for i in range(query_num):
		#if i%1000==0:
		#	print "make_test_dict: %d"%i
		query = query_list[i]
		left = 0 if i<40 else i-39
		right = i+1
		for k in range(left,right):
			if is_same_cate(query_label[i], url_label[k], label_dim):
				break
		for flag in range(k,k+20):
			try:
				url = url_list[flag]
			except:
				print (flag)
				pdb.set_trace()
			push_query(query,url,query_pos)
		for flag in range(0,url_num):
			url = url_list[flag]
			push_query(query,url,query_url)
	return query_url, query_pos	

def load_all_query_url(list_dir, label_dim):
	train_img = open(list_dir + 'train_img.txt', 'r').read().split('\n')
	test_img = open(list_dir + 'test_img.txt', 'r').read().split('\n')
	train_txt = open(list_dir + 'train_txt.txt', 'r').read().split('\n')
	test_txt = open(list_dir + 'test_txt.txt', 'r').read().split('\n')
	train_label = open(list_dir + 'train_label.txt', 'r').read().split('\n')
	test_label = open(list_dir + 'test_label.txt', 'r').read().split('\n')
	train_i2t_pair = make_train_dict(train_img, train_txt,  train_label, train_label, label_dim)
	train_t2i_pair = make_train_dict(train_txt, train_img,  train_label, train_label, label_dim)
	
	test_i2t, test_i2t_pos = make_test_dict(test_img, test_txt,test_label, test_label, label_dim)
	test_t2i, test_t2i_pos = make_test_dict(test_txt, test_img,test_label, test_label, label_dim)
	
	return train_img, test_img, train_txt, test_txt, train_i2t_pair, train_t2i_pair, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos

def load_all_lstm_feature_for_train(list_dir, feature_dir):
	feature_dict = {}
	train_img = load_train_img()
	train_txt = load_train_txt()
	for dataset in ['train']:
			for modal in ['img', 'txt']:
				list1 = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
					#feature1 = open(feature_dir + dataset + '_' + modal + '2.txt', 'r').read().split('\r\n')
					#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + ".mat")[dataset + '_' + modal]
					#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + "_LSTM.mat")[dataset + '_' + modal]
				if modal == 'img':
					feature2 = train_img
				else:
					feature2 = train_txt
				for i in range(len(list1) - 1):
						item = list1[i]
						#feature_string = feature1[i].split()
						#feature_float_list = []
						#for j in range(len(feature_string)):
						#       feature_float_list.append(float(feature_string[j]))
						#feature_dict[item] = feature_float_list + list(feature2[i])
						feature_dict[item] = list(feature2[i,:])
	return feature_dict
def load_all_lstm_feature_for_test(list_dir, feature_dir):
	feature_dict = {}
	test_img = load_test_img()
	test_txt = load_test_txt()
	for dataset in ['test']:
			for modal in ['img', 'txt']:
				list1 = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
					#feature1 = open(feature_dir + dataset + '_' + modal + '2.txt', 'r').read().split('\r\n')
					#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + ".mat")[dataset + '_' + modal]
					#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + "_LSTM.mat")[dataset + '_' + modal]
				if modal == 'img':
						feature2 = test_img
				else:
						feature2 = test_txt
				for i in range(len(list1) - 1):
					item = list1[i]
					#feature_string = feature1[i].split()
					#feature_float_list = []
					#for j in range(len(feature_string)):
					#       feature_float_list.append(float(feature_string[j]))
					#feature_dict[item] = feature_float_list + list(feature2[i])
					feature_dict[item] = list(feature2[i,:])
	return feature_dict

def load_all_feature_for_train(list_dir, feature_dir):
	feature_dict = {}
	for dataset in ['train']:
		for modal in ['img', 'txt']:
			list1 = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			feature1 = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + ".mat")[dataset + '_' + modal]
			#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + "_LSTM.mat")[dataset + '_' + modal]
			for i in range(len(list1) - 1):
				item = list1[i]
				feature_string = feature1[i].split()
				feature_float_list = []
				for j in range(len(feature_string)):
					feature_float_list.append(float(feature_string[j]))
				feature_dict[item] = feature_float_list
				#feature_dict[item] = list(feature2[i])
	return feature_dict
def load_all_feature_for_test(list_dir, feature_dir):
	feature_dict = {}
	for dataset in ['test']:
		for modal in ['img', 'txt']:
			list1 = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			feature1 = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + ".mat")[dataset + '_' + modal]
			#feature2 = sio.loadmat(feature_dir + dataset + '_' + modal + "_LSTM.mat")[dataset + '_' + modal]
			for i in range(len(list1) - 1):
				item = list1[i]
				feature_string = feature1[i].split()
				feature_float_list = []
				for j in range(len(feature_string)):
					feature_float_list.append(float(feature_string[j]))
				feature_dict[item] = feature_float_list
				#feature_dict[item] = list(feature2[i])
	return feature_dict
def load_all_label(list_dir):
	label_dict = {}
	for dataset in ['train',  'test']:
		for modal in ['img', 'txt']:
			list1 = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			#list1 = list(map(lambda x:x.strip(),open(list_dir + dataset + '_' + modal + '.txt', 'r').readlines()[:200]))
			label = open(list_dir + dataset + '_label.txt', 'r').read().split('\n')
			#label = list(map(lambda x:x.strip(),open(list_dir + dataset + '_label.txt', 'r').readlines()[:200]))
			for i in range(len(list1) - 1):
				item = list1[i]
				label_string = label[i].split()
				label_float_list = []
				for j in range(len(label_string)):
					label_float_list.append(float(label_string[j]))
				label_dict[item] = label_float_list
	return label_dict		
if __name__=='__main__':
    data = load_all_query_url("../data/list/",20)
    print (data[1])
