# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:15:07 2017

@author: Administrator
"""

import numpy as np
import scipy.io as sio
import scipy.spatial
from numpy.matlib import repmat
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
def save_mat(sess, model,feature_dict,label_dict):
	test_image=[]
	test_text=[]
	category=[]
	for item in feature_dict:
		if item.endswith("jpg"):
			number = item.split('.')[0]
			#print sess.run(model.img_prob, feed_dict={model.image_data:np.asarray(feature_dict[number+".jpg"]).reshape(-1,4096)})
			test_image.append(sess.run(model.img_prob, feed_dict={model.image_data:np.asarray(feature_dict[number+".jpg"]).reshape(-1,4096)})[0])
			test_text.append(sess.run(model.txt_prob, feed_dict={model.text_data:np.asarray(feature_dict[number+".txt"]).reshape(-1,300)})[0])
			category.append(label_dict[item].index(1.0)+1)
		else:
			continue
	sio.savemat("savedata.mat",{"test_image":test_image,"test_text":test_text,"category":category})

def save_mat_new(sess, model,feature_dict,feature_lstm_dict,label_dict,test_img):
        test_image=[]
        test_text=[]
        category=[]
        for item in test_img:
                if item.endswith("jpg"):
                        number = item.split('.')[0]
                        #print sess.run(model.img_prob, feed_dict={model.image_data:np.asarray(feature_dict[number+".jpg"]).reshape(-1,4096)})
                        test_image.append(sess.run(model.img_prob, feed_dict={model.image_lstm_data:np.asarray(feature_lstm_dict[number+".jpg"]).reshape(-1,49,512),model.image_whole_data:np.asarray(feature_dict[number+".jpg"]).reshape(-1,4096),model.batch_size:1})[0])
                        test_text.append(sess.run(model.txt_prob, feed_dict={model.text_lstm_data:np.asarray(feature_lstm_dict[number+".txt"]).reshape(-1,857),model.text_whole_data:np.asarray(feature_dict[number+".txt"]).reshape(-1,300),model.batch_size:1})[0])
                        category.append(label_dict[item].index(1.0)+1)
                else:
                        continue
        sio.savemat("savedata.mat",{"test_image":test_image,"test_text":test_text,"category":category})

def MAP(sess, model, query_pos_test, query_index_url_test, feature_dict, feature_lstm_dict, label_dict, flag):
	rs = []
	distance_dict = {}
	#print len(query_pos_test)
	for item in feature_dict:
		input_data = np.asarray(feature_dict[item])
		input_data_dim = input_data.shape[0]
		input_data = input_data.reshape(1, input_data_dim)
		input_lstm_data = np.asarray(feature_lstm_dict[item])
		
		if item.split('.')[-1] == 'jpg':
			input_lstm_data = input_lstm_data.reshape(1, input_lstm_data.shape[0], input_lstm_data.shape[1])
			distance = sess.run(model.img_prob, feed_dict={model.image_lstm_data: input_lstm_data,model.image_whole_data: input_data,model.batch_size: 1})
			
		elif item.split('.')[-1] == 'txt':
			input_lstm_data = input_lstm_data.reshape(1, input_lstm_data.shape[0])
			distance = sess.run(model.txt_prob, feed_dict={model.text_lstm_data: input_lstm_data,model.text_whole_data: input_data,model.batch_size: 1})
			
		distance_dict[item] = distance

	for query in query_pos_test.keys():
		pos_set = set(query_pos_test[query])
		pred_list = query_index_url_test[query]
		
		pred_list_score = []
		query_distance = distance_dict[query]
		for candidate in pred_list:
			candidate_distance = distance_dict[candidate]
			#pred_list_score.append(np.mean(np.square(query_distance-candidate_distance)))
			pred_list_score.append(np.sum(query_distance*candidate_distance)/(np.sqrt(np.sum(np.square(query_distance))*np.sqrt(np.sum(np.square(candidate_distance))))))
			#pred_list_score.append(np.corrcoef(query_distance,candidate_distance)[0,1])
		pred_url_score = zip(pred_list, pred_list_score)
		pred_url_score = sorted(pred_url_score, key=lambda x: x[1],reverse=True)#注意是否需要加reverse
		#print pred_url_score
		r = [0.0] * len(pred_list_score)
		for i in range(0, len(pred_list_score)):
			(url, score) = pred_url_score[i]
			if url in pos_set:
				r[i] = 1.0
		rs.append(r)
	return np.mean([average_precision(r) for r in rs])

def MAP_new(sess, model, query_pos_test, query_index_url_test, feature_dict, feature_lstm_dict, label_dict, best_map):
	dic = {}
	dic = np.zeros((3,))
	distance_image = []
	distance_text = []
	label = []
	BATCH_SIZE = 200
	'''
	indices = sio.loadmat('indice.mat')
	indices = indices['indice']
	indices = indices.reshape(4000,)-1	
	'''
	test_size = len(query_pos_test)-1
	index = 0
	while index < test_size:
		input_data_img = []
		input_lstm_data_img = []
		input_data_txt = []
		input_lstm_data_txt = []
		if index + BATCH_SIZE <= test_size:
			for i in range(index, index + BATCH_SIZE):
				query_img = query_pos_test[i]
				input_data_img.append(feature_dict[query_img])
				input_lstm_data_img.append(feature_lstm_dict[query_img])
				query_txt = query_index_url_test[i]
				input_data_txt.append(feature_dict[query_txt])
				input_lstm_data_txt.append(feature_lstm_dict[query_txt])
				label.append(label_dict[query_img].index(1.0)+1)
			real_batchsize = BATCH_SIZE
		else:
			for i in range(index, test_size):
				query_img = query_pos_test[i]
				input_data_img.append(feature_dict[query_img])
				input_lstm_data_img.append(feature_lstm_dict[query_img])
				query_txt = query_index_url_test[i]
				input_data_txt.append(feature_dict[query_txt])
				input_lstm_data_txt.append(feature_lstm_dict[query_txt])
				label.append(label_dict[query_img].index(1.0)+1)
			real_batchsize = test_size-index
		index += BATCH_SIZE
		input_data_img = np.asarray(input_data_img)
		input_lstm_data_img = np.asarray(input_lstm_data_img)
		input_data_txt = np.asarray(input_data_txt)
		input_lstm_data_txt = np.asarray(input_lstm_data_txt)
		distance_img,distance_txt = sess.run([model.img_prob,model.txt_prob], feed_dict={model.image_lstm_data: input_lstm_data_img,model.image_whole_data: input_data_img,model.text_lstm_data: input_lstm_data_txt,model.text_whole_data: input_data_txt,model.batch_size: real_batchsize,model.keep_prob: 1.0})
		#distance_txt = sess.run(model.txt_prob, feed_dict={model.text_lstm_data: input_lstm_data_txt,model.text_whole_data: input_data_txt,model.batch_size: real_batchsize})
		distance_image.extend(distance_img)
		distance_text.extend(distance_txt)

	distance_image = np.array(distance_image)
	distance_text = np.array(distance_text)
	#print(distance_image.shape)
	#distance_image = distance_image[indices]
	#distance_text = distance_text[indices]
	#distance_image = znorm(distance_image)
	#distance_text = znorm(distance_text)

	dic[0] = fx_calc_map_label(distance_image, distance_text, label, k=0, dist_method='COS')
	dic[1] = fx_calc_map_label(distance_text, distance_image, label, k=0, dist_method='COS')
	dic[2] = (dic[0]+dic[1]) / 2

	print(dic)
	if dic[2] > best_map:
		sio.savemat("savedata.mat",{"test_image":distance_image,"test_text":distance_text,"category":label})

	return dic

def znorm(inMat):
  col=inMat.shape[0]
  row=inMat.shape[1]
  mean_val=np.mean(inMat, axis=0)
  std_val=np.std(inMat, axis=0)
  mean_val=repmat(mean_val, col, 1)
  std_val=repmat(std_val, col, 1)
  x = np.argwhere(std_val==0)
  for y in x:
    std_val[y[0],y[1]]=1
  return (inMat-mean_val)/std_val

def fx_calc_map_label(image, text, label, k = 0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

