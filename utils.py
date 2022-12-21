from sklearn import datasets
import pandas as pd

from sklearn.cluster import DBSCAN
import matplotlib.pyplot  as plt
import seaborn as sns


import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

import random
from collections import defaultdict


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.CNN4Backbone(
            #hidden=hid_dim,
            channels=x_dim,
            max_pool=True,
       )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        #models = DBSCAN(eps = 6 , min_samples= 2)
        #x = models.fit_predict(x )

        return x.view(x.size(0), -1)
        #return x


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def pairwise_distances_logits(a, b):
	n = a.shape[0]
	m = b.shape[0]
	logits = -((a.unsqueeze(1).expand(n, m, -1) -
					b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
	return logits


def accuracy(predictions, targets):
	predictions = predictions.argmax(dim=1).view(targets.shape)
	return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):
	
	def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
		super().__init__()
		self.encoder = l2l.vision.models.CNN4Backbone(
			# hidden=hid_dim,
			channels=x_dim,
			max_pool=True,
		)
		self.out_channels = 1600
	
	def forward(self, x):
		x = self.encoder(x)
		# models = DBSCAN(eps = 6 , min_samples= 2)
		# x = models.fit_predict(x )
		
		return x.view(x.size(0), -1)
	# return x


def adapt_step1(model, batch, ways, shot, query_num, metric=None, device=None, clustering=None):
	data, labels = batch
	data = data.to(device)
	labels = labels.to(device)
	n_items = shot * ways
	
	# Sort data samples by labels
	# TODO: 오류 잡아서 continue로 넘어가기
	sort = torch.sort(labels)
	data = data.squeeze(0)[sort.indices].squeeze(0)
	labels = labels.squeeze(0)[sort.indices].squeeze(0)
	
	# Compute support and query embeddings
	embeddings = model(data)
	selection = np.arange(ways) * (shot + query_num)
	support_indices = np.zeros(data.size(0), dtype=bool)
	
	return embeddings, data, labels, selection, support_indices


def cal_cluster(embeddings, clustering=None, fixed_eps=6):
	if clustering == "dbscan_fix":
		models = DBSCAN(eps=fixed_eps, min_samples=2)
		features = embeddings.cpu().detach().numpy()
		predict = models.fit_predict(features)
		cls = len(np.unique(predict))
		# print(cls)
		# predict_onehot = np.eye(cls)[predict]
		# res_tmp = embeddings.shape[0]
		# embeddings= torch.cat([embeddings, torch.tensor(predict_onehot).to(device).reshape(res_tmp,-1) ],dim = 1)
		return predict, cls
	
	
	if clustering == "dbscan_tune":
		features = embeddings.cpu().detach().numpy()
		best_eps = 6
		best_cls = 0
		for i in range(1, 11):
			models = DBSCAN(eps=i, min_samples=2)
			predict = models.fit_predict(features)
			cls = len(np.unique(predict))
			# print(cls)
			if cls > best_cls:
				best_cls = cls
				best_eps = i
				best_predict = predict
		
		# print(best_cls)
		return best_predict, best_cls
	# predict_onehot = np.eye(best_cls)[best_predict]
	# res_tmp  = embeddings.shape[0]
	# embeddings= torch.cat([embeddings, torch.tensor(predict_onehot).to(device).reshape(res_tmp,-1) ],dim = 1)


def check_support(shot, predict, selection, support_indices):
	non_minus_idx = np.where(predict != -1)
	point_idx = non_minus_idx[0]
	support_list = []
	unit = selection[1] - selection[0]
	for board in selection:
		
		tmp = point_idx[point_idx >= board]
		tmp2 = tmp[tmp < (board + unit)]
		if len(tmp2) < shot:
			raise Exception(f'insufficient value error {len(tmp2)} is samller than support shot {shot}')
		else:
			tmp2 = tmp2[:5]
			support_indices[tmp2] = True
	
	return support_indices


def concat_cluster_feature(embeddings, predict, cls, device):
	predict_onehot = np.eye(cls)[predict]
	res_tmp = embeddings.shape[0]
	embeddings = torch.cat([embeddings, torch.tensor(predict_onehot).to(device).reshape(res_tmp, -1)], dim=1)
	return embeddings


def adapt_step2(support_indices, ways, shot, query_num, embeddings, labels):
	query_indices = torch.from_numpy(~support_indices)
	support_indices = torch.from_numpy(support_indices)
	support = embeddings[support_indices]
	support = support.reshape(ways, shot, -1).mean(dim=1)
	query = embeddings[query_indices]
	labels = labels[query_indices].long()
	
	logits = pairwise_distances_logits(query, support)
	loss = F.cross_entropy(logits, labels)
	acc = accuracy(logits, labels)
	return loss, acc


def slow_adapt(model, batch, ways, shot, query_num, metric=None, device=None, fixed_eps=6, clustering=None,
					filtering=False, concating=False):
	if metric is None:
		metric = pairwise_distances_logits
	
	if device is None:
		device = model.device()
	
	# make embedding output
	embeddings, data, labels, selection, support_indices = adapt_step1(model,
																							 batch,
																							 ways,
																							 shot,
																							 query_num,
																							 metric,
																							 device)
	
	# clustering : calculating cluster using method you select
	if clustering != None:
		predict, cls = cal_cluster(embeddings,
											clustering=clustering,
											fixed_eps=fixed_eps)
	
	# filtering : delete noisy points
	if filtering == True:
		#######################
		try:
			support_indices = check_support(shot,
													  predict,
													  selection,
													  support_indices)
		# print("제대로 안되네")
		# print(support_indices)
		except Exception as e:
			print(e)
			# print("제대로 되나?")
			
			return _, _, True
		########################
	
	else:  # it there is no filtering, use episode' samples 100%
		for offset in range(shot):
			support_indices[selection + offset] = True
	
	# concating : new_embeddings = embeddings + cluster_predict_features
	if concating == True:
		embeddings = concat_cluster_feature(embeddings,
														predict,
														cls,
														device)
	
	loss, acc = adapt_step2(support_indices,
									ways,
									shot,
									query_num,
									embeddings,
									labels)
	
	return loss, acc, False

