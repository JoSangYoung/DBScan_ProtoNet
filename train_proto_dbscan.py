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

from utils import *
from dataset import *

from tensorboardX import SummaryWriter
import time
import os
import datetime


def launch_tensor_board(log_path, port, host):  # asdf
   os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
   return True

#TODO Base는 돌릴 수가 없게 되어잇네 그것도 돌아가게 해보자 !

def parser_args():
	parser = argparse.ArgumentParser()
	
	#1217
	# sd
	parser.add_argument('--sd', help='default)  ', type=int, default=1)
	
	
	
	# iteration
	parser.add_argument('--num_iterations', dest='num_iterations', type=int, default=20, help='Number of train episodes')
	# gpu
	parser.add_argument('--gpu_number', help='default) 0', default=0, dest='GPU_NUM', type=int)
	# exp name
	parser.add_argument('--exp_name', help='default) "EXP_JH" ', default="EXP_0", dest='exp_name', type=str)
	# file name (exp-detail)
	parser.add_argument('--file_name', help='default) "EXP_0" ', default="Base", dest='file_name', type=str)
	# file name (exp-detail)
	parser.add_argument('--dataset', help='default) ', default="cifarfs", dest='dataset', type=str)
	
	#shot
	parser.add_argument('--shot', dest='shot', type=int, default=5, help='shot')
	#train_way
	parser.add_argument('--train_way', dest='train_way', type=int, default=5, help='train_way')
	#train_query
	parser.add_argument('--train_query', dest='train_query', type=int, default=15, help='train_query')
	
	# test_shot
	parser.add_argument('--test_shot', dest='test_shot', type=int, default=5, help='shot')
	# test_way
	parser.add_argument('--test_way', dest='test_way', type=int, default=5, help='train_way')
	# test_query
	parser.add_argument('--test_query', dest='test_query', type=int, default=15, help='train_query')
	
	#Option
	# step
	parser.add_argument('--step', dest='step', type=int, default=1, help='step')
	# fixed_eps
	parser.add_argument('--fixed_eps', dest='fixed_eps', type=int, default=6, help='fixed_eps')

	# clustering
	parser.add_argument('--clustering', help='default)  ', default=None, dest='clustering', type=str)
	# filtering
	parser.add_argument('--filtering', help='default)  ', default=False, dest='filtering', type=str)
	# concating
	parser.add_argument('--concating', help='default)  ', default=False, dest='concating', type=str)
	# loss_eps
	parser.add_argument('--loss_eps', dest='loss_eps', default=False, type=str)
	
	args = parser.parse_args()
	
	return args


def main():
	cuda = True
	seed = 42
	#if sd == True :
	# config option
	args = parser_args()
	evaluation_unit = 2500
	evaluation_roud = 301
	#evaluation_unit = 100
	#evaluation_roud = 21
	
	GPU_NUM = args.GPU_NUM
	sd_list = []
	for seed in range(args.sd):
		best_valid_acc = 0
		before_valid_acc = 0
		best_test_acc = 0
		#before_test_acc = 0
		patience = 0
	
		#Set CUDA
	 
		random.seed(seed)
		np.random.seed(seed)
		#print(seed)
		torch.manual_seed(seed)
		#device = torch.device('cpu')
		if cuda and torch.cuda.device_count():
		  torch.cuda.manual_seed(seed)
		  device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
		print('Current cuda device ', torch.cuda.current_device())
		
		import timeit
		start_time = timeit.default_timer()  # 시작 시간 체크
		
		
		###dataset
		transf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
		path_data = "./data"
		
		if args.dataset == "cifarfs":
			train_dataset = l2l.vision.datasets.CIFARFS(
				 root=path_data, mode='train',download=True,transform = transf)
			
			valid_dataset = l2l.vision.datasets.CIFARFS(
				 root=path_data, mode='validation',download=True,transform = transf)
			
			test_dataset = l2l.vision.datasets.CIFARFS(
				 root=path_data, mode='test',download=True,transform = transf)
		elif args.dataset == "mini":
			valid_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='validation', download=True)
			test_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='test', download=True)
			train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train', download=False)
			
		
		#train
		train_dataset = l2l.data.MetaDataset(train_dataset)
		train_transforms = [
			 NWays(train_dataset, args.train_way),
			 KShots(train_dataset, args.train_query + args.shot),
			 LoadData(train_dataset),
			 RemapLabels(train_dataset),
		]
		train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
		train_loader = DataLoader(train_tasks , pin_memory=True, shuffle=True)
		#valid
		valid_dataset = l2l.data.MetaDataset(valid_dataset)
		valid_transforms = [
			 NWays(valid_dataset, args.test_way),
			 KShots(valid_dataset, args.test_query + args.test_shot),
			 LoadData(valid_dataset),
			 RemapLabels(valid_dataset),
		]
		valid_tasks = l2l.data.TaskDataset(valid_dataset,
														task_transforms=valid_transforms,
														num_tasks=60)
		valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)
		#test
		test_dataset = l2l.data.MetaDataset(test_dataset)
		test_transforms = [
			 NWays(test_dataset, args.test_way),
			 KShots(test_dataset, args.test_query + args.test_shot),
			 LoadData(test_dataset),
			 RemapLabels(test_dataset),
		]
		test_tasks = l2l.data.TaskDataset(test_dataset,
													 task_transforms=test_transforms,
													 num_tasks=-1)
		test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
		
		
		####텐서보드
		exp_detail = args.file_name + "_Ways=" + str(args.train_way) +  "_Shots=" + str(args.shot) + "_Clustering="+str(args.clustering) +"_Concating="+str(args.concating)+"_Filtering="+str(args.filtering)
		log_path = os.path.join('./log/{}'.format(args.exp_name))
		time.sleep(random.randrange(1, 6, 1) / 10)
		log_path = os.path.join(log_path, exp_detail + "_" + str(
			datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + "_GPU=" + str(GPU_NUM))
		writer = SummaryWriter(logdir=log_path)
		
		
		
		#initialization
		model = Convnet()
		model.to(device)
		
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(
			 optimizer, step_size=2000, gamma=0.5)
		
		loss_ctr,n_loss,n_acc,error_cnt = 0,0,0,0
		
		#TODO : 텐서보드 파일명 지정할때 train_way,test_way,test_shot,query_num
		# clustering / filtering / concating /inferece 기록
		# 아예 텐서보드 옵션으로 넣어서 한눈에 볼 수 있는 표 만들어지도록
		#####################Train
		for epoch in range(1, args.num_iterations + 1):
			
			if patience > 5:
				print("break")
				sd_list.append(final_acc)
				break
			model.train()
			# TODO trian acc도 100번에 한번 하도록 수정 (완료)
			###########(adaptation step part at original code )
			##original code : https://github.com/learnables/learn2learn/blob/master/examples/vision/protonet_miniimagenet.py
			
			resampling = True
			while resampling:
				error_cnt += 1
				batch = next(iter(train_loader))
				loss, acc, resampling = slow_adapt(model,
															  batch,
															  args.train_way,
															  args.shot,
															  args.train_query,
															  metric=pairwise_distances_logits,
															  device=device,
															  fixed_eps=args.fixed_eps,
															  clustering=args.clustering,
															  filtering=args.filtering,
															  concating=args.concating)
			
			loss_ctr += 1
			n_loss += loss.item()
			n_acc += acc
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			
			if epoch % 100 == 0:
				loss_rec = n_loss / loss_ctr
				acc_res = n_acc / loss_ctr
				writer.add_scalar('train_acc_100', acc_res , epoch)
				writer.add_scalar('train_loss_100', loss_rec , epoch)
				print('train, epoch {}, loss={:.4f} acc={:.4f}'.format(
					epoch, loss_rec, acc_res))
				if args.filtering == True:
					print(f"error iteration : {error_cnt - epoch}")
					writer.add_scalar('error_resampling', (error_cnt - epoch), epoch)
					
				if (args.loss_eps == True) and (epoch == 100):
					loss_zero = loss_rec
					eps_zero = args.fixed_eps
					
				if args.loss_eps == True:
					args.fixed_eps = eps_zero * (loss_rec/loss_zero)
					
					
				loss_ctr = 0
				n_loss = 0
				n_acc = 0
			
			model.eval()
			
			valid_loss_ctr = 0
			valid_n_loss = 0
			valid_n_acc = 0
			if (epoch >= 10) and (epoch % evaluation_unit == 0):
				for i, batch in enumerate(valid_loader):
					loss, acc, resampling = slow_adapt(model,
																  batch,
																  args.test_way,
																  args.test_shot,
																  args.test_query,
																  metric=pairwise_distances_logits,
																  device=device,
																  clustering=args.clustering,
																  filtering=args.filtering,
																  concating=args.concating)
					
					valid_loss_ctr += 1
					valid_n_loss += loss.item()
					valid_n_acc += acc
				valid_loss_rec = valid_n_loss / valid_loss_ctr
				valid_acc_res = valid_n_acc / valid_loss_ctr
				writer.add_scalar('valid_acc_100', valid_acc_res, epoch)
				writer.add_scalar('valid_loss_100', valid_loss_rec, epoch)
				print('val, epoch {}, loss={:.4f} acc={:.4f}'.format(
					epoch, valid_loss_rec, valid_acc_res))
				
				valid_loss_ctr = 0
				valid_n_acc = 0
				
				for i in range(1, evaluation_roud ):
					batch = next(iter(valid_loader))
					loss, acc, resampling = slow_adapt(model,
																  batch,
																  args.test_way,
																  args.test_shot,
																  args.test_query,
																  metric=pairwise_distances_logits,
																  device=device,
																  clustering=args.clustering,
																  filtering=args.filtering,
																  concating=args.concating)
					valid_loss_ctr += 1
					valid_n_acc += acc
					if i == evaluation_roud-1:
						valid_acc_record = valid_n_acc / valid_loss_ctr * 100
						if before_valid_acc > valid_acc_record:
							patience += 1
							
							
				
							
							#TODO Break가 안먹히는 거 해결 필요, 해당 시퀀스가 아예 종료되어야함
								
						before_valid_acc = valid_acc_record
						writer.add_scalar('valid_acc_res _600_inference', valid_acc_res, epoch)
						
						print('valid, epoch {},Inference Setting, total episode {}: {:.2f}({:.2f})'.format(
							epoch, i, valid_n_acc / valid_loss_ctr * 100, acc * 100))
						if valid_acc_record > best_valid_acc :
							best_valid_acc = valid_acc_record
						
				
				# TO DO : args.inferenc_clustering / filtering /concating 지정해주기
				
				test_loss_ctr = 0
				test_n_acc = 0
				
				for i in range(1, evaluation_roud):
					batch = next(iter(test_loader))
					loss, acc, resampling = slow_adapt(model,
																  batch,
																  args.test_way,
																  args.test_shot,
																  args.test_query,
																  metric=pairwise_distances_logits,
																  device=device,
																  clustering=args.clustering,
																  filtering=args.filtering,
																  concating=args.concating)
					test_loss_ctr += 1
					test_n_acc += acc
					if i == evaluation_roud -1:
						test_acc_res = test_n_acc / test_loss_ctr * 100
						writer.add_scalar('test_acc_res _600_inference', test_acc_res, epoch)
						print(args.fixed_eps, args.clustering, args.filtering, args.concating, args.loss_eps)
						print('test, epoch {},Inference Setting, total episode {}: {:.2f}({:.2f})'.format(
							epoch, i, test_n_acc / test_loss_ctr * 100, acc * 100))
						if best_valid_acc == valid_acc_record:
							final_acc = test_acc_res
						terminate_time = timeit.default_timer()
						time_cost = (terminate_time - start_time)
						print('\n')
						print("%f초 걸렸습니다." % (time_cost))
				'''
				test_loss_ctr = 0
				test_n_acc = 0
				for i in range(1, 601):
					batch = next(iter(test_loader))
					loss, acc, resampling = slow_adapt(model,
																  batch,
																  args.test_way,
																  args.test_shot,
																  args.test_query,
																  metric=pairwise_distances_logits,
																  device=device,
																  clustering=None,
																  filtering=False,
																  concating=False)
					test_loss_ctr += 1
					test_n_acc += acc
					if i == 600:
						test_acc_res = test_n_acc / test_loss_ctr * 100
						writer.add_scalar('test_acc_res _600_only_embeddings', test_acc_res, epoch)
						print('test, epoch {},only embedding NN, total episode {}: {:.2f}({:.2f})'.format(
							epoch, i, test_n_acc / test_loss_ctr * 100, acc * 100))'''
					
					# TODO : Train/Valid/TEST1/TEST2 ACC/LOSS 텐서보드 기록
					# TODO : 시간소요 기록
			
		
		
		
		
		
		
		
		######끝날떄
		terminate_time = timeit.default_timer()
		time_cost = (terminate_time - start_time)
		print('\n')
		print("%f초 걸렸습니다." % (time_cost))
		sd_list.append(final_acc)
		

	print(args.fixed_eps , args.clustering, args.filtering, args.concating , args.loss_eps)
	print(sd_list)
	sd_list = [x.item() for x in sd_list]
	#print(np.mean(sd_list), round(np.std(sd_list), 2))
	try:
		print(np.mean(sd_list),round(np.std(sd_list),3) )
	except:
		print(sd_list)
	writer.flush()
	writer.close()

if __name__ == '__main__':
    main()


