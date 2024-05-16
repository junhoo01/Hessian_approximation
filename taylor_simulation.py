from __future__ import print_function
import time
import random
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

from resnet import *

from ASGDOptimizer import ASGDOptimizer
from DCASGDTaylorOptimizer import DCASGDTaylorOptimizer
from HVPTaylorOptimizer import HVPTaylorOptimizer

##############################################
# 하이퍼 파라미터
##############################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser(description='asgd_simulation_arguments')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay step (default is 0.1)')
parser.add_argument('--batch-size', type=int, help='batch size')
parser.add_argument('--num-local-workers', type=int, help='the number of local workers <= 200')
parser.add_argument('--momentum', type=float, help='momentum value', default=0)
parser.add_argument('--nesterov', dest='nesterov', help='use nesterov?', default=False, action='store_true')
parser.add_argument('--no-nesterov', dest='nesterov', help='use nesterov?', default=False, action='store_false')
parser.add_argument('--seed', type=int, help='random seed', required=True)
parser.set_defaults(nesterov=False)
parser.add_argument('--weight-decay', type=float, help='weight decay', default=0.0)


parser.add_argument('--epoches', type=int, help='maximum epoches', default=100)
parser.add_argument('--gpu-id', type=int, help='gpu index', default=5)
parser.add_argument('--optim', type=str, help='choose optimizer: hvp, newton, dcasgd, else(asgd)')
parser.add_argument('--variance-factor', type=float, help='variance factor fro optimizer', default=1.0)
args = parser.parse_args()

learning_rate = args.lr
gamma = args.gamma
batch_size = args.batch_size
num_local_workers = args.num_local_workers
momentum_value = args.momentum
is_nesterov = args.nesterov
weight_decay = args.weight_decay
epoches = args.epoches
gpu_id = args.gpu_id
myoptim = args.optim
variance_factor = args.variance_factor

##############################################
# 랜덤 시드 고정
##############################################
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

##############################################
# 저장할 데이터
##############################################
test_loss_log = []
test_accruacy_log = []
train_loss_log = []
train_accruacy_log = []

##############################################
# 디바이스 설정 : GPU를 사용할 수 있다면 GPU 사용
##############################################
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

##############################################
# main : train/test를 실행하는 함수
# num_local_worker : 생성할 Local Worker 개수
##############################################
def main(num_local_workers):
    print(f'learning rate = {learning_rate}')
    print(f'step learning rate decay size = {gamma}')
    print(f'batch size = {batch_size}')
    print(f'number of local workers = {num_local_workers}')
    print(f'momentum value = {momentum_value}')
    print(f'is nesterov = {is_nesterov}')
    print(f'weight decay = {weight_decay}')
    print(f'epoches = {epoches}')
    print(f'gpu index = {gpu_id}')
    print(f'optimizer = {myoptim}')
    print(f'variance_factor = {variance_factor}')
    print('-----------------------')
    print(f'random seed = {random_seed}')
    print('-----------------------')

    ##############################################
    # suffle_indices: 데이터셋을 numpy에서 설정한 random index로 순서를 섞음
    # e.g.) [1, 2, 3, 4, 5] -> [5, 3, 2, 1, 4]
    # dataset_size : 데이터셋 크기
    # seed : 랜덤 시드
    ##############################################
    def shuffle_indices(dataset_size, seed=random_seed):
        np.random.seed(seed)
        indices = np.arange(dataset_size)
        indices = np.random.permutation(indices)
        return indices

    #############################################
    transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    indices = shuffle_indices(len(trainset), seed=random_seed)
    shuffled_train_datasets = Subset(trainset, indices)
    trainloader = DataLoader(shuffled_train_datasets, batch_size = batch_size, shuffle=False, num_workers=2, drop_last=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    ##############################################
    # 모델 구축
    ##############################################
    main_net = ResNet18().to(device=device)     # Local Model 파라미터를 받아서 Forward/Backward 연산을 수행.
    buffer_net = ResNet18().to(device=device)   # 기울기 비교 실험을 위해, 설정된 Master Model의 사본
    buffer_net.load_state_dict(main_net.state_dict())

    local_workers = []
    for local_worker_index in range(num_local_workers):
        local_worker = ResNet18().to(device='cpu')                      # Local Worker 생성 (GPU는 메모리가 부족할 수 있으므로, CPU에 저장)
        local_worker.load_state_dict(main_net.state_dict())             # Local Worker Parameter를 Master Model Parameter와 동일하게 설정
        local_workers.append(local_worker)                              # list에 방금 생성된 Local Worker를 삽입


    criterion = nn.CrossEntropyLoss()

    # main모델, 파라미터들 넘겨줌!


    if args.optim == 'hvp':
        print("** We are using HVPTaylorOptimizer **")
        optimizer = HVPTaylorOptimizer(main_net.parameters(), lr = learning_rate, weight_decay=weight_decay, momentum=momentum_value, nesterov=is_nesterov)
    elif args.optim == 'dcasgd':
        print("** We are using DCASGDTaylorOptimizer **")
        optimizer = DCASGDTaylorOptimizer(main_net.parameters(), lr = learning_rate, weight_decay=weight_decay, momentum=momentum_value, nesterov=is_nesterov)
    elif args.optim == 'asgd':
        print("** We are using Delayed Gradeint (ASGD) **")
        optimizer = ASGDOptimizer(main_net.parameters(), lr = learning_rate, weight_decay=weight_decay, momentum=momentum_value, nesterov=is_nesterov)
    else:
        print(f"** Optimizer를 모르겠음 {args.optim}**")
        exit(-1)
    
    # 정답과의 차이 비교를 위해서
    optimizer.register_buffer_net(buffer_net=buffer_net)                
    optimizer.register_num_local_workers(num_local_workers=num_local_workers)
    optimizer.register_variance_factor(variance_factor=variance_factor)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    ##############################################
    # 훈련 시작
    ##############################################
    accumulated_time = 0
    epoch_time = 0

    for epoch in range(epoches):    # loop over the dataset multiple times

        optimizer.set_epoch(cur_epoch=epoch)  

        main_net.train()            # main_net과 buffer_net 모두 train으로 설정하자.
        buffer_net.train()          # 정답과의 차이를 계산하기 위해서

        train_correct = 0
        train_loss = 0
        train_total = 0

        epoch_start_time = time.time()
        for i, data in enumerate(trainloader, 0):

            ##############################################
            # 1. 데이터로더에서 입력 데이터를 받음
            ##############################################
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True) # GPU로 옮기기
            labels = labels.to(device, non_blocking=True) # GPU로 옮기기

            ##############################################
            # 2. 동기화할 Local Worker 인덱스를 계산
            # round robin
            ##############################################
            
            local_worker_index = i % num_local_workers
            #local_worker_index = random.randint(0,num_local_workers-1) % num_local_workers
            


            ##############################################
            # main_net에는 최신 master_model 파라미터가 있음. 이것을 buffer_net에다가 복사.
            # 앞서 말했지만, 이제 buffer_net은 Master Model로써 동작한다.
            ##############################################
            buffer_net.load_state_dict(main_net.state_dict())
            buffer_net.zero_grad()

            ##############################################
            # 3. local_worker_index에 해당하는 Local Worker 파라미터를 main_net에 복사
            #    이렇게 되면, 현재 다음 상황과 같이 됨.
            #    buffer_net << 최신 Master Model 파라미터가 있음.
            #    main_net << Local Model 파라미터가 있음.
            #
            #    아무튼 이제 Local Worker의 파라미터를 가지고 기울기를 계산할 수 있음.
            ##############################################
            main_net.load_state_dict(local_workers[local_worker_index].state_dict())

            ##############################################
            # 4. gradient가 누적합되지 않도록 현재 optimizer(main_net)에 저장된 gradient들을 전부 0으로 초기화
            ##############################################
            optimizer.zero_grad()

            ##############################################
            # 5. Forward Pass, Backward Pass 계산
            #    이때, 계산은 Local Worker Parameter에 대해 진행됨
            ##############################################
            outputs = main_net(inputs)
            loss = criterion(outputs, labels)
            train_loss += float(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            ##############################################
            # Hessian Vector Product를 사용한다면, create_graph=True가 되어야함
            ##############################################
            loss.backward(create_graph=True)                  


            buffer_outputs = buffer_net(inputs)
            buffer_loss = criterion(buffer_outputs, labels)
            buffer_loss.backward()

            ##############################################
            # 7. Update
            # 여기서 Taylor expansion 계산과 Optimizer 파라미터 업데이트 실행
            # optimizer 내부에선 current_maste란, master_model 파라미터에 해당하는 파라미터가 있음.
            # 따라서, optimizer.step()을 통해 delayed gradient가 master_model 파라미터에 반영됨.
            ##############################################
            optimizer.step()

            ##############################################
            # 8. 방금 새롭게 업데이트된 Master Model Parameter를
            #     local_worker_index에 해당하는 Local Worker에 복사
            ##############################################
            local_workers[local_worker_index].load_state_dict(main_net.state_dict())



        ##############################################
        # 11. Test Model
        ##############################################
        epoch_end_time = time.time()
        correct = 0
        test_loss = 0
        total = 0

        with torch.no_grad():

            main_net.eval()
            buffer_net.eval()

            for data in testloader:
                images, labels = data
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = main_net(images)                      # 현재 Master Model 파라미터가 있는 main_net으로 실행
                
                loss = criterion(outputs, labels)
                test_loss += float(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)

                correct += (predicted == labels).sum().item()



        epoch_time = epoch_end_time - epoch_start_time
        accumulated_time += epoch_time

        train_loss_log.append(train_loss / len(trainloader))
        train_accruacy_log.append((100 * train_correct / train_total))
        test_loss_log.append(test_loss / len(testloader))
        test_accruacy_log.append((100 * correct / total))
        print(f"[ {epoch} in {epoches} : total time {accumulated_time}] : Train Loss: {train_loss_log[-1] : .2f}, Train Accuracy: {train_accruacy_log[-1] : .2f}, Test Loss: {test_loss_log[-1] : .2f}, Test Accuracy: {test_accruacy_log[-1] : .2f} in Total {total}, learning rate = {scheduler.get_last_lr()}")

        ##############################################
        # Learning Rate Decay 실행
        ##############################################
        scheduler.step()

        ##############################################
        # train dataset 다시 설정
        ##############################################
        indices = shuffle_indices(len(trainset), seed=(random_seed + (epoch + 1) * 10))
        shuffled_train_datasets = Subset(trainset, indices)
        trainloader = DataLoader(shuffled_train_datasets, batch_size = batch_size, shuffle=False, num_workers=2, drop_last=True)

        ##############################################
        # train loss 또는 test loss가 발산했다면, 훈련 종료
        ##############################################
        if math.isnan(train_loss_log[-1]) or math.isnan(test_loss_log[-1]):
            print('NaN is detected!')
            break



    # ##############################################
    # # 훈련 종료 후 로그 데이터 csv 저장 후 최종 훈련된 모델 저장
    # ##############################################
    import pandas as pd
    log_dict = {'train loss' : train_loss_log,
                'train accuracy' : train_accruacy_log,
                'test loss' : test_loss_log,
                'test accuracy' : test_accruacy_log}

    df = pd.DataFrame(log_dict)
    if args.optim == "asgd":
        file_name = str(random_seed) + "_log_ResNet18_" + str(args.optim) + "_" + \
        str(batch_size) + "_" + str(learning_rate) + "_" + str(num_local_workers)
    elif args.optim == "dcasgd":
        file_name = str(random_seed) + "_log_ResNet18_" + str(args.optim) + "_" + \
        str(variance_factor) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_local_workers)
    elif args.optim == "hvp":
        file_name = str(random_seed) + "_log_ResNet18_" + str(args.optim) + "_" + \
        str(variance_factor) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_local_workers)
    else:
        file_name = str(random_seed) + "_log_ResNet18_" + str(args.optim) + "_" + \
        str(batch_size) + "_" + str(learning_rate) + "_" + str(num_local_workers)

    df.to_csv('./data_log/'+ file_name + '.csv')

    print('Finished!')

if __name__ == '__main__':
    main(num_local_workers)