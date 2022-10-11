from pickletools import optimize
from tracemalloc import start
from turtle import forward
import h5py
import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import time
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from TFRegNCI_model import loss_func, merge_model
import math
import sys

torch.set_default_dtype(torch.float32)

# Learning rate
def adjust_learning_rate(optimizer, epoch, lr, total_epochs):
    """
    Decay the learning rate with half-cycle cosine after warmup
    warmup_epochs : 40; min_lr : 0
    """
    warmup_epochs = 10
    min_lr = 0
    total_epochs = total_epochs
    init_lr = lr

    if epoch < warmup_epochs:
        lr = init_lr * (epoch / warmup_epochs)
    else:
        lr = min_lr + (init_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# Training
def train(train_loader, train_data, train_cube, train_labels_n, optimizer, net, scaler, epoch, lr, epoch_size):
    net.train(mode=True)
    accum_iter = 2
    optimizer.zero_grad()
    for index, items in enumerate(train_loader):
        des_train = torch.tensor(train_data[items]).cuda(0).reshape(-1,21).float()
        cube_train = torch.tensor(train_cube[items]).cuda(0).reshape(-1,137,133,124).float()
        train_label = torch.tensor(train_labels_n[items]).cuda(0).reshape(-1).float()

        with autocast():
            output = net(cube_train, des_train)
            output = output.reshape(-1)
            loss = loss_func(train_label, output)/accum_iter

            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                sys.exit(1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)
            scaler.step(optimizer)
            scaler.update()

            if (index + 1) % accum_iter == 0:
                optimizer.zero_grad()

# Evaluation
def evaluate(data_loader, chem, cube, labels, bzhcs, net):
    net.eval()
    for index, items in enumerate(data_loader):
        chem_test = torch.tensor(chem[items]).cuda(0).reshape(-1,21).float()
        cube_test = torch.tensor(cube[items]).cuda(0).reshape(-1,137,133,124).float()

        output = net(cube_test, chem_test)
        real_outputs = output.cpu().data.numpy()
        real_outputs = bzhcs.inverse_transform(real_outputs)
        real_outputs = real_outputs.reshape(1,-1)

        real_labels = labels[items]
        real_labels = real_labels.reshape(1,-1)

        if index == 0:
            member_list = real_outputs 
            member_labelslist = real_labels
        else:
            member_list = np.hstack((member_list,real_outputs))
            member_labelslist = np.hstack((member_labelslist,real_labels))
    
    member_labelslist = member_labelslist.reshape(-1)
    member_list = member_list.reshape(-1)
    loss = np.sqrt(np.mean(np.square(member_labelslist-member_list)))
    return loss

# Main process
def main():

    # load data
    chem_train_data  = h5py.File("D:\File\data\des_1635_train.hdf5","r")
    chem_test_data  = h5py.File("D:\File\data\des_1635_test.hdf5","r")
    chem_valid_data  = h5py.File("D:\File\data\des_1635_valid.hdf5","r")

    cube_train_data = h5py.File("D:\File\data\cube_1635_train.hdf5","r")
    cube_test_data = h5py.File("D:\File\data\cube_1635_test.hdf5","r")
    cube_valid_data = h5py.File("D:\File\data\cube_1635_valid.hdf5","r")
    
    train_data = np.array(chem_train_data['data'], dtype=np.float32)
    test_data = np.array(chem_test_data['data'], dtype=np.float32)
    valid_data = np.array(chem_valid_data['data'], dtype=np.float32)

    train_cube = np.array(cube_train_data['data'], dtype=np.float32)
    test_cube = np.array(cube_test_data['data'], dtype=np.float32)
    valid_cube = np.array(cube_valid_data['data'], dtype=np.float32)

    train_label = np.array(cube_train_data['labels'], dtype=np.float32).reshape(-1,1)
    test_label = np.array(cube_test_data['labels'], dtype=np.float32).reshape(-1,1)
    valid_label = np.array(cube_valid_data['labels'], dtype=np.float32).reshape(-1,1)

    bzhcs = preprocessing.StandardScaler().fit(train_label)
    train_labels_n = bzhcs.transform(train_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = merge_model().to(device) 

    Batch_size = 50
    lr = 1e-2

    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=5e-2 , lr=lr, betas=(0.9, 0.999))
    
    train_index = np.arange(train_cube.shape[0])
    test_index = np.arange(test_cube.shape[0])
    valid_index = np.arange(valid_cube.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_index,batch_size=Batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_index,batch_size=Batch_size,shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_index,batch_size=Batch_size,shuffle=False)

    epoch_size = 300

    train_RMSEloss_list=[]
    valid_RMSEloss_list=[]
    test_RMSEloss_list=[]

    scaler = GradScaler()

    # Training
    for epoch in range(epoch_size):
        train(train_loader, train_data, train_cube, train_labels_n, optimizer, net, scaler, epoch, lr, epoch_size)
        with torch.no_grad():
            train_loss = evaluate(train_loader, train_data, train_cube, train_label, bzhcs, net)
            test_loss = evaluate(test_loader, test_data, test_cube, test_label, bzhcs, net)
            valid_loss = evaluate(valid_loader, valid_data, valid_cube, valid_label, bzhcs, net)

            train_RMSEloss_list.append(train_loss)
            test_RMSEloss_list.append(test_loss)
            valid_RMSEloss_list.append(valid_loss)

        adjust_learning_rate(optimizer, epoch, lr, epoch_size)  # optimizer, epoch

        # Save model
        torch.save(net.state_dict(), 'net'+str(epoch)+'.pkl')

        print("Epoch [{}/{}]: Test {:.4f}, Train {:.4f}, Valid {:.4f}".format(epoch, epoch_size, np.round(test_loss, 4), np.round(train_loss,4), np.round(valid_loss,4)))

    # Loss curve
    epoch_list = np.arange(epoch_size)
    plt.plot(epoch_list, train_RMSEloss_list, label = 'Train loss', color='k', linewidth=1)
    plt.plot(epoch_list, valid_RMSEloss_list, label = 'Valid loss', color='g', linewidth=1)
    plt.plot(epoch_list, test_RMSEloss_list, label='Test loss', color='r', linewidth=3)
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.legend()
    plt.show()

    chem_train_data.close()
    chem_test_data.close()
    chem_valid_data.close()

    cube_train_data.close()
    cube_test_data.close()
    cube_valid_data.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))