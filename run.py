import pickle
import time
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.utils.data as data_utils
from torch.nn.parameter import Parameter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class bilstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional=True,
                 batch_first=True):
        super(bilstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first, dropout=dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.rl = nn.ReLU(True)
        self.sm = nn.Softmax()
        self.batch_first = batch_first


    def forward(self, x, seq_lengths):
        # get the batch_size
        batch_size = x.shape[0]
        # reset h_0 every time
        self.h_0 = Parameter(torch.randn(num_layers * 2, batch_size, hidden_size))
        self.c_0 = Parameter(torch.randn(num_layers * 2, batch_size, hidden_size))
        pack = rnn.pack_padded_sequence(x, seq_lengths, batch_first=self.batch_first)
        out, h_t = self.lstm(pack, (self.h_0, self.c_0))
        out, unpacked_len = rnn.pad_packed_sequence(out, batch_first=self.batch_first)
        # the output is h_t for each step, we only need the final state
        # TODO:I am not quite sure about this
        out = out[:, -1, :]
        out = self.rl(self.fc(out))
        out = self.sm(out)
        return out


def load_data(data_file, batch_first):
    with open(data_file, 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        lens_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)
        lens_test = pickle.load(f)
    x_train = rnn.pad_sequence(x_train, batch_first)
    x_test = rnn.pad_sequence(x_test, batch_first)
    return x_train, y_train, x_test, y_test, lens_train, lens_test


def save_checkpoint(state, is_best, prec, name, epoch, filename='checkpoint'):
    """Saves checkpoint to disk"""
    # save current epoch
    directory = "checkpoint/%s/" % (name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = directory + filename + '.pth.tar'
    torch.save(state, file_name)

    # save the best model
    if is_best:
        pth = directory + 'best' + '.pth.tar'
        shutil.copyfile(file_name, pth)
        print(('epoch [{0}]\t prec@[{1}]\t checkpoint saved at :{2}').format(epoch, prec, pth))



def validate(val_loader, model, loss, epoch, batch_size, lens, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        start = time.time()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var, lens[i * batch_size : (i + 1) * batch_size])
        prec = accuracy(output.data, target)
        los = loss(output, target_var)
        top1.update(prec[0], input.size(0))
        losses.update(los.data, input.size(0))
        batch_time.update(time.time() - start)
        # print the result
        if i == 0:
            print('-' * 50 + ' Validating ' + '-' * 50)
        if i % print_freq == 0 or i == len(val_loader) - 1:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time: {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                'Loss: {loss.val:.2f} ({loss.avg:.1f})\t'
                'Prec@1: {top1.val:.2f} ({top1.avg:.2f})'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))
    return top1.avg



def train(train_loader, model, loss, optimizer, epoch, batch_size, lens, print_freq=1):
    """training the model for an epoch

    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :return:
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()
    # resume the timing
    for i, (input, target) in enumerate(train_loader):
        start = time.time()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var, lens[i * batch_size : (i + 1) * batch_size])
        # compute loss and accuracy
        prec = accuracy(output.data, target)
        los = loss(output, target_var)
        top1.update(prec[0], input.size(0))
        losses.update(los.data, input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        los.backward(retain_graph=True)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - start)
        # print the result
        if i == 0:
            print('-' * 50 + ' Training ' + '-' * 50)
        if i % print_freq == 0 or i == len(train_loader) - 1:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time: {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                'Loss: {loss.val:.2f} ({loss.avg:.1f})\t'
                'Prec@1: {top1.val:.2f} ({top1.avg:.2f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k for the whole network"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    # experiment name
    name = '2016test'
    # this is used for test, a small dataset
    # data_file = 'data/2015train'
    data_file = 'data/2016test'
    input_size = 50
    hidden_size = 40
    num_layers = 1
    batch_size = 100
    output_size = 3
    dropout = 0.5
    # these two are default to be true
    bidirectional = True
    batch_first = True
    # training parameters
    epochs = 50
    learning_rate = 0.01
    # wrap the data
    x_train, y_train, x_test, y_test, lens_train, lens_test = load_data(data_file, batch_first)
    training = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(training, batch_size=batch_size, shuffle=False)
    testing = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(testing, batch_size=batch_size, shuffle=False)
    # init the model
    model = bilstm(input_size, hidden_size, num_layers, output_size, dropout)
    print('=> Number of model parameters are: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # set the optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.9)
    # set the loss
    loss = nn.CrossEntropyLoss()
    # start the training
    best_prec1 = 0
    for epoch in range(epochs):
        train(train_loader, model, loss, optimizer, epoch, batch_size, lens_train)
        prec1 = validate(test_loader, model, loss, epoch, batch_size, lens_test)
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        save_checkpoint({
            'epoch': epoch,
            'best_prec1': best_prec1,
            'state_dict': model.state_dict()
        }, is_best, prec1, name, epoch)
