from __future__ import print_function

import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import trange
from collections import defaultdict
from utils import AverageMeter, accuracy, get_optim_lr

class Trainer(object):
    def __init__(self, model_name, model, args, train_on_gpu=True):
        '''
        Initial FP16 Trainer
        :param model_name: the name of the model
        :param model: model instance
        :param args: input setting parser
        :param train_on_gpu: train on GPUs
        '''
        self.model = model
        self.args = args
        self.model_name = model_name
        self.train_on_gpu = train_on_gpu
        self.loss_scaling = args.loss_scaling
        self.fp16_mode = args.fp16

        if train_on_gpu and torch.backends.cudnn.enabled:
            self.fp16_mode = args.fp16
        else:
            self.fp16_mode = False
            self.loss_scaling = False
            print("CuDNN backend not available. Can't train with FP16.")

        self.best_acc = 0
        self.best_epoch = 0
        self._LOSS_SCALE = 128.0

        if self.train_on_gpu: self.model = self.model.cuda()

        if self.fp16_mode:
            self.model = self.network_to_half(self.model)
            self.model_params, self.master_params = self.prep_param_list(self.model)

        if self.train_on_gpu: self.model = nn.DataParallel(self.model)

        # model save directory
        if not os.path.exists('result'): os.makedirs('result')
        fdir = f'result/{model_name}'
        if not os.path.exists(fdir): os.makedirs(fdir)
        self.DIR = fdir

        # save training history
        self.history = defaultdict(lambda: [])

        print('\nModel: {} | Training on GPU: {} | Mixed Precision: {} | '
              'Loss Scaling: {}'.format(self.model_name, self.train_on_gpu, self.fp16_mode, self.loss_scaling))

    def prep_param_list(self, model):
        """
        Create two set of of parameters. One in FP32 and other in FP16.
        Since gradient updates are with numbers that are out of range
        for FP16 this a necessity. We'll update the weights with FP32
        and convert them back to FP16.
        """
        model_params = [p for p in model.parameters() if p.requires_grad]
        master_params = [p.detach().clone().float() for p in model_params]

        for p in master_params:
            p.requires_grad = True

        return model_params, master_params

    def master_params_to_model_params(self, model_params, master_params):
        """
        Move FP32 master params to FP16 model params.
        """
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

    def model_grads_to_master_grads(self, model_params, master_params):
        for model, master in zip(model_params, master_params):
            if master.grad is None:
                master.grad = Variable(master.data.new(*master.data.size()))
            master.grad.data.copy_(model.grad.data)

    def BN_convert_float(self, module):
        '''
        Designed to work with network_to_half.
        BatchNorm layers need parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            self.BN_convert_float(child)
        return module

    class tofp16(nn.Module):
        """
        Add a layer so inputs get converted to FP16.
        Model wrapper that implements::
            def forward(self, input):
                return input.half()
        """
        def __init__(self):
            super(Trainer.tofp16, self).__init__()

        def forward(self, input):
            return input.half()

    def network_to_half(self, network):
        """
        Convert model to half precision in a batchnorm-safe way.
        """
        return nn.Sequential(self.tofp16(), self.BN_convert_float(network.half()))

    def warmup_learning_rate(self, init_lr, no_of_steps, epoch, len_epoch):
        """Warmup learning rate for 5 epoch"""
        factor = no_of_steps // 30
        lr = init_lr * (0.1**factor)
        lr = lr * float(1 + epoch + no_of_steps * len_epoch) / (5. * len_epoch)
        return lr

    def train(self, epoch, trainloader):
        self.model.train()

        train_loss = AverageMeter()
        prec = AverageMeter()

        # Declare optimizer.
        params = self.master_params if self.fp16_mode else self.model.parameters()
        optimizer = optim.SGD(params, self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        # learning rate scheduler
        scheduler = MultiStepLR(optimizer, milestones=[80, 120, 160, 180], gamma=0.1)

        # If epoch less than 5 use warmup, else use scheduler.
        if epoch < 5 and self.args.warm_up:
            lr = self.warmup_learning_rate(self.args.lr, self.args.epochs, epoch, len(trainloader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step(epoch=epoch)

        # Loss criterion is in FP32.
        criterion = nn.CrossEntropyLoss()

        with trange(len(trainloader)) as t:
            for idx, (inputs, targets) in enumerate(trainloader):
                if self.train_on_gpu: inputs, targets = inputs.cuda(), targets.cuda()
                self.model.zero_grad()
                outputs = self.model(inputs)
                # We calculate the loss in FP32 since reduction ops can
                # be wrong when represented in FP16.
                loss = criterion(outputs, targets)

                # Sometime the loss may become small to be represente in FP16
                # So we scale the losses by a large power of 2, 2**7 here.
                if self.loss_scaling: loss = loss * self._LOSS_SCALE
                # Calculate the gradients
                loss.backward()
                if self.fp16_mode:
                    # Move the calculated gradients to the master params
                    # so that we can apply the gradient update in FP32.
                    self.model_grads_to_master_grads(self.model_params, self.master_params)
                    if self.loss_scaling:
                        # If we scaled our losses now is a good time to scale it
                        # back since our gradients are in FP32.
                        for params in self.master_params:
                            params.grad.data = params.grad.data / self._LOSS_SCALE
                    # Apply weight update in FP32.
                    optimizer.step()
                    # Copy the updated weights back FP16 model weights.
                    self.master_params_to_model_params(self.model_params, self.master_params)
                else:
                    optimizer.step()

                train_loss.update(loss.item() / self._LOSS_SCALE, inputs.size(0))
                top1 = accuracy(outputs, targets)[0]
                prec.update(top1.item(), inputs.size(0))

                metrics = {'Epoch': f'{epoch + 1}',
                           'Loss': '%.2f' % train_loss.avg,
                           'Acc': '%.1f' % prec.avg,
                           'LR': '%.4f' % get_optim_lr(optimizer)}
                t.set_postfix(metrics)
                t.update()
            t.close()

        self.history['loss'].append(train_loss.avg)
        self.history['acc'].append(prec.avg)

    def evaluate(self, epoch, testloader):
        self.model.eval()

        test_loss = AverageMeter()
        prec = AverageMeter()

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            with trange(len(testloader)) as t:
                for idx, (test_x, test_y) in enumerate(testloader):
                    if self.train_on_gpu: test_x, test_y = test_x.cuda(), test_y.cuda()
                    outputs = self.model(test_x)
                    loss = criterion(outputs, test_y)
                    top1 = accuracy(outputs, test_y)

                    test_loss.update(loss.item(), test_x.size(0))
                    prec.update(top1.item(), test_x.size(0))

                    metrics = {'Loss': '%.2f' % test_loss.avg,
                               'Acc': '%.1f' % prec.avg}
                    t.set_postfix(metrics)
                    t.update()
            t.close()

        self.history['val_loss'].append(test_loss.avg)
        self.history['val_acc'].append(prec.avg)

        if prec.avg > self.best_acc: self.save_model(self.model, self.model_name, prec.avg, epoch)

    def save_model(self, model, model_name, acc, epoch):
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if self.fp16_mode:
            save_name = os.path.join(self.DIR, model_name + '_fp16',
                                     'weights.%03d.%.03f.pt' % (epoch, acc))
        else:
            save_name = os.path.join(self.DIR, model_name,
                                     'weights.%03d.%.03f.pt' % (epoch, acc))

        torch.save(state, save_name)
        print("\nSaved state at %.03f%% accuracy. Prev accuracy: %.03f%%" % (acc, self.best_acc))
        self.best_acc = acc
        self.best_epoch = epoch

    def load_model(self, path=None):
        """
        Load previously saved model. THis doesn't check for precesion type.
        """
        if path is not None:
            checkpoint_name = path
        elif self.fp16_mode:
            checkpoint_name = os.path.join(
                self.DIR, self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        else:
            checkpoint_name = os.path.join(
                self.DIR, self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        if not os.path.exists(checkpoint_name):
            print("Best model not found")
            return
        checkpoint = torch.load(checkpoint_name)
        self.model.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.best_epoch = checkpoint['epoch']
        print("Loaded Model with accuracy: %.3f%%, from epoch: %d" %
              (checkpoint['acc'], checkpoint['epoch'] + 1))

    def train_and_evaluate(self, traindataloader, testdataloader):
        self.best_acc = 0.0
        for epoch in range(self.args.epochs):
            self.train(epoch, traindataloader)
            self.evaluate(epoch, testdataloader)

    def return_history(self):
        return dict(self.history)