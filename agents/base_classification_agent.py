import os
import sys
import shutil
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader , sampler

# project imports
from agents.base_agent import BaseAgent
from datasets.basic_datasets import ClassificationDataset

from util.util_functions import *
from util.util_classes import Tee, AverageMeter

class BaseClassicationAgent(BaseAgent):
    def __init__( self, config ):
        super().__init__(config)
        self.global_step = 0
        self.best_acc = 0
        if self.config.tb_log == True:
            self.tblogger = get_tblogger( self.config.output_dir)
        
    def load_checkpoint(self, load_best=False, 
                        filename='checkpoint.pth.tar',
                        best_filename='model_best.pth.tar'):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        ## models_dir
        models_dir = os.path.join(self.config.output_dir, 'models')
        # mkdirs(models_dir)


        if not load_best:
            model_file = os.path.join(models_dir, filename)
        else:
            model_file = os.path.join(models_dir, best_filename)

        ## If model file not present donot bother loading
        if not os.path.isfile(model_file):
            return


        checkpoint = torch.load(model_file)
        self.best_acc = checkpoint['best_acc']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        



    
    
    def save_checkpoint(self, state, is_best = 0,
                        file_name="checkpoint.pth.tar",
                        best_filename='model_best.pth.tar'):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """


        ## create models dir
        models_dir = os.path.join(self.config.output_dir, 'models')
        mkdirs(models_dir) 


        # filenames
        filename = os.path.join(models_dir, file_name)
        best_filename = os.path.join(models_dir, best_filename)
        # -------------------------------------------------
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_filename)
        

    def run(self):
        """
        The main operator
        :return:
        """
        if ( self.config.mode == 'train'):
            self.train()
        elif ( self.config.mode == 'sc'):
            self.write_score()
        elif (self.config.mode == 'acc'):
            self.get_accuracy()
        elif (self.config.mode == 'cf'):
            self.get_confusion_matrix()


    def train(self):
        """
        Main training loop
        :return:
        """
        ## Initialize model, optimizer and loss
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.loss = self.get_loss()

        ## Initialize the dataloaders
        self.train_loader = self.get_dataloader_train()
        self.test_loader  = self.get_dataloader_eval( partition = 'test' )
        self.val_loader   = self.get_dataloader_eval( partition = 'val'  )

        ## Log in both file and stdout
        log_file_path = os.path.join(self.config.output_dir, 'log.txt')
        mkdirs(os.path.dirname(log_file_path))

        if self.config.resume:
            log_file = open(log_file_path, 'a+')
        else:
            log_file = open(log_file_path, 'w+')

        sys.stdout = Tee(log_file, sys.stdout)
        assert os.path.isfile(log_file_path)
        

        ## Save Configuration
        config_file_path = os.path.join(self.config.output_dir, 'config.txt')
        mkdirs(os.path.dirname(config_file_path))
        save_params(config_file_path,self.config)
        assert os.path.isfile(config_file_path)
    


        if self.config.resume:
            self.load_checkpoint()

        for epoch in range(self.config.start_epoch, self.config.epochs + 1):
            self.adjust_learning_rate(epoch)
            trainloss , trainacc = self.train_one_epoch(epoch)
            validloss , validacc = self.validate()
            
            val_acc = validacc.avg
            is_best = val_acc >= self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            
            
            ## save checkpoint
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step
            }, is_best)

            ## print in console
            print('TRAIN SET: AVERAGE LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
                trainloss.avg, int(trainacc.sum), trainacc.count,
                100. * trainacc.avg))
            print('TEST SET: AVERAGE LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
                validloss.avg, int(validacc.sum), validacc.count,
                100. * validacc.avg))

            ## tensorboard logging 
            ## https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py           
            if (self.config.tb_log == True):
                # log the scalar values
                info = {
                    'train_loss': trainloss.avg,
                    'train_acc' : trainacc.avg ,
                    'valid_loss': validloss.avg,
                    'valid_acc' : validacc.avg
                }

                for tag, value in info.items():
                    self.tblogger.scalar_summary(tag, value, epoch + 1 )

            ## copy back to my computer from remote computer
            # if self.config.copy_back:
            #     self.pool.apply_async(methods_util.copy_to_shehabk ,[self.config.exp_dir])

    def train_one_epoch(self, epoch):
        """
        One epoch of training
        :return:
        """
        losses     = AverageMeter()
        accuracies = AverageMeter()

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data , requires_grad = False), Variable(target , requires_grad = False)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()

            # Gradien Clipping If necessary
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)
            self.optimizer.step()

            self.global_step+=1

            ## update loss 
            losses.update(loss.data[0] , target.size(0))
            
            ## uppdate accuracy 
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc =   float(correct) / target.size(0)
            accuracies.update(acc , target.size(0))

            if batch_idx % self.config.log_interval == 0:
                print('Train {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.config.log_prefix, epoch, batch_idx * len(data), len(self.train_loader.sampler), #train_loader.dataset # do train_loader.sampler
                                     100. * batch_idx / len(self.train_loader), loss.data[0]))

                ## log tensorboard
                if (self.config.tb_log == True):
                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'loss': loss.data[0],
                        # 'accuracy': accuracy.data[0]
                    }

                    for tag, value in info.items():
                        self.tblogger.scalar_summary(tag, value, self.global_step + 1)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in self.model.named_parameters():

                        if value.grad is None:
                            continue

                        tag = tag.replace('.', '/')
                        self.tblogger.histo_summary(tag, value.data.cpu().numpy(), self.global_step + 1)
                        self.tblogger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.global_step + 1)

                    # (3) Log the images
                    # info = {
                    #     'images': to_np(images.view(-1, 28, 28)[:10])
                    # }
                    #
                    # for tag, images in info.items():
                    #     logger.image_summary(tag, images, step + 1)

        return losses , accuracies

    def validate(self, partition = 'val'):
        """
        One cycle of model validation
        :return:
        """
        losses     = AverageMeter()
        accuracies = AverageMeter()

        if partition == 'val':
            loader = self.test_loader
        else:
            loader = self.val_loader

        self.model.eval()

        for data, target in loader:
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            loss   = self.loss(output, target)  # sum up batch loss

            # output = F.softmax(output)

            ## update loss
            losses.update(loss.data[0] , target.size(0))
            
            ## update accuracy
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc =   float(correct) / target.size(0)
            accuracies.update(acc , target.size(0))




        return losses , accuracies

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError 


    def get_model(self):

        if self.config.architecture in ['resnet34' , 
            'resnet50' , 'resnet101']:
            if self.config.architecture == 'resnet34':
                model = models.resnet34(pretrained=True)
            if self.config.architecture == 'resnet50':
                model = models.resnet50(pretrained=True)
            if self.config.architecture == 'resnet101':
                model = models.resnet101(pretrained=True)

            num_classes = self.config.num_classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)


        if self.config.architecture in ['densenet121', 
            'densenet169', 'densenet201', 'densenet161']:

            if self.config.architecture == 'densenet121':
                model = models.densenet121(pretrained=True)
            if self.config.architecture == 'densenet169':
                model = models.densenet169(pretrained=True)
            if self.config.architecture == 'densenet201':
                model = models.densenet201(pretrained=True)
            if self.config.architecture == 'densenet161':
                model = models.densenet161(pretrained=True)

            num_classes = self.config.num_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)

        if self.config.architecture in ['vgg11_ad', 'vgg11_bn_ad']:
            if self.config.architecture == 'vgg11_ad':
                model = models.vgg11(pretrained=True)



        if self.config.architecture in [ 'vgg11', 'vgg11_bn',
         'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']:
            if self.config.architecture == 'vgg11':
                model = models.vgg11(pretrained=True)
            if self.config.architecture == 'vgg11_bn':
                model = models.vgg11_bn(pretrained=True)
            if self.config.architecture == 'vgg13':
                model = models.vgg13(pretrained=True)
            if self.config.architecture == 'vgg13_bn':
                model = models.vgg13_bn(pretrained=True)
            if self.config.architecture == 'vgg16':
                model = models.vgg16(pretrained=True)
            if self.config.architecture == 'vgg16_bn':
                model = models.vgg16_bn(pretrained=True)
            if self.config.architecture == 'vgg19_bn':
                model = models.vgg19_bn(pretrained=True)
            if self.config.architecture == 'vgg19':
                model = models.vgg19(pretrained=True)

            num_classes = self.config.num_classes
            in_features = model.classifier[6].in_features
            n_module = nn.Linear(in_features, num_classes)
            n_classifier = list(model.classifier.children())[:-1]
            n_classifier.append(n_module)
            model.classifier = nn.Sequential(*n_classifier)




        if self.config.cuda:
            model.cuda()

        return model

    
    
    def get_optimizer(self):

        if self.config.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.config.lr,
                                  momentum=self.config.momentum,
                                  weight_decay=self.config.weight_decay
                                  )
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr = self.config.lr,
                                   betas = (self.config.beta1 , 
                                            self.config.beta2),
                                   weight_decay=self.config.weight_decay)

        return optimizer


    def get_loss(self):
        loss = torch.nn.CrossEntropyLoss()
        return loss


    def get_dataloader_train(self , balance = True):
    # kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
        kwargs = {'num_workers': self.config.num_workers, 
                'pin_memory': self.config.pin_memory }

        image_list = self.config.train_list
        assert os.path.exists( image_list ),\
                ("train_list invalid")
        
        dataset = ClassificationDataset(
                    self.config.image_root,
                    image_list,
                    transform=transforms.Compose([
                        transforms.RandomRotation(3),
                        # transforms.RandomCrop(224),
                        transforms.RandomResizedCrop(
                            224, 
                            scale=(0.74, 0.78),
                            ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
                    ]),
                    )



        # # Balancing the classes
        if balance == True:
            prob = np.zeros(self.config.num_classes)
            for i in range(len(dataset)):
                cur_class = dataset.labels[i]
                prob[cur_class]+=1
            prob = 1.0 / prob


            reciprocal_weights = np.zeros(len(dataset))
            epoch_length = self.config.epoch_length
            for i in range(len(dataset)):
                label = dataset.labels[i]
                reciprocal_weights[i] = prob[label]
            weights  = torch.from_numpy(reciprocal_weights)


            weighted_sampler = sampler.WeightedRandomSampler(weights,
                                epoch_length)
            loader = DataLoader(dataset, batch_size=self.config.batch_size,
                                sampler=weighted_sampler, **kwargs)
        else:
            loader = DataLoader(dataset, batch_size=self.config.batch_size,
                                shuffle=True)

        return loader

    def get_dataloader_eval(self , partition = 'val'):

        kwargs = {'num_workers': self.config.num_workers,
                 'pin_memory': self.config.pin_memory}

        if partition == 'val':
            image_list = self.config.valid_list
        else:
            image_list = self.config.test_list

        assert os.path.exists( image_list ), \
                (partition + "_list invalid")

        dataset = ClassificationDataset(
                                    self.config.image_root,
                                    image_list,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406], 
                                            [0.229, 0.224, 0.225]
                                            )
                                    ]))



        loader = DataLoader(dataset, batch_size=self.config.batch_size,
                            shuffle=False, **kwargs)

        return loader


    def write_score(self):
        # load model
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.loss = self.get_loss()

        # update the model parameter with the best model.
        self.load_checkpoint(load_best = True)
        self.test_loader = self.get_dataloader_eval(partition = 'test')

        # create results directory
        results_dir = os.path.join( self.config.output_dir, 'results')
        mkdirs(results_dir)

        # score and label file

        score_file_path = os.path.join( results_dir, 'score.txt')
        label_file_path = os.path.join( results_dir, 'label.txt')


        #############################################
        # change model in eval mode (no Dropout like random events)
        self.model.eval()

        outputs = []
        labels = []

        for data, target in self.test_loader:
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)

            # output = F.softmax(output)
            outputs.append(output.data.cpu().numpy())
            labels.append(target.data.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)

        np.savetxt(score_file_path, outputs)
        np.savetxt(label_file_path, labels)

    
    def get_accuracy(self):
        
        # results dir
        results_dir = os.path.join( self.config.output_dir, 'results')   

        # score and label file
        score_file_path = os.path.join( results_dir, 'score.txt')
        label_file_path = os.path.join( results_dir, 'label.txt')

        assert os.path.exists(score_file_path), 'score file not present'
        assert os.path.exists(label_file_path), 'label file not present'

        outputs = np.loadtxt(score_file_path)
        labels  = np.loadtxt(label_file_path)

        acc = get_accuracy( outputs, labels )
        print ("Average: %.4f"% (acc))

    def get_confusion_matrix(self):

        # results dir
        results_dir = os.path.join( self.config.output_dir, 'results') 
    

        # score and label file
        score_file_path = os.path.join( results_dir, 'score.txt')
        label_file_path = os.path.join( results_dir, 'label.txt')

        assert os.path.exists(score_file_path), 'score file not present'
        assert os.path.exists(label_file_path), 'label file not present'

        outputs = np.loadtxt(score_file_path)
        labels  = np.loadtxt(label_file_path)

        cf = get_confusion_matrix(outputs, 
                labels, 
                self.config.num_classes)

        print ( matrix_to_str (cf))



   # This is old fashioned, use lr_sheduler instead
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.config.lr * ( self.config.lr_decay ** (epoch // self.config.lr_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



        