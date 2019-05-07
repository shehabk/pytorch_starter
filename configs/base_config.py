import argparse
import os
import torch

# Inspired by 
class BaseClassificationOptions():

    def __init__(self):
        """"""
        self.initialized = False
        self.parser = None
        self.opt = None
        pass

    def initialize(self, parser):

        # Expriment specific arguments
        parser.add_argument("-m", "--mode", metavar='N', default='train',
                             help='mode train/test (default: train)')

        parser.add_argument("-r", "--resume", help="Resuming Experiment after stop",
                             action="store_true")

        parser.add_argument("-cpb", "--copy_back", help="copy back models after every experiment",
                             action="store_true")

        parser.add_argument("-ww", "--which-way", type=int, metavar='N', default=0,
                             help='0:individual samples|1:max of samples|2:sum of samples')

        parser.add_argument("-gi",'--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


        parser.add_argument("-nc", "--num-classes", default=7, type=int, metavar='N',
                             help='number of classes in classification')


        parser.add_argument("-arch", '--architecture', default='vgg16',
                             metavar='A',
                             help='architecture (resnet101|vgg_16|zibonet)')


        parser.add_argument("-lf", '--log_prefix', default='',
                             metavar='N',
                             help='print prefix for logs (default=\'\') (example=\'mmi_set1\')')

        # Should not be necessary
        # parser.add_argument("-ename", '--experiment_name', default='def',
        #                      metavar='N',
        #                      help='experiment name (default=\'\') example=\'emo_baselines\'')

        # parser.add_argument("-epad", '--experiment_padding', default='default',
        #                      metavar='N',
        #                      help='experiment padding for storing (default=\'\') (example=\'run_1\')')



        ####### Data Loader Options #####################################
  
        # Used in all optimizers
        parser.add_argument( "-nw", '--num-workers',  default=4, type=int,
                             metavar='W', help='num-workers (default: 4)')

        parser.add_argument("-pm", "--pin-memory", help="Pin Memory for CUDA",
                            action="store_true")

        parser.add_argument( "-epl", '--epoch-length',  default=2000, type=int,
                             metavar='W', help='epoch-length (default: 2000)')

        
        
        ####### Optimizer Related Options ########

        parser.add_argument("-ep", "--epochs", default=35, type=int, metavar='N',
                             help='number of epochs in the experiment default (35)')


        parser.add_argument("-sep", "--start-epoch", default=1, type=int, metavar='N',
                             help='manual epoch number '
                             '(useful on restarts)')

        parser.add_argument("-ptnc", '--patience', default=0, type=int,
                             metavar='N',
                             help=('patience for early stopping'+
                                  '(0 means no early stopping)'))


        parser.add_argument("-lr", '--lr', default=0.01, type=float,
                             metavar='LR',
                             help='initial learning rate (default: 0.1)')
        parser.add_argument("-lrd", '--lr-decay', default=0.1, type=float,
                             metavar='N',
                             help='decay rate of learning rate (default: 0.4)')
        parser.add_argument("-lrs", "--lr-step", default=10, type=int,
                             metavar='N',
                             help='learning step')
        parser.add_argument("-bs", '--batch-size', default=64, type=int,
                             metavar='N', help='mini-batch size (default: 64)')
        
        # Select optimizer
        parser.add_argument("-opt", '--optimizer', default='sgd',
                             choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                             help='optimizer (default=sgd)')



        # Parameters for SGD
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                             help='momentum (default=0.9)')
        parser.add_argument('--no_nesterov', dest='nesterov',
                             action='store_false',
                             help='do not use Nesterov momentum')

        # Parameters for rmsprop
        parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                             help='alpha for ')

        # Parameters for adam
        parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                             help='beta1 for Adam (default: 0.9)')
        parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                             help='beta2 for Adam (default: 0.999)')

        # Used in all optimizers
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                             metavar='W', help='weight decay (default: 1e-4)')





        ####### Log Related Options ##############

        parser.add_argument( "-li", '--log-interval',  default=100, type=int,
                             metavar='L', help='log-interval (default: 100)')


        ####### Image Path Related #############

        parser.add_argument("-imr", '--image_root', default='def',
                             metavar='N',
                             help='root directory of the image')
        parser.add_argument("-imtr", '--train_list', default='def',
                             metavar='N',
                             help='image list training')
        parser.add_argument("-imvl", '--valid_list', default='def',
                             metavar='N',
                             help='image list validation')
        parser.add_argument("-imts", '--test_list', default='def',
                             metavar='N',
                             help='image list test')

        ####### Output Directory
        parser.add_argument("-od", '--output_dir', default='def',
                             metavar='N',
                             help='output directory for results models and logs'
                             )


        ####### Tensorboard Related Arguments
        parser.add_argument("-tb", "--tb_log", help="Use Tensorboard For Logging",
                             action="store_true")

        self.initialized = True
        self.parser = parser
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # We can add or change default value of additional options we want to use
        # refer to cyclegan implementation to see how it is done



        # save and return the parser
        self.parser = parser
        return parser.parse_args()    

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # Added to match pytorch ck probably can get around with it.
        if len(opt.gpu_ids) > 0 and \
            torch.cuda.is_available():
            opt.cuda = True
        else:
            opt.cuda = False

        self.opt = opt
        return self.opt