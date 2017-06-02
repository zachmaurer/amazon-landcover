from os import path, mkdir
from datetime import datetime
from torch import FloatTensor
from torch import cuda
import argparse

# Class: Config
# ----
# Stores all the parameters for each run of an experiment.
# TODO
#
class Config:
  def __init__(self, args): 
    if args.test:
      self._testSettings()   
    else:
      # Model and Experiment Settings
      self.epochs = args.e if args.e else None
      self.batch_size = args.bs if args.bs else None
      self.num_train = args.nt if args.nt else None
      self.num_val = args.nv if args.nv else None
      self.lr = args.lr if args.lr else None
      self.print_every = args.pe if args.pe else None
      self.eval_every = args.ee if args.ee else None
    self.checkpoint = args.checkpoint if args.checkpoint else None
    self.save_every = args.save_every if args.save_every else None
    self.no_save = args.no_save if args.no_save else None
    self.predict = args.predict if args.predict else None
    self.seed = args.seed if args.seed else None

    if self.predict:
      assert(self.checkpoint)
    assert(self.batch_size > 1)
    # GPU Settings
    self.use_gpu = args.gpu if args else None
    self.dtype = cuda.FloatTensor if self.use_gpu else FloatTensor
    if self.use_gpu: assert(cuda.is_available())
    
    # Loaders
    self.train_loader = None
    self.val_loader = None
    self.test_loader = None
    
    # Metadata
    self.title = args.title  if args else ""
    self.timestamp = datetime.now().strftime("%m%d_%H.%M.%S")
    self.experiment_id = "{}_{}".format(self.title, self.timestamp) if self.title else "{}".format(self.timestamp)

    # Saved Files Params
    self._save_path = args.path if args.path else "./experiments" 
    self.save_dest = path.join(self._save_path, self.experiment_id) # Root folder for experiment
    self.log_dest = path.join(self.save_dest, "logs") # logs
    self.logs = path.join(self.log_dest, "console.txt") # logs
    self.checkpoint_dest = path.join(self.save_dest, "checkpoints") # checkpoints
    self.plots_dest = path.join(self.save_dest, "plots") # plots
    self._createSubfolders() # TODO: *args?
   
    # Logger
    self.logger = ResultsLogger(self.log_dest)

  def _testSettings(self):
    # Test Settings
    self.epochs = 2
    self.batch_size = 2
    self.num_train = 2
    self.num_val = 2
    self.lr = 1e-3
    self.print_every = 1
    self.eval_every = None


  def __str__(self):
    config_str = "Config for experiment:   {}".format(self.experiment_id)
    config_str += "\n\ttitle: {}".format(self.title)
    config_str += "\n\tgpu: {}".format(self.use_gpu)
    config_str += "\n\tepochs: {}".format(self.epochs)
    config_str += "\n\tbatch_size: {}".format(self.batch_size)
    config_str += "\n\tlearning_rate: {}".format(self.lr)
    config_str += "\n\tnum_train (if None using all): {}".format(self.num_train)
    config_str += "\n\tnum_val (if None using all): {}".format(self.num_val)  
    config_str += "\n"
    config_str += "\n\tsave_dest: {}".format(self.save_dest)
    config_str += "\n\tsave_every: {}".format(self.save_every)
    config_str += "\n\tprint_every: {}".format(self.print_every)
    config_str += "\n\teval_every: {}".format(self.eval_every)
    config_str += "\n"
    return config_str

  def _createSubfolders(self):
    # Creates a folder structure for saving the elements of each experienment.
    if not path.isdir(self._save_path): mkdir(self._save_path)
    if (path.isdir(self.save_dest)):
      print("Save directory aready exists!")
      exit(1)
    else:
      mkdir(self.save_dest)
      mkdir(self.log_dest)
      mkdir(self.checkpoint_dest)
      mkdir(self.plots_dest)

  def log(self, message, echo = True):
    if echo:
      print(message)
    with open(self.logs, 'a') as f:
      print(message, file = f)


def parseConfig(description="Default Model Description"):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--test', action='store_true', help='sanity test to run model on small input', default = False)
  parser.add_argument('--bs', type=int, help='batch size for training', default = 20)
  parser.add_argument('--e', type=int, help='number of epochs', default = 10)
  parser.add_argument('--nt', type=int, help='number of training examples', default = None)
  parser.add_argument('--nv', type=int, help='number of validation examples', default = None)
  parser.add_argument('--lr', type=float, help='learning rate', default = 1e-3)
  parser.add_argument('--gpu', action='store_true', help='use gpu', default = False)
  parser.add_argument('--pe', type=int, help='print frequency', default = None)
  parser.add_argument('--ee', type=int, help='eval frequency', default = None)
  parser.add_argument('--seed', type=int, help='random seed', default = 231)
  parser.add_argument('--title', help='experiment title', default = None)
  parser.add_argument('--path', help='save path for results, logs, checkpoints', default = "./experiments")
  parser.add_argument('--checkpoint', action='store', help='resume from an exisiting model', type=str, default = None)
  parser.add_argument('--predict', action='store_true', help='predict only, no training', default = False)
  parser.add_argument('--save_every', type=int, help='save checkpoint after n epochs', default = 1)
  parser.add_argument('--no_save', action='store_true', help='save checkpoint after every epoch', default = False)
  args = parser.parse_args()
  return args


# Class: ResultsLogger
# ----
# Takes care of writing to all the different CSV files that logs the results
# TODO
#
class ResultsLogger:
  def __init__(self, path):
    self.path = path

  def writeRow(self):
    pass