from os import path, mkdir
from datetime import datetime
from torch import FloatTensor
from torch import cuda

# Class: Config
# ----
# Stores all the parameters for each run of an experiment.
# TODO
#
class Config:
  def __init__(self, args=None): 

    # Model and Experiment Settings
    self.epochs = args.e if args else None
    self.batch_size = args.bs if args else None
    self.num_train = args.nt if args else None
    self.num_val = args.nv if args else None
    self.lr = args.lr if args else None

    # GPU Settings
    self.use_gpu = args.gpu if args else None
    self.dtype = cuda.FloatTensor if self.use_gpu else FloatTensor
    if self.use_gpu: assert(cuda.is_available())

    # Loaders
    self.train_loader = None
    self.val_loader = None
    self.test_loader = None
    
    # Metadata
    self.title = '_' + args.title  if args else ""
    self.timestamp = datetime.now().strftime("%m%d_%H.%M.%S")
    self.experiment_id = "{}_{}".format(self.title, self.timestamp)

    # Saved Files Params
    self._save_path = "./experiments" # Never used by client
    self.save_dest = path.join(self._save_path, self.experiment_id) # Root folder for experiment
    self.log_dest = path.join(self.save_dest, "logs") # logs
    self.checkpoint_dest = path.join(self.save_dest, "checkpoints") # checkpoints
    self.plots_dest = path.join(self.save_dest, "plots") # plots
    self._createSubfolders() # TODO: *args?
   
    # Logger
    self.logger = ResultsLogger(self.log_dest)

  def __str__(self):
    config_str = "Config for experiment:   {}".format(self.experiment_id)
    config_str += "\n\gpu: {}".format(self.use_gpu)
    config_str += "\n\tepochs: {}".format(self.epochs)
    config_str += "\n\tbatch_size: {}".format(self.batch_size)
    config_str += "\n\learning_rate: {}".format(self.lr)
    config_str += "\n\tnum_train examples (if None using all): {}".format(self.num_train)
    config_str += "\n\tnum_val examples (if None using all): {}".format(self.num_val)  
    config_str += "\n\tsave_dest: {}".format(self.save_dest)
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