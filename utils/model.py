import shutil
import os
import torch
from .constants import BEST_MODEL_FILENAME

def countParams(model, config):
    num_params = sum([p.data.nelement() for p in model.parameters()])
    config.log('Number of model parameters: {}\n'.format(num_params))
    return num_params

def checkpointModel(model, config, optimizer, epoch, stats, is_best):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'stats': stats,
        'optimizer' : optimizer.state_dict(),
    }
    file = config.experiment_id + "_{}.ckpt".format(epoch+1)
    filename = os.path.join(config.checkpoint_dest, file)
    config.log("Saving checkpoint...{}".format(file))
    torch.save(checkpoint, filename)
    if is_best:
      config.log("New best model: {} with Train F2:  {}".format(file, stats['train_f2'])) #TODO: change to val
      shutil.copyfile(filename, os.path.join(config.checkpoint_dest, BEST_MODEL_FILENAME))
      config.log("Done saving checkpoint.")

def loadModel(model, config, optimizer):
    config.log("Loading checkpoint: {}".format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint)
    start_epoch = checkpoint['epoch']
    stats = checkpoint['stats']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for k, v in stats.items():
        stat = str(k) + " : " + str(v)
        config.log(stat)
    return start_epoch, stats