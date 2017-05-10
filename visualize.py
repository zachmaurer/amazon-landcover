import matplotlib.pyplot as plt

# Jupyter Boilerplate
# TODO: Can this be run as a client function outside of a notebook?

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# Function: plot_results
# ----
# Helper function to plot training loss, and train+val accuracy by epoch
# Args:
#    - results_dict, a dictionary to lists
#               - train_loss
#               - train_acc
#               - val_acc
# 
def plot_results(results_dict, config = None):
    #results_dict has keys train_loss, train_acc, val_acc for lists of value vs iteration number
    loss_history = results_dict['train_loss']
    train_acc_history = results_dict['train_acc']
    val_acc_history = results_dict['val_acc']
    
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(train_acc_history, '-o', label='train')
    plt.plot(val_acc_history, '-o', label='val')
    plt.plot([0.7] * len(val_acc_history), 'k--')
    plt.xlabel('iteration_num')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()
    if config:
        # TODO: save plots to file



