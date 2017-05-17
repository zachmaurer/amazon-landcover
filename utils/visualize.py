import matplotlib.pyplot as plt
import os

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

def plot_results(results_dict, config = None, display = False):
    if config.predict:
        return
    #results_dict has keys train_loss, train_acc, val_acc for lists of value vs iteration number
    loss_history = results_dict['train_loss']

    train_f2 = results_dict['train_f2']
    train_all_or_none = results_dict['train_all_or_none']
    train_global_recall = results_dict['train_global_recall']

    val_f2 = results_dict['val_f2']
    val_all_or_none = results_dict['val_all_or_none']
    val_global_recall = results_dict['val_global_recall']
    
    plt.subplot(4, 1, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration #')

    plt.subplot(4, 1, 2)
    plt.title('F2 Score')
    plt.plot(train_f2, '-o', label='train')
    plt.plot(val_f2, '-o', label='val')
    plt.plot([0.82] * len(val_f2), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(4, 1, 3)
    plt.title('Exact Match')
    plt.plot(train_all_or_none, '-o', label='train')
    plt.plot(val_all_or_none, '-o', label='val')
    plt.plot([0.82] * len(val_all_or_none), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(4, 1, 4)
    plt.title('Global Recall')
    plt.plot(train_global_recall, '-o', label='train')
    plt.plot(val_global_recall, '-o', label='val')
    plt.plot([0.82] * len(val_global_recall), 'k--')
    plt.xlabel('Epoch')
    
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(12, 20)
    plt.subplots_adjust(hspace = 0.4)

    if display:
        plt.show()

    if config:
        plt.savefig(os.path.join(config.plots_dest, 'results.png'), bbox_inches='tight')



def main():    
    results_dict = {
        'train_loss': [10]*100,
        'train_f2': [95]*100,
        'train_all_or_none': [70]*100,
        'train_global_recall': [60]*100,
        'val_f2': [30]*100, 
        'val_all_or_none': [20]*100, 
        'val_global_recall': [10]*100
    }
    plot_results(results_dict, display = True)


if __name__ == '__main__':
    main()
