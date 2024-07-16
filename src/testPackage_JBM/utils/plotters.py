
import numpy as np
import matplotlib.pyplot as plt


def plot_epoch_loss(epoch_loss, epochs, title):

    """
    Plots epoch loss of trained network

    Args:
        epoch_loss (list): output of network, epoch loss over training
        epochs (int): epochs network was trained over
        title (string): title of file saved for graph
    
    Figure saved in current directory
    
    """
    
    plt.semilogy(np.linspace(1, epochs, epochs),epoch_loss)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(title)
    plt.clf()
    return


def plot_solution_prediction(t, solPred, title):
    """
    Plots epoch loss of trained network

    Args:
        t (list): from ode_solution class, equally spaced points along t
        solPred (list): output of network, predicted solution of trained network
        title (string): title of file saved for graph
    
    Figure saved in current directory
    
    """

    plt.plot(t, solPred)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('Predicted u')

    plt.savefig(title)
    plt.clf()
    return