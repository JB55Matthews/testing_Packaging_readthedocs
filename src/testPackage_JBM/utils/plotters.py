
import numpy as np
import matplotlib.pyplot as plt


def plot_epoch_loss(epoch_loss, epochs, title):
    
    plt.semilogy(np.linspace(1, epochs, epochs),epoch_loss)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(title)
    plt.clf()
    return


def plot_solution_prediction(t, solPred, title):

    plt.plot(t, solPred)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('Predicted u')

    plt.savefig(title)
    plt.clf()
    return