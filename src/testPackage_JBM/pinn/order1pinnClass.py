
import numpy as np
from ..utils import plotters
from ..utils import points
from .order1pinnSolver import PINNtrain_IVP

""" Class containing ode solution information
    returned on solveODE calls
"""
class ode_solution:
    
    def __init__(self, eqn, order, inits, t_bdry, N_pde, epochs):

        print(t_bdry)
        self._eqn = eqn
        self._inits = inits
        self._t_bdry = t_bdry
        self._N_pde = N_pde
        self._epochs = epochs
        self._order = order
       #de_points, inits, order, t0, epochs, eqn

        print(self._t_bdry)
        self._t = np.linspace(self._t_bdry[0], self._t_bdry[1], self._N_pde)

        self._de_points = points.defineCollocationPoints(self._t_bdry, self._N_pde)

        self._epoch_loss, self._model = PINNtrain_IVP(self._de_points, 
               self._inits, self._order, self._t_bdry[0], self._epochs, self._eqn)

        self._solutionPred = self._model(np.expand_dims(self._t, axis=1))[:,0]
    
    
    def plot_epoch_loss(self, title = "ODE-Epoch-Loss"):
        plotters.plot_epoch_loss(self._epoch_loss, self._epochs, title)
        return
    
    
    def plot_solution_predition(self, title = "ODE-solution-pred"):
        plotters.plot_solution_prediction(self._t, self._solutionPred, title)
        return