
from .pinn.order1pinnClass import ode_solution

def solve_order1ODE_pinn(eqn, order, init_data, t_bdry=[0,1], N_pde=100, epochs=1000):
    """
    Solves a first order ode

    Args:
        eqn (string): string equation
        order (int): order of equation to solve, package supports only 1
        init_data (list): list containting single inital value of u(t)
        t_bdry (list): list containing two values, interval to solve equation on
        N_pde (int): Number of selection points the network learns along
        epochs (int): epochs network is trained for

    Returns:
        ode_solution (obj): object containing trained network with plotting functions
    
    """


    solObj = ode_solution(eqn, order, init_data, t_bdry, N_pde, epochs)

    return solObj