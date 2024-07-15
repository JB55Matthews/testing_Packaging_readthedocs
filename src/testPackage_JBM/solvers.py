from .pinn.order1pinnClass import ode_solution
"""
solvers for ode

"""

def solve_order1ODE_pinn(eqn, order, init_data, t_bdry=[0,1], N_pde=100, epochs=1000):
    """
    
    Solves a first order ode
    
    """


    solObj = ode_solution(eqn, order, init_data, t_bdry, N_pde, epochs)

    return solObj
