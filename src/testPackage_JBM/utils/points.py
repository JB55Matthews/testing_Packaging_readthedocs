from pyDOE import lhs

def defineCollocationPoints(t_bdry, N_pde=100):

  """
  Generates collocation points network trains over

  Args:
    t_bdry (list): list containing two values, interval to solve equation on
    N_pde (int): Number of selection points the network learns along
  
  Returns:
    ode_points (list): randomly sampled ode_points
  
  """

    # Sample points where to evaluate the PDE
  ode_points = t_bdry[0] + (t_bdry[1] - t_bdry[0])*lhs(1, N_pde)

  return ode_points