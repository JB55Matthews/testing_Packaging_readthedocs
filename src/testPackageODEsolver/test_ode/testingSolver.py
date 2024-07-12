import solvers as solvers
mymodel = solvers.solve_order1ODE_pinn("ut + u", 1, [0.5], [0,1], 100, 10)

mymodel.plot_epoch_loss()

mymodel.plot_solution_predition()