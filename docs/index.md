# Welcome to test_Package_readthedocs!

Documentation for project testing packaging and documenting a project

## Getting Started

Package can be installed with: pip install -i https://test.pypi.org/simple/ testPackage-JBM

Then see Example Usage for how to use solver, and read API for all function explanation

## Example Usage

    import testPackage-JBM.solvers as slv

    eqn = "ut+u"
    order = 1
    init_data = [0.5]
    t_bdry = [0,1]
    N_pde = 100
    epochs = 1000

    mymodel = slv.solve_order1ODE_pinn(eqn, order, init_data, t_bdry, N_pde, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()
