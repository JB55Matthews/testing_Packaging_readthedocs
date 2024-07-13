from ..PDE.SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_periodic_soft as trainICBCP_periodic_soft
from ..PDE.SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_dirichlet_soft as trainBCP_dirichlet_soft
from ..PDE.SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_periodic as trainBCP_periodic
from ..PDE.SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_dirichlet_soft as trainICBCP_dirichlet_soft
from ..PDE.SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_hard as trainICBCP_hard
from ..PDE.SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_dirichlet_hard as trainBCP_dirichlet_hard
from ..PDE.SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_neumann_soft as trainBCP_neumann_soft
from ..PDE.SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_neumann_soft as trainICBCP_neumann_soft

from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_periodic_hard as trainICBCPDON_periodic_hard
from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_periodic_soft as trainICBCPDON_periodic_soft
from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_dirichlet_soft as trainICBCPDON_dirichlet_soft
from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_neumann_soft as trainICBCPDON_neumann_soft
from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP import pde_TrainingDeepONetBCP_periodic_hard as trainBCPDON_periodic_hard
from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP import pde_TrainingDeepONetBCP_dirichlet_soft as trainBCPDON_dirichlet_soft
from ..PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP import pde_TrainingDeepONetBCP_neumann_soft as trainBCPDON_neumann_soft


def PINNtrainSelect_tx(pde_points, init_points, epochs, eqn, t_order, N_pde, N_iv, setup_boundries, model, constraint, flag): #selecting which to train
   boundry_type = setup_boundries[0]

   if flag == "ICBCP-tx":
      if boundry_type == "periodic_timeDependent":
         if constraint == "soft":
            return trainICBCP_periodic_soft.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
         elif constraint == "hard":
               return trainICBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
         
      elif boundry_type == "dirichlet_timeDependent":
         if constraint == "soft":
            return trainICBCP_dirichlet_soft.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
         elif constraint == "hard":
            return trainICBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
         
      elif boundry_type == "neumann_timeDependent":
         if constraint == "soft":
            return trainICBCP_neumann_soft.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
   

def PINNtrainSelect_xy(pde_points, epochs, eqn, N_pde, setup_boundries, model, constraint, flag): #selecting which to train
   boundry_type = setup_boundries[0]

   if flag == "BCP-xy":
      if boundry_type == "dirichlet_timeIndependent":
         if constraint == "soft":
            return trainBCP_dirichlet_soft.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model)
         elif constraint == "hard":
            return trainBCP_dirichlet_hard.PINNtrain(pde_points, epochs, eqn, N_pde, model)
         
      elif boundry_type == "periodic_timeIndependent":
         if constraint == "soft":
            return trainBCP_periodic.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model)
         elif constraint == "hard":
            return trainBCP_periodic.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model)
         
      elif boundry_type == "neumann_timeIndependent":
         if constraint == "soft":
            return trainBCP_neumann_soft.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
def PINNtrainSelect_DeepONet_tx(pde_points, init_points, epochs, eqn, t_order, N_pde, N_iv, N_sensors, usensors, sensor_range, 
                                setup_boundries, model, constraint, flag): #selecting which to train
   
   boundry_type = setup_boundries[0]

   if flag == "DeepONet-ICBCP-tx":
      if boundry_type == "periodic_timeDependent":
         if constraint == "soft":
            return trainICBCPDON_periodic_soft.train(pde_points, init_points, usensors, t_order, epochs, model, eqn, N_iv)
         elif constraint == "hard":
               return trainICBCPDON_periodic_hard.train(pde_points, usensors, epochs, model, eqn)
         
      elif boundry_type == "dirichlet_timeDependent":
         if constraint == "soft":
            return trainICBCPDON_dirichlet_soft.train(pde_points, init_points, usensors, setup_boundries, t_order, epochs, model, eqn, N_iv)
         elif constraint == "hard":
            return #trainIBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
         
      elif boundry_type == "neumann_timeDependent":
         if constraint == "soft":
            return trainICBCPDON_neumann_soft.train(pde_points, init_points, usensors, setup_boundries, t_order, epochs, model, eqn, N_iv)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")


def PINNtrainSelect_DeepONet_xy(pde_points, epochs, eqn, N_pde, N_bc, N_sensors, usensors, sensor_range, setup_boundries, model, constraint, flag): #selecting which to train
   boundry_type = setup_boundries[0]

   if flag == "DeepONet-BCP-xy":
      if boundry_type == "dirichlet_timeIndependent":
         if constraint == "soft":
            return trainBCPDON_dirichlet_soft.train(pde_points, usensors, setup_boundries, epochs, model, eqn, N_bc)
         elif constraint == "hard":
            return trainBCPDON_periodic_hard.train(pde_points, usensors, epochs, model, eqn, N_bc)
         
      elif boundry_type == "periodic_timeIndependent":
         if constraint == "soft":
            return trainBCPDON_periodic_hard.train(pde_points, usensors, epochs, model, eqn, N_bc)
         elif constraint == "hard":
            return trainBCPDON_periodic_hard.train(pde_points, usensors, epochs, model, eqn, N_bc)
         
      elif boundry_type == "neumann_timeIndependent":
         if constraint == "soft":
            return trainBCPDON_neumann_soft.train(pde_points, usensors, setup_boundries, epochs, model, eqn, N_bc)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
        