import numpy as np
import tensorflow as tf
import utils.model as models
import ast

#The main trainign function
@tf.function
def train_network_general_IVP(odes, inits, order, model, gamma, eqnparam):

    # DE collocation points
    t_de = odes[:,:1]
    
    # Initial value points
    t_init, u_init = inits[:1], inits[1:2]
  

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:

      # Inner gradient for derivatives of u wrt x and t
      with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_de)
        u = model(t_de)
        ut = tape2.gradient(u, t_de)
      
      t = t_de

      # Define the differential equation loss
      parse_tree = ast.parse(eqnparam, mode="eval")
      eqn = eval(compile(parse_tree, "<string>", "eval"))

      DEloss = tf.reduce_mean(tf.square(eqn))

      # Define the initial value loss
      with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(t_init)
        u_init_pred = model(t_init)

      IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init))
    
      
      # Composite loss function
      loss = DEloss + gamma*IVloss

    grads = tape.gradient(loss, model.trainable_variables)
    return DEloss, IVloss, grads


def PINNtrain_IVP(de_points, inits, order, t0, epochs, eqn):

  # Total number of collocation points
  N_de = len(de_points)

  # Batch size
  bs_de = N_de

  # Weight factor gamma
  gamma = 1.0

  # Learning rate
  lr_model = 1e-3

  
  t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
  

  epoch_loss = np.zeros(epochs)
  ivp_loss = np.zeros(epochs)
  de_loss = np.zeros(epochs)
  nr_batches = 0

  # Generate the tf.Dataset for the initial collocation points
 
  inits = np.column_stack([t_init, u_init])

  ds_init = tf.data.Dataset.from_tensor_slices(inits)
  ds_init = ds_init.cache()

  # Generate the tf.Dataset for the differential equations points
  ds_ode = tf.data.Dataset.from_tensor_slices(de_points.astype(np.float32))
  ds_ode = ds_ode.cache().shuffle(N_de).batch(bs_de)

  # Generate entire dataset
  ds = tf.data.Dataset.zip((ds_ode, ds_init))
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Generate the model
  opt = tf.keras.optimizers.Adam(lr_model)
  model = models.build_model()

  # Main training loop
  for i in range(epochs):

    # Training for that epoch
    for (des, inits) in ds:

      nr_batches = 0

      # Train the network
      DEloss, IVloss, grads = train_network_general_IVP(des, inits, order, model, gamma, eqn)

      # Gradient step
      opt.apply_gradients(zip(grads, model.trainable_variables))

      epoch_loss[i] += DEloss + gamma*IVloss
      ivp_loss[i] += IVloss
      de_loss[i] += DEloss
      nr_batches += 1

    # Get total epoch loss
    epoch_loss[i] /= nr_batches

    if (np.mod(i, 100)==0):
      print("DE loss, IV loss in {}th epoch: {: 6.4f}, {: 6.4f}.".format(i, DEloss, IVloss))

  return epoch_loss, model