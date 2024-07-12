import numpy as np
import tensorflow as tf

# Define the network
def build_model(nr_units=40, nr_layers=4, summary=True):

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model