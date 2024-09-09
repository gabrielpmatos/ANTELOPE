import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
arch_dir = "architectures_saved/"

## ------------------------------------------------------------------------------------
##				Classes
## ------------------------------------------------------------------------------------

class VAE(keras.Model):
    #developed from https://keras.io/examples/generative/vae/
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reco_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(data, reconstruction)))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(data, reconstruction)))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reco_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):
        z_mean,z_log_var,x = self.encoder(data)
        reconstruction = self.decoder(x)
        return {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }

## ------------------------------------------------------------------------------------
class supervisedPFN(keras.Model):
  def __init__(self,graph, classifier):
    super().__init__()
    self.graph = graph
    self.classifier = classifier
  
  def call(self, X, *args):
    graph_rep = self.graph(X)
    result = self.classifier(graph_rep)
    return result

## ------------------------------------------------------------------------------------
## 		Functions
## ------------------------------------------------------------------------------------

## ------------------------------------------------------------------------------------
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


## ------------------------------------------------------------------------------------
def pfn_mask_func(X, mask_val=0):
  # map mask_val to zero and return 1 elsewhere
  return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))

## ------------------------------------------------------------------------------------
def get_full_PFN(input_dim, phi_dim):
  initializer = keras.initializers.HeUniform()
  loss = keras.losses.CategoricalCrossentropy()
  optimizer = keras.optimizers.Adam(learning_rate=0.001) 

  input_dim_x = input_dim[0]
  input_dim_y = input_dim[1]

  #input
  pfn_inputs = keras.Input(shape=(None,input_dim_y))
  masked = keras.layers.Lambda(pfn_mask_func, name="mask")(pfn_inputs)

  # Phi network
  dense1 = keras.layers.Dense(75, kernel_initializer=initializer, name="pfn1")
  x = keras.layers.TimeDistributed(dense1, name="tdist_0")(pfn_inputs)
  x = keras.layers.Activation('relu')(x)
  dense2 = keras.layers.Dense(75, kernel_initializer=initializer, name="pfn2") 
  x = keras.layers.TimeDistributed(dense2, name="tdist_1")(x)
  x = keras.layers.Activation('relu')(x)
  dense3 = keras.layers.Dense(phi_dim, kernel_initializer=initializer, name="phi") 
  x = keras.layers.TimeDistributed(dense3, name="tdist_2")(x)
  phi_outputs = keras.layers.Activation('relu')(x)

  # latent space
  sum_phi = keras.layers.Dot(1, name="sum")([masked,phi_outputs])
  graph = keras.Model(inputs=pfn_inputs, outputs=sum_phi, name="graph")
  graph.summary()
   
  # F network
  classifier_inputs = keras.Input(shape=(phi_dim,))
  x = classifier_inputs
  x = keras.layers.Dense(75, kernel_initializer=initializer, name = "F1")(classifier_inputs)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dense(75, kernel_initializer=initializer, name = "F2")(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dense(75, kernel_initializer=initializer, name="F3")(x)
  x = keras.layers.Activation('relu')(x)

  # output
  x = keras.layers.Dense(2, name="output")(x)
  output = keras.layers.Activation('softmax')(x)

  classifier = keras.Model(inputs=classifier_inputs, outputs=output, name="classifier")
  classifier.summary()

  pfn = supervisedPFN(graph, classifier)
  pfn.compile(optimizer=optimizer, loss=loss)
  return pfn, graph

## ------------------------------------------------------------------------------------
def get_variational_encoder(input_dim, encoding_dim, latent_dim):
  inputs = keras.Input(shape=(input_dim,))
  x = Dropout(0.1, input_shape=(input_dim,))(inputs)
  x = Dense(32)(x)
  x = Dropout(0.1)(x)
  x = LeakyReLU(alpha=0.3)(x)
  x = Dense(encoding_dim)(x)
  x = Dropout(0.1)(x)
  x = LeakyReLU(alpha=0.3)(x)
  x = Dense(encoding_dim/2.)(x)
  x = Dropout(0.1)(x)
  x = LeakyReLU(alpha=0.3)(x)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  
  encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()
  return encoder

## ------------------------------------------------------------------------------------
def get_decoder(input_dim, encoding_dim, latent_dim):
  latent_inputs = keras.Input(shape=(latent_dim,))
  x = keras.layers.Dense(encoding_dim, activation="relu")(latent_inputs)
  x = keras.layers.Dense(encoding_dim*2., activation="relu")(x)
  decoder_outputs = keras.layers.Dense(input_dim, activation="relu")(x)

  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()
  return decoder

## ------------------------------------------------------------------------------------
def get_vae(input_dim, encoding_dim, latent_dim):
  encoder = get_variational_encoder(input_dim, encoding_dim, latent_dim)
  decoder = get_decoder(input_dim, encoding_dim, latent_dim)

  vae = VAE(encoder, decoder)
  vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001))
  return vae

## ------------------------------------------------------------------------------------
def get_model(model_name, input_dims, encoding_dim, latent_dim, phi_dim=None):

  if 

  if (model_name == "VAE"):
    return get_vae(input_dims, encoding_dim, latent_dim)

  else:
    print("ERROR: model name", model_name," not recognized")

