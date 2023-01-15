# Ejemplo de entrenamiento de una GAN condicional con el conjunto de datos fashion mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
import time

# Definición independiente del modelo discriminador
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# entrada de etiqueta
	in_label = Input(shape=(1,))
	# incrustación para entrada categórica
	li = Embedding(n_classes, 50)(in_label)
	# escalar las dimensiones de la imagen con activación lineal
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# remodelar a canal adicional
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# entrada de imagen
	in_image = Input(shape=in_shape)
	# concatena la etiqueta como un canal
	merge = Concatenate()([in_image, li])
	# reducción de resolución (submuestreo)
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# reducción de resolución (submuestreo)
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# aplanar mapas de características
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# salida
	out_layer = Dense(1, activation='sigmoid')(fe)
	# definición del modelo
	model = Model([in_image, in_label], out_layer)
	# compilación de modelo
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# Definición independiente del modelo generador
def define_generator(latent_dim, n_classes=10):
	# entrada de etiqueta
	in_label = Input(shape=(1,))
	# incrustación para entrada categórica
	li = Embedding(n_classes, 50)(in_label)
	# multiplicación lineal
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# remodelar a canal adicional
	li = Reshape((7, 7, 1))(li)
	# entrada del generador de imágenes
	in_lat = Input(shape=(latent_dim,))
	# base para la imagen de 7x7
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# fusionar la generación de imágenes y la entrada de etiquetas
	merge = Concatenate()([gen, li])
	# muestreo ascendente a 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# muestreo ascendente a 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# salida
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# definición del modelo
	model = Model([in_lat, in_label], out_layer)
	return model

# definir el modelo combinado de generador y discriminador, 
# para actualizar el generador
def define_gan(g_model, d_model):
	# hacer los pesos del discriminador no entrenables
	d_model.trainable = False
	# obtener entradas de ruido y etiquetas del modelo del generador
	gen_noise, gen_label = g_model.input
	# obtener imagen de salida del modelo generador
	gen_output = g_model.output
	# conecte la salida de la imagen y etiquete la entrada del generador 
    # como entradas al discriminador
	gan_output = d_model([gen_output, gen_label])
	# definir el modelo gan para tomar ruido, etiquetar
    # y generar una clasificación
	model = Model([gen_noise, gen_label], gan_output)
	# compilación del modelo
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# carga de imágenes fashion mnist
def load_real_samples():
	# cargar conjunto de datos
	(trainX, trainy), (_, _) = load_data()
	# expander a 3d, e.g. agregar canales
	X = expand_dims(trainX, axis=-1)
	# convertir de enteros a flotantes
	X = X.astype('float32')
	# escalar de [0,255] a [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainy]

# # seleccionar muestras reales
def generate_real_samples(dataset, n_samples):
	# dividir en imágenes y etiquetas
	images, labels = dataset
	# elegir instancias aleatorias
	ix = randint(0, images.shape[0], n_samples)
	# seleccionar imágenes y etiquetas
	X, labels = images[ix], labels[ix]
	# generar etiquetas de clase
	y = ones((n_samples, 1))
	return [X, labels], y

# generar puntos en el espacio latente como entrada para el generador
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generar puntos en el espacio latente
	x_input = randn(latent_dim * n_samples)
	# remodelar en un lote de entradas para la red
	z_input = x_input.reshape(n_samples, latent_dim)
	# generar etiquetas
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use el generador para generar n ejemplos falsos, 
# con etiquetas de clase
def generate_fake_samples(generator, latent_dim, n_samples):
	# genera puntos en el espacio latente
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predecir salidas
	images = generator.predict([z_input, labels_input])
	# crear etiquetas de clase
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# Entrenar al generador y discriminador
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# enumerar épocas manualmente
	for i in range(n_epochs):
		# enumerar lotes sobre el conjunto de entrenamiento
		for j in range(bat_per_epo):
			# obtener muestras 'reales' seleccionadas al azar
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# actualizar los pesos del modelo discriminador
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generar ejemplos falsos
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# actualizar los pesos del modelo discriminador
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# preparar puntos en el espacio latente como entrada para el generador
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# crear etiquetas invertidas para las muestras falsas
			y_gan = ones((n_batch, 1))
			# actualizar el generador a través del error del discriminador
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# resumir la pérdida en este lote
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# guardar el modelo del generador
	g_model.save('cgan_generator.h5')


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tamaño del espacio latente
latent_dim = 250
# crear el discriminador
d_model = define_discriminator()
# crear el generador
g_model = define_generator(latent_dim)
# crear la gan
gan_model = define_gan(g_model, d_model)
# cargar datos de imagen
dataset = load_real_samples()

start_time = time.time()
# entrenar el modelo
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100)
print("--- %s seconds ---" % (time.time() - start_time))