# ejemplo de carga del modelo generador y generación de imágenes
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from matplotlib import pyplot

# generar puntos en el espacio latente como entrada para el generador
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# genera puntos en el espacio latente
	x_input = randn(latent_dim * n_samples)
	# remodela un lote de entradas para la red
	z_input = x_input.reshape(n_samples, latent_dim)
	# genera etiquetas
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# crear y guardar un gráfico de imágenes generadas
def save_plot(examples, n):
	# trazo de imágenes
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# apagar el eje
		pyplot.axis('off')
		# trazar datos de píxeles sin procesar
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

espacio_latente = 500
tam_incrustacion = 200
# carga el modelo
model = load_model('./cgan_generator.h5')
# genera imágenes
latent_points, labels = generate_latent_points(espacio_latente, 100)
# especificar etiquetas
labels = asarray([x for _ in range(10) for x in range(10)])
# genera imágenes
X  = model.predict([latent_points, labels])
# escala de [-1,1] a [0,1]
X = (X + 1) / 2.0
# Traza los resultados
save_plot(X, 10)

