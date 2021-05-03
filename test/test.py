import matplotlib.pyplot as plt
import tensorflow as tf
from Helpers.utils import Utilities
from Configuration.configuarion import Config
from NTS_Manager.manager import NTS_Manager
import tensorflow_hub as hub

content_path, style_path = NTS_Manager.get_paths(Config)
# ====================================================================================
content_image = Utilities.load_img(content_path)
style_image = Utilities.load_img(style_path)
# ====================================================================================

# VISUALIZE
# ====================================================================================
# plt.subplot(1, 2, 1)
# Utilities.imshow(content_image, 'Content Image')
#
# plt.subplot(1, 2, 2)
# Utilities.imshow(style_image, 'Style Image')
#
# plt.show()
# ====================================================================================

# MAKE USE PRE-TRAINED MODEL
# ====================================================================================
# hub_model = hub.load(Config.hub_model_location)
# stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
# Utilities.imshow(stylized_image, 'Style Image')
# plt.show()
# ====================================================================================

# OBJECT DETECTION
# ====================================================================================
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
# print("prediction_probabilities.shape\n", prediction_probabilities.shape)
#
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])
# ====================================================================================

# vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#
# for layer in vgg.layers:
#     print(layer.name)

# num_content_layers = len(content_layers)
# num_style_layers = len(style_layers)
#
# print("num_content_layers " + str(num_content_layers), "num_style_layers " + str(num_style_layers), sep='\n')

# style_extractor = Utilities.vgg_layers(Config.style_layers)
# style_outputs = style_extractor(style_image*255)
#
# #Look at the statistics of each layer's output
# for name, output in zip(Config.style_layers, style_outputs):
#     print(name)
#     print("  shape: ", output.numpy().shape)
#     print("  min: ", output.numpy().min())
#     print("  max: ", output.numpy().max())
#     print("  mean: ", output.numpy().mean())

# results = extractor(tf.constant(content_image))
#
# print('Styles:')
# for name, output in sorted(results['style'].items()):
#     print("  ", name)
#     print("    shape: ", output.numpy().shape)
#     print("    min: ", output.numpy().min())
#     print("    max: ", output.numpy().max())
#     print("    mean: ", output.numpy().mean())
#     print()
#
# print("Contents:")
# for name, output in sorted(results['content'].items()):
#     print("  ", name)
#     print("    shape: ", output.numpy().shape)
#     print("    min: ", output.numpy().min())
#     print("    max: ", output.numpy().max())
#     print("    mean: ", output.numpy().mean())

# content_path = tf.keras.utils.get_file(Config.content_img_fn, Config.content_img_path)
# content_image = Utilities.load_img(content_path)

# x_deltas, y_deltas = Utilities.high_pass_x_y(content_image)
#
# plt.figure(figsize=(14, 10))
# plt.subplot(2, 2, 1)
# Utilities.imshow(Utilities.clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")
#
# plt.subplot(2, 2, 2)
# Utilities.imshow(Utilities.clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")
#
# x_deltas, y_deltas = Utilities.high_pass_x_y(image)
#
# plt.subplot(2, 2, 3)
# Utilities.imshow(Utilities.clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")
#
# plt.subplot(2, 2, 4)
# Utilities.imshow(Utilities.clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")
#
# plt.show()

# plt.figure(figsize=(14, 10))
#
# sobel = tf.image.sobel_edges(content_image)
# plt.subplot(1, 2, 1)
# Utilities.imshow(Utilities.clip_0_1(sobel[..., 0]/4+0.5), "Horizontal Sobel-edges")
# plt.subplot(1, 2, 2)
# Utilities.imshow(Utilities.clip_0_1(sobel[..., 1]/4+0.5), "Vertical Sobel-edges")
#
# plt.show()
# image = tf.Variable(content_image)
# print("Utilities.total_variation_loss(image).numpy() " + str(Utilities.total_variation_loss(image).numpy()))
#
# print("tf.image.total_variation(image).numpy() " + str(tf.image.total_variation(image).numpy()))
