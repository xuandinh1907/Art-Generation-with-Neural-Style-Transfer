from Helpers.utils import Utilities
from NTS_Model.model import StyleContentModel
import IPython.display as display
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

class NTS_Manager():

    @staticmethod
    def get_paths(config):
        if config.content_img_path.startswith("http"):
            content_path = tf.keras.utils.get_file(config.content_img_fn, config.content_img_path)
        else:
            content_path = config.content_img_path + config.content_img_fn
        if config.style_img_path.startswith("http"):
            style_path = tf.keras.utils.get_file(config.style_img_fn, config.style_img_path)
        else:
            style_path = config.style_img_path + config.style_img_fn
        return content_path, style_path

    @staticmethod
    def read_data(content_path, style_path):
        content_image = Utilities.load_img(content_path)
        style_image = Utilities.load_img(style_path)
        return content_image, style_image

    @staticmethod
    def prepare_targets(content_path, style_path, config, content_image, style_image):
        content_image = Utilities.load_img(content_path)
        style_image = Utilities.load_img(style_path)
        extractor = StyleContentModel(config.style_layers, config.content_layers)

        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']
        return extractor, style_targets, content_targets

    @staticmethod
    def assign_input_and_prepare_optimizer(content_image, config):
        image = tf.Variable(content_image)
        opt = tf.optimizers.Adam(learning_rate = config.learning_rate, beta_1 = config.beta_1, epsilon = config.epsilon)
        return image, opt

    @staticmethod
    def style_content_loss(outputs, extractor, style_targets, content_targets, config):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= config.style_weight / extractor.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= config.content_weight / extractor.num_content_layers
        loss = style_loss + content_loss
        return loss

    @staticmethod
    @tf.function()
    def train_step(config, image, opt, extractor, style_targets, content_targets):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = NTS_Manager.style_content_loss(outputs, extractor, style_targets, content_targets, config)
            loss += config.total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(Utilities.clip_0_1(image))

    @staticmethod
    def execute(config, opt, image, extractor, style_targets, content_targets):
        start = time.time()

        step = 0
        for n in range(config.epochs):
            for m in range(config.steps_per_epoch):
                step += 1
                NTS_Manager.train_step(config, image, opt, extractor, style_targets, content_targets)
                print(".", end='')
            # display.clear_output(wait=True)
            # display.display(Utilities.tensor_to_image(image))
            # print("================================================")
            print("Train step: {}".format(step))
        end = time.time()
        print("Total time: {:.1f}".format(end-start))

        Utilities.imshow(image, config.output_image_name)
        plt.show()
        Utilities.save_img(image, config)
        # plt.savefig(config.save_img_path + config.save_img_fn, dpi = 300)

