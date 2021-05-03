class Config():

    # PATHS
    # content_img_fn = 'YellowLabradorLooking_new.jpg'
    # content_img_fn = "content.jpeg"
    # content_img_fn = "DQTV_TET_2021.jpg"
    content_img_fn = "2020-02-02.jpg"
    # content_img_path = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    # content_img_path = "E:/ML_PROJECT/coursera-deep-learning-specialization/C4 - Convolutional Neural Networks/Week 4/Neural Style Transfer/images/"
    content_img_path = "C:/Users/ASUS/Downloads/"
    style_img_fn = 'kandinsky5.jpg'
    # style_img_fn = "stone_style.jpg"
    # style_img_fn = "drop-of-water.jpg"
    style_img_fn = "The-Starry-Night.jpg"
    # style_img_path = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
    # style_img_path = "E:/ML_PROJECT/coursera-deep-learning-specialization/C4 - Convolutional Neural Networks/Week 4/Neural Style Transfer/images/"
    style_img_path = "C:/Users/ASUS/Downloads/"
    hub_model_location = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    save_img_fn = "styled_image.png"
    save_img_path = "C:/Users/ASUS/Downloads/"

    # STYLE AND CONTENT LAYERS
    content_layers = ['block5_conv2']

    num_content_layers = len(content_layers)

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_style_layers = len(style_layers)

    # WEIGHTS
    style_weight = 1e-2

    content_weight = 1e4

    total_variation_weight = 30

    # Adam Optimizer
    learning_rate = 0.02

    beta_1 = 0.99

    epsilon = 1e-1

    # Running
    epochs = 5

    steps_per_epoch = 5

    # Names
    output_image_name = "Styled Image"
