from Configuration.configuarion import Config
from NTS_Manager.manager import NTS_Manager

if __name__=="__main__":

    content_path, style_path = NTS_Manager.get_paths(Config)
    # print("content_path " + content_path, "style_path " + style_path, sep = '\n')
    content_image, style_image = NTS_Manager.read_data(content_path, style_path)

    extractor, style_targets, content_targets = NTS_Manager.prepare_targets(content_path, style_path, Config, content_image, style_image)
    # print("style_targets " + str(style_targets), "content_targets " + str(content_targets), sep = '\n')
    image, opt = NTS_Manager.assign_input_and_prepare_optimizer(content_image, Config)
    # print("opt\n", opt)
    NTS_Manager.execute(Config, opt, image, extractor, style_targets, content_targets)
