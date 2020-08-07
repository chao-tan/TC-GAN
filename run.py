import time
from datasets import create_dataset
from modules import create_model
from utils.visdom.visualizer import Visualizer
from utils import startup
import os
import utils.tools as util


def train(config):
    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)
    dataset_size = len(dataset)  # get the size of dataset
    print('The number of training images = %d' % dataset_size)
    visualizer = Visualizer(config)  # create visualizer to show/save iamge

    total_iters = 0  # total iteration for datasets points
    t_data = 0

    # outter iteration for differtent epoch; we save module via <epoch_count> and <epoch_count>+<save_latest_freq> options
    for epoch in range(1, int(config['niter']) + int(config['niter_decay']) + 1):
        epoch_start_time = time.time()  # note the starting time for current epoch
        iter_data_time = time.time()  # note the starting time for datasets iteration
        epoch_iter = 0  # iteration times for current epoch, reset to 0 for each epoch

        # innear iteration for single epoch
        for i, data in enumerate(dataset):
            iter_start_time = time.time()  # note the stating time for current iteration
            if total_iters % int(config['print_freq']) == 0:  # note during time each <print_freq> times iteration
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters = total_iters + int(config['train_batch_size'])
            epoch_iter = epoch_iter + int(config['train_batch_size'])
            model.set_input(data)  # push loading image to the module
            model.optimize_parameters()  # calculate loss, gradient and refresh module parameters

            if total_iters % int(config['display_freq']) == 0:  # show runing result in visdom each <display_freq> iterations
                save_result = total_iters % int(config['update_html_freq']) == 0  # save runing result to html each <update_html_freq> iteartions
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % int(config['print_freq']) == 0:  # print/save training loss to console each <print_freq> iterations
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / int(config['train_batch_size'])
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if int(config['display_id']) > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if epoch % int(config['save_epoch_freq']) == 0:  # save module each <save_epoch_freq> epoch iterations
            print('saving the module at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, int(config['niter']) +int(config['niter_decay']), time.time() - epoch_start_time))

        # update learning rate after each epoch
        model.update_learning_rate()



def test(config):
    config['num_threads'] = 1                     # only <num_threads = 1> supported when testing_usr
    config['flip'] = False                        # not allowed to flip image
    config['add_colorjit'] = False                # not allowed to use color jitting

    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)

    result_root_path = os.path.join(config['checkpoints_dir'], config['name'], config['results_dir'])
    util.mkdir(result_root_path)
    print(" create testing_usr folder: " + result_root_path)

    # set module to testing_usr mode
    model.eval()

    for i, data in enumerate(dataset):
        model.set_input(data)  # push test datasets to module
        model.test()  # forward module

        for k in range(len(model.test_result)):
            img = util.tensor2im(model.test_result[k][1])
            img_path = os.path.join(result_root_path,data['PATH'][0])
            util.save_image(img, img_path)

        print("Testing forward-- complete:" + str(i + 1) + "  total:" + str(dataset.__len__()))


if __name__ == '__main__':
    configs = startup.SetupConfigs(config_path='configs/tcgan_usr256.yaml')
    configs = configs.setup()

    if configs['status'] == "train":
        train(configs)

        configs['status'] = "test"
        test(configs)

    else:
        test(configs)

