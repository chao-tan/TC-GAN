import numpy as np
import os
import sys
import ntpath
import time
from utils import tools
from subprocess import Popen, PIPE
from utils.visdom import html

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError





class Visualizer(object):
    """"""
    """ this class include some function to visualize/save iamge and print/save log message

        it use "visdom" to visualize and "dominate" to cerate image html file
    """

    def __init__(self, config):
        """"""
        """ initial Visualizer class

        args:
            config --  
            
        step 1: cache configs;
        step 2: connect to a visdom server;
        step 3: create html object to save the iamge and text;
        step 4: cerate a log file to save traing loss
        """

        self.config = config
        self.display_id = int(config['display_id'])
        self.use_html = config['status'] == "train" and bool(config['use_visdom']) == True
        self.win_size = int(config['display_winsize'])
        self.name = config['name']
        self.port = int(config['display_port'])
        self.saved = False

        # using modify <display_port> and <display_server> to connect visdom server
        if self.display_id > 0:
            import visdom
            self.ncols = int(config['display_ncols'])
            self.vis = visdom.Visdom(server=config['display_server'], port=self.port, env=config['display_env'])
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # create html objerct in <checkpoints_dir>/web/>; images will be saved in <checkpoints_dir>/web/images/>
        if self.use_html:
            self.web_dir = os.path.join(config['checkpoints_dir'], self.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            tools.mkdirs([self.web_dir, self.img_dir])

        # cerate a log file to save the training loss
        self.log_name = os.path.join(config['checkpoints_dir'], self.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)



    def reset(self):
        """"""
        """ reset self.saved status"""

        self.saved = False



    def create_visdom_connections(self):
        """"""
        """ if can not connect to visdom server, this function will startup a new server using port <self.port> """

        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)



    def display_current_results(self, visuals, epoch, save_result):
        """"""
        """ diaplay current result in visdom and save to html

        args:
            visuals (OrderedDict) - -     image OrderedDict to visulize of save
            epoch (int) - -               current epoch
            save_result (bool) - -        save current result to html or not
        """

        # show image in browser using visdom
        if self.display_id > 0:
            ncols = self.ncols
            # show all the iamge in one visdom web pannel
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create CSS of the tabel

                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = tools.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = tools.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):
            self.saved = True

            for label, image in visuals.items():
                image_numpy = tools.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                tools.save_image(image_numpy, img_path)

            # refresh html
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()



    def plot_current_losses(self, epoch, counter_ratio, losses):
        """"""
        """ plot current loss image, loss label and value in visdom

        args:
            epoch (int)           -- current poch
            counter_ratio (float) -- current epoch progress , range from 0 to 1
            losses (OrderedDict)  -- OrderedDict to save name/value pair
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()



    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """"""
        """ print current loss and save to disk

        args:
            epoch (int) --  current epoch
            iters (int) --  iteration times in current epoch (reset to 0 in each epoch)
            losses (OrderedDict) -- OrderedDict to save name/value pair
            t_comp (float) -- runing time for each datapoint
            t_data (float) -- loading time for each datapoint
        """
        message = '(epoch: %d, iters: %d, time: %.3f, datasets: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save message




def save_images(webpage, visuals, image_path, width=256):
    """"""
    """ this function save images in "visuals" to html file.

    args:
        webpage (the HTML class)   -- the html class to save the images (see html.py for details)
        visuals (OrderedDict)      -- OrderedDict to save name/iamges
        image_path (str)           -- iamge path
        aspect_ratio (float)       -- the aspect ratio for saving image (default 1.0)
        width (int)                -- set iamge width, the image will be scaled width x width size (default 256)

    """

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tools.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        tools.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)