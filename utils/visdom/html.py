import os
import dominate
from dominate.tags import h3, table, tr, td, p, a, img, br



class HTML:
    """"""
    """HTML class write image and text into a html file
       it has the following functions:

       --- <add_header>     add header to a html file
       --- <add_images>     add a row of image to a html file
       --- <save>           save html file to the disk

       it base on a Python libary named "dominate" which use DOM API to create and operate html file
    """

    def __init__(self, web_dir, title):
        """"""
        """ initial HTML class

        args:
            web_dir (str) -- dir path for store the html. HTML file will be saved in <web_dir>/index.html and iamge will be saved in <web_dir/images/ by default
            title (str)   -- html title
            refresh (int) -- refresh frequencey, do not refresh when set <refresh = 0>
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)

    def get_image_dir(self):
        """"""
        """get image folder path"""
        return self.img_dir

    def add_header(self, text):
        """"""
        """ insert head text to html

        args:
            text (str) -- header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """"""
        """ insert image to html

        args:
            ims (str list)   -- image path List
            txts (str list)  -- image name List
            links (str list) -- image super-link List
        """
        t = table(border=1, style="table-layout: fixed;")  # insert table
        self.doc.add(t)
        with t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """"""
        """ save current to html file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()