import subprocess as sp
from tempfile import NamedTemporaryFile
import PIL.Image as Image

class GrimImager:
    def __init__(self, display):
        self.display_ = display

    def capture_area(self, bbox):
        with NamedTemporaryFile(suffix=".png") as fp:
            grim = ["grim", "-g", f"{2560},{bbox[1]} {bbox[2]}x{bbox[3]}", fp.name]
            proc = sp.run(grim)
            im = Image.open(fp.name)
            im.load()
            return im
