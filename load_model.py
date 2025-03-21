import numpy as np
from PIL import Image, ImageDraw, ImageFilter,ImageOps
class network:
    def __init__(self):
        self.soft = lambda x:(1/(1+np.exp(-x)))
        self.w = self.setup_weight()
        self.b = self.setup_b()
    def anh(self,img : Image):
        arr_img = np.array(img)
        arr_img = arr_img.reshape(1,-1)
        return 255 - arr_img
    def setup_b(self):
        loaded = np.load('arrays_of_b.npz')
        # Chuyển đổi dữ liệu đã tải thành một danh sách các mảng numpy
        b = [loaded[key] for key in loaded]
        return b
    def setup_weight(self):
        loaded = np.load('arrays_of_w.npz')
        # Chuyển đổi dữ liệu đã tải thành một danh sách các mảng numpy
        w = [loaded[key] for key in loaded]
        return w
    def coputing(self,a):
        c = [a]
        a = a.T
        for index, i in enumerate(self.w):
            k = self.b[index]
            sd = i @ a
            d = sd.shape
            op = k.shape
            a = self.soft(sd + k)
            c.append(a)
        return c
if __name__ == "__main__":
    pass