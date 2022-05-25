import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse


def read_img_in_gray(file, coefs=None):
    if coefs is None:
        coefs = [0.2126, 0.7152, 0.0722]

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], coefs)

    img = cv2.imread(file, cv2.COLOR_BGR2RGB)
    return rgb2gray(img)


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.abspath(path)


PATH_TO_GRAY_IMGS = make_dir("gray_images")

RESULT_DIR = make_dir("res")
RESULT_HM_DIR = make_dir(os.path.join(RESULT_DIR, "heatmaps"))
RESULT_DIF_DIR = make_dir(os.path.join(RESULT_DIR, "diff"))

COLORMAP = 'seismic'
# COLORMAP = 'YlOrBr'
BIT_IMAGES_LEN = 2 ** 8 - 1


def get_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]


class HeatMap:
    def __init__(self, file1, file2, method=0, use_gui=False):
        if not os.path.isfile(file1) or not os.path.isfile(file2):
            raise FileNotFoundError(file1 + " or " + file2 + " not exist")
        self.__file_name1 = get_file_name(file1)
        self.__file_name2 = get_file_name(file2)
        self.__diff_save_path = os.path.join(RESULT_DIF_DIR, f"{self.__file_name1}_{self.__file_name2}.png")
        self.__heatmap_path = os.path.join(RESULT_HM_DIR, f"{self.__file_name1}_{self.__file_name2}.png")
        self.__method = method + 1
        self.__use_gui = use_gui
        gray_image = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
        gray_image2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

        # gray_image = read_img_in_gray(file1)
        # gray_image2 = read_img_in_gray(file2)

        # self.__gray_image = read_img_in_gray(file1)
        # self.__gray_image2 = read_img_in_gray(file2)

        self.__image1 = plt.imread(file1)
        self.__image2 = plt.imread(file2)
        # self.__image2 = self.__gray_image2
        self.__ar1 = np.asarray(gray_image, float)
        self.__ar2 = np.asarray(gray_image2, float)
        if self.__ar1.shape != self.__ar2.shape:
            raise ValueError("images have different shapes")
        self.__diff_arr = None

    def __make_diff(self):
        if self.__diff_arr is not None:
            return
        d = []

        # способ 0
        # d = self.__ar1 - self.__ar2

        # способ 1
        if self.__method == 1:
            # d = self.__ar1 - self.__ar2

            d = np.abs(self.__ar1 - self.__ar2)
            mx = np.max(d)
            d = d / mx
            d *= BIT_IMAGES_LEN

        # способ 2
        if self.__method == 2:
            mx = np.vstack((self.__ar1.ravel(), self.__ar2.ravel())).max(axis=0).reshape(self.__ar1.shape)
            d = np.abs(self.__ar1 - self.__ar2) / mx
            d[np.isnan(d)] = 0.0
            d = d * BIT_IMAGES_LEN

        # способ 3
        if self.__method == 3:
            d = self.__ar1 / (self.__ar2 + 0.01)
            d = np.log(d)
            mx = np.max(d)
            if mx == 0:
                mx = 0.0001
            d = d / mx
            d *= BIT_IMAGES_LEN

        # print(d)
        self.__diff_arr = d

    def calc_statistics(self):
        self.__make_diff()
        window_size = 10
        a = self.__diff_arr
        res = 0
        for i in range(len(a) - window_size + 1):
            for j in range(len(a[i]) - window_size + 1):
                res = max(res, a[i:i + window_size, j:j + window_size].mean())
        print(res / BIT_IMAGES_LEN * 100, "% - square")
        print(np.mean(self.__diff_arr) / BIT_IMAGES_LEN * 100, "% - mean")
        print(np.max(self.__diff_arr) / BIT_IMAGES_LEN * 100, "% - max")

    def __create_heatmap(self):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if self.__use_gui:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = plt.figure(figsize=(12, 3))
            ax = fig.add_subplot(1, 3, 3)

        img = ax.imshow(self.__diff_arr, cmap=COLORMAP, aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='5%', pad=0.1)
        cbar = plt.colorbar(img, cax=cax, ax=ax)
        mn = np.min(self.__diff_arr)
        sr = np.mean(self.__diff_arr)
        mx = np.max(self.__diff_arr)
        cbar.set_ticks([mn, sr, mx])
        cbar.set_ticklabels(["0 %", "50 %", "100 %"])
        # cbar.set_ticks([0, BIT_IMAGES_LEN / 4, BIT_IMAGES_LEN / 2, BIT_IMAGES_LEN * 3 / 4, BIT_IMAGES_LEN - 1])
        # cbar.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
        if not self.__use_gui:
            plt.title("Heatmap", loc='right')

            img2 = fig.add_subplot(1, 3, 1)
            # img2.imshow(self.__image1, aspect='auto')
            img2.imshow(self.__ar1, aspect='auto', cmap='gray')
            plt.title(self.__file_name1)

            img3 = fig.add_subplot(1, 3, 2)
            # img3.imshow(self.__image2, aspect='auto')
            img3.imshow(self.__ar2, aspect='auto', cmap='gray')
            plt.title(self.__file_name2)

        plt.savefig(self.__heatmap_path)

        if not self.__use_gui:
            plt.show()

    def __make_heatmap_from_gray(self):
        self.__make_diff()
        cv2.imwrite(self.__diff_save_path, self.__diff_arr)
        self.__create_heatmap()

    def get_path(self):
        return self.__heatmap_path

    def create(self):
        self.__make_heatmap_from_gray()


def heatmap_from_one(origin_file):
    if not os.path.isfile(origin_file):
        print("File " + origin_file + " not exist")
        return

    gray1 = read_img_in_gray(origin_file)
    r, g = 0.1, 0.2
    b = 1 - r - g
    gray2 = read_img_in_gray(origin_file, [r, g, b])
    file_name = get_file_name(origin_file)
    file_1 = os.path.join(PATH_TO_GRAY_IMGS, file_name + '_gray1.jpg')
    file_2 = os.path.join(PATH_TO_GRAY_IMGS, file_name + '_gray2.jpg')
    cv2.imwrite(file_1, gray1)
    cv2.imwrite(file_2, gray2)

    hm = HeatMap(origin_file, file_2)
    hm.create()


def main():
    global RESULT_DIR, RESULT_HM_DIR, RESULT_DIF_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--one', type=str, help='Path to one origin image file')
    parser.add_argument('-c', '--compare', type=str, nargs=2, help='Paths to images files')
    # parser.add_argument('-i', '--info', help='Print statistics for comparing')
    parser.add_argument('-s', '--save', type=str, help='Path to save directory')
    args = vars(parser.parse_args())
    if args['save']:
        RESULT_DIR = make_dir(args['save'])
        RESULT_HM_DIR = make_dir(os.path.join(RESULT_DIR, "heatmaps"))
        RESULT_DIF_DIR = make_dir(os.path.join(RESULT_DIR, "diff"))
    if args['one']:
        heatmap_from_one(args['one'])
    elif args['compare']:
        try:
            hm = HeatMap(args['compare'][0], args['compare'][1])
            hm.create()
            # if args['info']:
            # hm.calc_statistics()
        except Exception as e:
            print(e)
    else:
        print("Not enough arguments.")
        parser.print_help()


if __name__ == '__main__':
    main()
