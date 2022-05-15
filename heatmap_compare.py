import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.abspath(path)


PATH_TO_IMGS = make_dir("images")
PATH_TO_GRAY_IMGS = make_dir(os.path.join(PATH_TO_IMGS, "gray"))
RESULT_DIR = make_dir("res")
RESULT_HM_DIR = make_dir(os.path.join(RESULT_DIR, "heatmaps"))
RESULT_DIF_DIR = make_dir(os.path.join(RESULT_DIR, "diff"))

N_GRAY = 1
diff_save_path = ""
need_save_gray = False
need_save_heat_map = True
file1 = ""
file2 = ""
COLORMAP = 'seismic'


def make_diff(ar1, ar2):
    assert ar1.shape == ar2.shape
    arr_diff = np.abs(ar1 - ar2)
    mx = np.max(arr_diff)
    # нормализация
    arr_diff = arr_diff / mx
    arr_diff *= 255
    return arr_diff


def get_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]


def make_gray_img(file, coefs=None):
    global N_GRAY
    if coefs is None:
        coefs = [0.333, 0.333, 0.334]

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], coefs)

    img = cv2.imread(file, cv2.COLOR_BGR2RGB)
    # img = img[0:250, 0:250]
    gray_img = rgb2gray(img)
    if need_save_gray:
        file_name = get_file_name(file) + str(N_GRAY)
        N_GRAY += 1
        file_name = os.path.join(PATH_TO_GRAY_IMGS, file_name + '_gray.jpg')
        cv2.imwrite(file_name, gray_img)
    return gray_img


def make_heat_map(arr, ar1=None, ar2=None):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    rc = {"axes.spines.left": False,
          "axes.spines.right": False,
          "axes.spines.bottom": False,
          "axes.spines.top": False,
          "xtick.bottom": False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
          "ytick.left": False}
    plt.rcParams.update(rc)
    fig = plt.figure(figsize=(4, 4))
    # fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax2 = fig.add_subplot(1, 4, 4)
    # ax = img1.gca()
    # img = ax.matshow(arr, cmap='RdYlGn_r')
    img = ax.imshow(arr, cmap=COLORMAP, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%', pad=0.1)
    cbar = plt.colorbar(img, cax=cax, ax=ax)
    cbar.set_ticks([0, 255 / 4, 255 / 2, 255 * 3 / 4, 254])
    cbar.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    plt.title("Heatmap", loc='right')
    # plt.colorbar(img, ax=ax, format='%')

    # if ar1 is not None and ar2 is not None:
    #     img2 = fig.add_subplot(1, 3, 1)
    #     img2.imshow(ar1, cmap='gray', aspect='auto')
    #     plt.title(file1 if file1 else "first")
    #     img3 = fig.add_subplot(1, 3, 2)
    #     img3.imshow(ar2, cmap='gray', aspect='auto')
    #     plt.title(file2 if file2 else "second")

        # fig.suptitle('Heatmaps with `Axes.matshow`', fontsize=16)
    res_heatmap = os.path.join(RESULT_HM_DIR, f"{file1}_{file2}.png")
    plt.savefig(res_heatmap)
    return res_heatmap
    # plt.show()


def make_heat_map_from_gray(gray_image, gray_image2):
    ar1 = np.asarray(gray_image, float)
    ar2 = np.asarray(gray_image2, float)
    diff_arr = make_diff(ar1, ar2)
    if diff_save_path:
        cv2.imwrite(diff_save_path, diff_arr)
    return make_heat_map(diff_arr, ar1, ar2)


def make_two_gray(gray_file, gray_file2):
    global file1, file2, diff_save_path
    if not os.path.isfile(gray_file2) or not os.path.isfile(gray_file):
        print("File " + gray_file + " or " + gray_file2 + " not exist")
        return
    file1 = get_file_name(gray_file)
    file2 = get_file_name(gray_file2)
    diff_save_path = os.path.join(RESULT_DIF_DIR, f"{file1}_{file2}.png")

    gray_image = cv2.imread(gray_file, cv2.IMREAD_GRAYSCALE)
    gray_image2 = cv2.imread(gray_file2, cv2.IMREAD_GRAYSCALE)
    return make_heat_map_from_gray(gray_image, gray_image2)


def make_with_origin(origin_file):
    if not os.path.isfile(origin_file):
        print("File " + origin_file + " not exist")
        return

    gray1 = make_gray_img(origin_file)
    # gray2 = make_gray_img(origin_file)
    r, g = 0.1, 0.2
    b = 1 - r - g
    gray2 = make_gray_img(origin_file, [r, g, b])
    make_heat_map_from_gray(gray1, gray2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--origin', type=str, help='Path to origin file')
    parser.add_argument('-c', '--compare', type=str, nargs=2, help='Paths to 2 gray files')
    parser.add_argument('-m', '--heat_map', type=str, help='Paths to diff file')
    parser.add_argument('-s', '--save', type=str, help='Path for save file')
    args = vars(parser.parse_args())
    if args['save']:
        diff_save_path = os.path.join(RESULT_DIR, args['save'] + "_diff.jpg")
    if args['origin']:
        make_with_origin(args['origin'])
    elif args['compare']:
        make_two_gray(args['compare'][0], args['compare'][1])
    elif args['heat_map']:
        diff_img = cv2.imread(args['heat_map'], cv2.IMREAD_GRAYSCALE)
        make_heat_map(diff_img)
    else:
        print("Not enough arguments.")
        parser.print_help()
