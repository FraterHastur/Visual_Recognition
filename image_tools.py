from PIL import Image
from pylab import *
import numpy as np
import os


class JpgConverter:
    def __init__(self, file_path):
        self.path = file_path

    def __get_im_list(self):
        """Возвращает список файлов с заданными расширениями"""
        ext = ('.bmp', '.gif', '.ico', '.png', '.psd', '.tga', '.tiff', '.raw', '.svg')
        return [os.path.join(self.path, file) for file in os.listdir(self.path)
                if file.endswith(ext)]

    def convert_to_jpg(self):
        """Преобразует полученные файлы в формат . jpg"""
        # todo сделать возможность выбора в какой формат преобразовать
        file_list = self.__get_im_list()
        for infile in file_list:
            outfile = os.path.splitext(infile)[0] + '.jpg'
            if infile != outfile:
                try:
                    Image.open(infile).save(outfile)
                except IOError:
                    print('Can not convert file', infile)


class DrawLine:
    def __init__(self, file_path):
        # todo добавить описание цветов и линий
        self.path = file_path

    def __open_image(self):
        return array(Image.open(self.path))

    def __get_im_list(self, points_number):
        """Возвращает список с отмеченными точками"""
        imshow(self.__open_image())
        points_list = ginput(points_number)
        show()
        return points_list

    def draw_points(self, points_number, points_color='r', points_type='*', line_type=':', line_color='m'):
        points_list = self.__get_im_list(points_number)

        imshow(self.__open_image())
        x_points = []
        y_points = []

        for _, value in enumerate(points_list):
            x_points.append(value[0])
            y_points.append(value[1])
        x_points.append(x_points[0])
        y_points.append(y_points[0])

        draw_points = points_color + points_type
        draw_lines = line_color + line_type

        plot(x_points, y_points, draw_points)
        plot(x_points[:points_number + 1], y_points[:points_number + 1], draw_lines)
        show()


class ImageHistogram:
    def __init__(self, file_path, interval=128):
        self.path = file_path
        self.interval = interval

    def __open_image(self):
        return array(Image.open(self.path).convert('L'))

    def show_histogram(self):
        figure()
        gray()
        contour(self.__open_image(), origin='image')
        axis('equal')
        axis('off')
        figure()
        hist(self.__open_image().flatten(), self.interval)
        show()

    def hist_equation(self, nbr_bins=256):
        """Выравнивание гистограммы полутонового изображения"""
        im_hist, bins = np.histogram(self.__open_image().flatten(), nbr_bins, density=True)
        cdf = im_hist.cumsum()
        cdf = 255 * cdf / cdf[-1]

        image_eq = np.interp(self.__open_image().flatten(), bins[:-1], cdf)

        figure()
        gray()
        contour(image_eq.reshape(self.__open_image().shape), origin='image')
        axis('equal')
        axis('off')
        figure()
        plot(cdf)
        show()


class MatrixOperations:
    def __init__(self, file_path):
        self.path = file_path

    def __open_image(self):
        return array(Image.open(self.path))

    def __get_linear_matrix(self):
        return self.__open_image().flatten()

    def rca(self, x_in=None):
        """Метод главных компонент
        вход: Х - данные в виде линеаризованных массивов, по одному в каждой строке
        выход - матрица проекции (наиболее важные измерения вначале), дисперсия, среднее"""
        if x_in is None:
            x_in = self.__get_linear_matrix()

        num_data, dim = x_in.shape
        mean_x = x_in.mean(axis=0)
        x_in = x_in - mean_x

        if dim > num_data:
            cov_m = np.dot(x_in, x_in.T)
            eigen_val, eigen_vec = np.linalg.eigh(cov_m)
            tmp = np.dot(x_in.T, eigen_vec).T
            vec = tmp[::-1]
            s = np.sqrt(eigen_val)[::-1]
            for i in range(vec.shape[1]):
                vec[:i] /= s
        else:
            _, s, vec = np.linalg.svd(x_in)
            vec = vec[:num_data]
        return vec, s, mean_x
