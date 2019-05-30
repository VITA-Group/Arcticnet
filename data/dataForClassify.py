from .data import grantXdataset
import shapefile
import os.path as osp
from pyproj import Proj, transform
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json
import random


class grantXdatasetForClassify(grantXdataset):
    """input and label image dataset"""

    def __init__(self, root, cropSize, imageRGBfilenames, rgbContours, rgbXYWHCR, depthImgFilename, depthXYWHCR):
        super(grantXdatasetForClassify, self).__init__(root, cropSize, 'train0',
                                                       imageRGBfilenames, rgbContours, rgbXYWHCR,
                                                       depthImgFilename, depthXYWHCR)

        # change filenames
        # for i, filename in enumerate(self.imageRGBfilenames):
        #     self.imageRGBfilenames[i] = '../' + filename

        # extract target the shapes in Extent
        sf = shapefile.Reader(osp.join(root, "Extent/Extent.shp"))
        # every image's shape to extract
        shape = sf.shape(0).points
        input_projection = Proj(init="epsg:4326")
        output_projection = Proj(init="epsg:32610")

        newShape = []
        for point in shape:
            newShape.append(transform(input_projection, output_projection, point[0], point[1]))
        shape = newShape

        originX = min([x[0] for x in shape])
        originY = max([x[1] for x in shape])
        pixelWidth = 10
        pixelHeight = -10

        cols = int(round((max([x[0] for x in shape]) - originX) / pixelWidth))
        rows = int(round((min([x[1] for x in shape]) - originY) / pixelHeight))

        self.extentXYWHCR = [originX, originY, pixelWidth, pixelHeight, cols, rows]

        # shape contour in image and draw it
        self.XY = np.array([originX, originY])
        self.pWH = np.array([pixelWidth, pixelHeight])
        self.RC = [rows, cols]
        shapeCnt = [np.array([(np.array(pt) - self.XY) / self.pWH for pt in shape], dtype=np.int32)]

        drawing = np.zeros([rows, cols], np.uint8)
        cv2.drawContours(drawing, shapeCnt, 0, 1, cv2.FILLED)

        print("generating all centers in (cols, rows)")

        self.center = np.transpose(np.nonzero(drawing), (1, 0))
        print("generating finished")

        # plt.imshow(drawing)
        # plt.show()
        return

    def __getitem__(self, index):
        """
        :param index:
        :return:
        image: np.array:
        coordinate: np.array:
        imageFlag: bool: indicate if this image should be used
        """
        # gene shape by center
        testShape = []
        centerImg = np.array([self.center[index, 1], self.center[index, 0]])
        center = self.center[index, :][::-1] * self.pWH + self.XY
        testShape.append((center[0] - self.cropSize / 2, center[1] + self.cropSize / 2))
        testShape.append((center[0] + self.cropSize / 2, center[1] + self.cropSize / 2))
        testShape.append((center[0] + self.cropSize / 2, center[1] - self.cropSize / 2))
        testShape.append((center[0] - self.cropSize / 2, center[1] - self.cropSize / 2))

        # judge in which rgbcontour
        shapeInImg = []
        for j, rgbCnt in enumerate(self.rgbContours):
            for cnt, pt in enumerate(testShape):
                if cv2.pointPolygonTest(rgbCnt, pt, False) < 0:
                    break
                if cnt == 3:
                    shapeInImg.append(j)

        # extract it
        if shapeInImg:
            image1, image2 = self.imgInPolygon(random.choice(shapeInImg), testShape, (self.cropSize, )*2)

            # return the image with coordinate
            # if not in any rgbcontour, return with -1s
            return {'image1': image1,
                    'image2': image2,
                    'coordinate': centerImg,
                    'imageFlag': True}
        else:
            return {'image1': np.zeros(3, self.cropSize, self.cropSize),
                    'image2': np.zeros(3, self.cropSize, self.cropSize),
                    'coordinate': centerImg,
                    'imageFlag': False}

    def __len__(self):
        return self.center.shape[0]