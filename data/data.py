# Prepare Dataset
from os.path import join, isfile
import torch.utils.data as data
import numpy as np
import math
import cv2
from osgeo import gdal, gdalconst
from osgeo.gdalconst import *
import shapefile
from pyproj import Proj, transform
import random
import json
from torchvision import transforms


def findContourForTif(image, noDataVal, originalX, originalY, pixelWidth, pixelHeight):
    def findPoint(image, noDataVal, initRow, initCol, plusRow, plusCol, RowfirstFlag,
                  originalX, originalY, pixelWidth, pixelHeight):
        rows, cols = image.shape
        i, j = initRow, initCol
        while (1):
            if image[i][j] != noDataVal:
                break
            if not RowfirstFlag:
                j += plusCol
            else:
                i += plusRow
            if j == cols or i == rows:
                if RowfirstFlag:
                    j += plusCol
                    i = initRow
                else:
                    i += plusRow
                    j = initCol
        return np.array([[originalX + j * pixelWidth, originalY + i * pixelHeight]], dtype=np.float32)

    # find points
    rows, cols = image.shape
    point1 = findPoint(image, noDataVal, 0, 0, 1, 1, False, originalX, originalY, pixelWidth, pixelHeight)
    point2 = findPoint(image, noDataVal, 0, cols - 1, 1, -1, True, originalX, originalY, pixelWidth, pixelHeight)
    point3 = findPoint(image, noDataVal, rows - 1, 0, -1, 1, False, originalX, originalY, pixelWidth, pixelHeight)
    point4 = findPoint(image, noDataVal, 0, 0, 1, 1, True, originalX, originalY, pixelWidth, pixelHeight)

    contour = np.array([point1, point2, point3, point4], dtype=np.float32)

    # print("point contour test")
    # print(contour)
    # print("{}*{}".format(cols, rows))
    return contour


def grantXdatasetGene(root, cropSize):
    # read all rgb tif images and load the image's polygon for each image
    # print('read all planet images and depth image')
    # mypath = join(root, 'PlanetImagery')
    # imageRGBfilenames = [join(mypath, f) for f in listdir(mypath)
    #                      if isfile(join(mypath, f)) and f[-4:] == '.tif']
    # imageRGBRelativeFilenames = [join('PlanetImagery', f) for f in listdir(mypath)
    #                              if isfile(join(mypath, f)) and f[-4:] == '.tif']
    #
    # rgbContours = []
    # rgbXYWHCR = []
    #
    # for planetImgName in imageRGBfilenames:
    #     # read planet image
    #     driver = gdal.GetDriverByName('HFA')
    #     driver.Register()
    #
    #     ds = gdal.Open(planetImgName, GA_ReadOnly)
    #     if ds is None:
    #         raise RuntimeError('Could not open ' + planetImgName)
    #
    #     cols = ds.RasterXSize
    #     rows = ds.RasterYSize
    #     bands = ds.RasterCount
    #
    #     geotransform = ds.GetGeoTransform()
    #     originalX = geotransform[0]
    #     originalY = geotransform[3]
    #     pixelWidth = geotransform[1]
    #     pixelHeight = geotransform[5]
    #
    #     band = ds.GetRasterBand(1)
    #     noDataValue = band.GetNoDataValue()
    #     data = band.ReadAsArray(0, 0, cols, rows)
    #     rgbContours.append(findContourForTif(data, noDataValue, originalX, originalY, pixelWidth, pixelHeight))
    #     rgbXYWHCR.append([originalX, originalY, pixelWidth, pixelHeight, cols, rows])
    #
    # for i, cnt in enumerate(rgbContours):
    #     rgbContours[i] = cnt.tolist()
    #
    # planetInfo = {'imageRGBfilenames': imageRGBRelativeFilenames,
    #               'rgbContours': rgbContours,
    #               'rgbXYWHCR': rgbXYWHCR}
    # with open('planetInfo.json', 'w') as outfile:
    #     json.dump(planetInfo, outfile)
    # print("finish planet info saving")

    with open("planetInfo.json") as file:
        planetInfo = json.load(file)
    imageRGBfilenames = planetInfo['imageRGBfilenames']
    rgbContours = planetInfo['rgbContours']
    rgbXYWHCR = planetInfo['rgbXYWHCR']
    depthImgFilename = join(root, "wholeDEM.tiff")
    depthXYWHCR = [-3150000.0, -700000.0, 2.0, -2.0, 25000, 25000]
    for i, cnt in enumerate(rgbContours):
        rgbContours[i] = np.array(cnt, dtype=np.float32)

    # cross validation datasets
    crossValiDatasets = {}
    for i in range(3):
        crossValiDatasets['train' + str(i)] = grantXdataset(root, cropSize, 'train' + str(i),
                                                            imageRGBfilenames, rgbContours, rgbXYWHCR,
                                                            depthImgFilename, depthXYWHCR)
        crossValiDatasets['vali' + str(i)] = grantXdataset(root, cropSize, 'vali' + str(i),
                                                           imageRGBfilenames, rgbContours, rgbXYWHCR,
                                                           depthImgFilename, depthXYWHCR)

    testDataset = grantXdataset(root, cropSize, 'test', imageRGBfilenames, rgbContours, rgbXYWHCR,
                                depthImgFilename, depthXYWHCR)

    return crossValiDatasets, testDataset


def writeShape(shapes, categories, name):
    w = shapefile.Writer(join('testOut', name))

    categorieNames = ['Water', 'Bog', 'Channel_Fen',
                      'Forest_Dense', 'Forest_Sparse', 'Wetland']
    w.field('Category')
    for shape, category in zip(shapes, categories):
        w.poly([list(shape)])
        w.record(categorieNames[category])

    w.close()
    return


def shapeAug(shape):
    # random move
    shape = np.array(shape, dtype=np.float32)
    x_dir = (shape[1] - shape[0]) / 3
    y_dir = (shape[2] - shape[1]) / 3

    newShape = []
    dir = random.choice([-x_dir, 0, x_dir]) + random.choice([-y_dir, 0, y_dir])
    for pt in shape:
        newShape.append(pt + dir)
    shape = newShape

    newShape = []
    shape = np.array(shape)
    # random rotate
    a = random.choice([- math.pi / 4, math.pi / 4])
    center = np.mean(shape, axis=0, dtype=np.float32)
    for i, pt in enumerate(shape):
        ptNew = np.zeros(2, dtype=np.float32)
        ptNew[0] = ((pt[0] - center[0]) * math.cos(a)) - ((pt[1] - center[1]) * math.sin(a)) + center[0]
        ptNew[1] = ((pt[0] - center[0]) * math.sin(a)) + ((pt[1] - center[1]) * math.cos(a)) + center[1]
        newShape.append(ptNew)
    shape = newShape

    return list(shape)


class grantXdataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, cropSize, trainFlag, imageRGBfilenames, rgbContours, rgbXYWHCR, depthImgFilename,
                 depthXYWHCR):

        super(grantXdataset, self).__init__()
        """
        Args:
        root(string):  directory with all the input images.
        cropSize(integer): size of classification image (in meter)
        trainFlag(string): 'train', 'vali' or 'test'

        Outputs:
        self.imageRGBfilenames: for future reading
        self.rgbContours: The contours of all rgb pics
        self.rgbXYWHCR: The originalX, originalY, pixel width, pixel height, cols and rows of rgb pics
        self.trainShapes: 
        self.trainCategory:
        self.testShapes:
        self.testCategory:
        self.valiShapes:
        self.valiCategory:
        """
        self.root = root
        self.cropSize = cropSize

        # read all rgb tif images and load the image's polygon for each image
        self.imageRGBfilenames = imageRGBfilenames
        self.rgbContours = rgbContours
        self.rgbXYWHCR = rgbXYWHCR
        self.depthImgFilename = depthImgFilename
        self.depthXYWHCR = depthXYWHCR

        self.trainFlag = trainFlag

        # transform function
        # self.toTensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                       std=[0.229, 0.224, 0.225])

        # augmentation
        self.color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)

        # read all shapes and divide them
        self.categories = {'Water': 0, 'Bog': 1, 'Channel_Fen': 2,
                           'Forest_Dense': 3, 'Forest_Sparse': 4, 'Wetland': 5,
                           'Dense_Forest': 3, 'Sparse_Forest': 4}

        # training flag
        if trainFlag == 'train':
            self.augmentationFlag = True
        else:
            self.augmentationFlag = False

        # sf = shapefile.Reader(join(root, "Class_Samples/Class_Samples.shp"))
        sf = shapefile.Reader(join(self.root, "Class_Samples/{}Shapes.shp".format(trainFlag)))

        shapes = []
        shapeCategories = []
        for rec in sf.shapeRecords():
            pts = []
            for point in rec.shape.points:
                pts.append(point)
            shapes.append(pts)
            shapeCategories.append(self.categories[rec.record[-1]])

        # training shape balance
        if 'train' in trainFlag:
            trainingShapeEachCategory = [[], [], [], [], [], []]
            categoryState = np.zeros(6, dtype=np.int32)
            for shape, category in zip(shapes, shapeCategories):
                trainingShapeEachCategory[category].append(shape)
                categoryState[category] += 1
            upbound = np.max(categoryState)
            for cntCategory, categoryNum in enumerate(categoryState):
                for q in range(upbound - categoryNum):
                    trainingShapeEachCategory[cntCategory].append(
                        shapeAug(random.choice(trainingShapeEachCategory[cntCategory])))

            shapes = []
            shapeCategories = []
            for cntCategory, shapeVec in enumerate(trainingShapeEachCategory):
                for shape in shapeVec:
                    shapes.append(shape)
                    shapeCategories.append(cntCategory)

        # writeShape(self.shapes, self.ShapeCategories, 'fortesting')

        # generate RGB shapes for pics
        self.originShapes = shapes
        self.shapes = []
        if cropSize is None:
            pass
        else:
            for i, shape in enumerate(self.originShapes):
                resizeRate = cropSize / 30
                centerPt = (np.array(shape[0]) + np.array(shape[1]) + np.array(shape[2]) + np.array(shape[3])) / 4
                newShape = []
                for pt in shape:
                    newShape.append(tuple(resizeRate * (pt - centerPt) + centerPt))
                self.shapes.append(newShape)

        # generate which shape is in which image
        shapeInImg = []
        for i, shape in enumerate(self.shapes):
            shapeInImg.append([])
            for j, rgbCnt in enumerate(self.rgbContours):
                for cnt, pt in enumerate(shape):
                    if cv2.pointPolygonTest(rgbCnt, pt, False) < 0:
                        break
                    if cnt == 3:
                        shapeInImg[i].append(j)

        # if shape is not in any image
        newOriginShape, newShape, newCategory, newShapeInImg = [], [], [], []
        for i, shapeImgInfo in enumerate(shapeInImg):
            if not shapeImgInfo:
                continue
            newOriginShape.append(self.originShapes[i])
            newShape.append(self.shapes[i])
            newCategory.append(shapeCategories[i])
            newShapeInImg.append(shapeInImg[i])
        self.originShapes = newOriginShape
        self.shapes = newShape
        self.ShapeCategories = newCategory
        self.shapeInImg = newShapeInImg

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        shape = self.shapes[index]
        category = self.ShapeCategories[index]
        shapeInImg = self.shapeInImg[index]
        assert len(shapeInImg) > 0

        if self.augmentationFlag:
            imageIdx = random.choice(shapeInImg)
        else:
            imageIdx = shapeInImg[0]
        image1, image2 = self.imgInPolygon(imageIdx,
                                           shape, (self.cropSize,) * 2, augmentationFlag=self.augmentationFlag)

        return {'image1': image1,
                'image2': image2,
                'label': category,
                'originShape': np.array(self.originShapes[index], dtype=np.float32)}

    def shapeInWhich(self, shape):
        """
        :param shape: list of points of a shape
        :return: idxs: list of the index of pics which  contains the shape
        """
        idxs = []
        for idx, cnt in enumerate(self.rgbContours):
            inFlag = True
            for point in shape:
                dist = cv2.pointPolygonTest(cnt, point, False)
                if dist < 0:
                    inFlag = False
                    break
            if inFlag:
                idxs.append(idx)
        return idxs

    def imgInPolygon(self, idx, polygon, size, augmentationFlag=False):
        """
        :param idx: integer: the idx of target pics
        :param polygon: list of points: the polygon of the pic
        :param size: expect (cols, rows)
        :param readBands: list of nums: the bands to read
        :return: the image in the polygon
        """
        # part1: read rgb image
        # the range of pixel to read
        originalX, originalY, pixelW, pixelH, cols, rows = self.rgbXYWHCR[idx]

        # coordinate convert
        polygonNew = []
        for point in polygon:
            polygonNew.append(((point[0] - originalX) / pixelW, (point[1] - originalY) / pixelH))

        xMin = int(min([point[0] for point in polygonNew]))
        yMin = int(min([point[1] for point in polygonNew]))
        xMax = int(math.ceil(max([point[0] for point in polygonNew])))
        yMax = int(math.ceil(max([point[1] for point in polygonNew])))

        # image read and polygon in New image
        driver = gdal.GetDriverByName('HFA')
        driver.Register()

        ds = gdal.Open(join(self.root, self.imageRGBfilenames[idx]), GA_ReadOnly)
        if ds is None:
            raise RuntimeError('Could not open ' + join(self.root, self.imageRGBfilenames[idx]))

        band = ds.GetRasterBand(1)
        dataRGBIR = band.ReadAsArray(xMin, yMin, xMax - xMin, yMax - yMin)
        dataRGBIR = [dataRGBIR]
        for bandNum in range(2, 5):
            band = ds.GetRasterBand(bandNum)
            dataRGBIR.append(band.ReadAsArray(xMin, yMin, xMax - xMin, yMax - yMin))

        np.concatenate(dataRGBIR, axis=0)
        dataRGBIR = np.stack(dataRGBIR)
        dataRGBIR = np.transpose(dataRGBIR, (1, 2, 0))

        # new image coordinate
        for i, point in enumerate(polygonNew):
            polygonNew[i] = (point[0] - xMin, point[1] - yMin)

        pts1 = np.float32(polygonNew[:3])
        pts2 = np.float32([(0, 0), (size[0], 0), size])
        M = cv2.getAffineTransform(pts1, pts2)

        dataRGBIR = cv2.warpAffine(dataRGBIR, M, size, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        # warnings.filterwarnings("error")
        NDVI = (dataRGBIR[:, :, 3] - dataRGBIR[:, :, 0]) / (dataRGBIR[:, :, 3] + dataRGBIR[:, :, 0])

        # part2: read depth image
        # hard coding
        if not ('remote_test' in self.trainFlag):
            inputProjection = Proj(init="epsg:32610")
            outputProjection = Proj(init="epsg:3413")
        else:
            inputProjection = Proj(init="epsg:32614")
            outputProjection = Proj(init="epsg:4617")

        # the range of pixel to read
        originalX, originalY, pixelW, pixelH, cols, rows = self.depthXYWHCR

        # coordinate convert
        polygonNew = []
        for point in polygon:
            point = transform(inputProjection, outputProjection, point[0], point[1])
            polygonNew.append(((point[0] - originalX) / pixelW, (point[1] - originalY) / pixelH))

        xMin = int(min([point[0] for point in polygonNew]))
        yMin = int(min([point[1] for point in polygonNew]))
        xMax = int(math.ceil(max([point[0] for point in polygonNew])))
        yMax = int(math.ceil(max([point[1] for point in polygonNew])))

        ds = gdal.Open(self.depthImgFilename, GA_ReadOnly)
        if ds is None:
            raise RuntimeError('Could not open ' + self.depthImgFilename)

        band = ds.GetRasterBand(1)
        dataDepth = band.ReadAsArray(xMin, yMin, xMax - xMin, yMax - yMin)

        # new image coordinate
        for i, point in enumerate(polygonNew):
            polygonNew[i] = (point[0] - xMin, point[1] - yMin)

        pts1 = np.float32(polygonNew[:3])
        pts2 = np.float32([(0, 0), (size[0], 0), size])
        M = cv2.getAffineTransform(pts1, pts2)

        dataDepth = cv2.warpAffine(dataDepth, M, size)

        image1 = dataRGBIR[:, :, :3].astype(np.float32)
        image2 = np.stack([dataRGBIR[:, :, 3].astype(np.float32) / 10000, NDVI,
                           dataDepth.astype(np.float32) / 1000], axis=2).astype(np.float32)

        # augmentation
        if augmentationFlag:
            # transform to PIL image
            # dst = Image.fromarray(dst.astype(np.uint32))
            # if np.random.random() > 0.5:
            #     dst = self.color_jitter(dst)

            if np.random.random() > 0.5:
                image1 = np.fliplr(image1)
                image2 = np.fliplr(image2)

            if np.random.random() > 0.5:
                image1 = np.flipud(image1)
                image2 = np.flipud(image2)

        # transform
        image1 = np.transpose(image1, (2, 0, 1))
        image2 = np.transpose(image2, (2, 0, 1))
        return image1, image2

