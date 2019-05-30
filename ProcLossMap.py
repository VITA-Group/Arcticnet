import pandas as pd
import numpy as np
import math
import gdal
from matplotlib import pyplot as plt
import time


def main():
    lossMapPath = "lossMap.hdf"
    lossMap = pd.read_hdf(lossMapPath).values
    width = int(math.sqrt(lossMap.shape[0]))
    lossMap = lossMap.reshape(width, width, -1)

    colors = [(0, 0, 255), (255, 255, 255), (255, 255, 0),
              (46, 139, 87), (60, 179, 113), (255, 0, 0), (0, 0, 0)]

    lossThreshold = 0.0001
    # startTime = time.time()
    # for cntRow in range(width):
    #     print("time used {:.3f}, time rest {:.3f}".format(time.time()-startTime,
    #                                                       (width-cntRow)*(time.time()-startTime)/(cntRow+1)))
    #     for cntCol in range(width):
    #         # print("process row {}, col {}".format(cntRow, cntCol))
    #         predictVec = lossMap[cntRow, cntCol, :]
    #         if np.max(predictVec) < lossThreshold:
    #             labelMap[cntRow, cntCol, :] = colors[6]
    #         else:
    #             labelMap[cntRow, cntCol, :] = colors[np.argmax(predictVec)]

    lossMap = np.concatenate([lossMap, np.ones([width, width, 1])*lossThreshold], axis=-1)
    index = np.argmax(lossMap, axis=-1).reshape(-1).tolist()
    colors = np.array(colors, dtype=np.uint8)
    labelMap = colors[index, :].reshape(width, width, -1)




    # # show image under this threshold
    # plt.imshow(labelMap)
    # plt.show()
    # pass

    # write the tiff image
    originX, originY = [561050.10031856, 6830191.21701998]
    pixelWidth, pixelHeight = [10, -10]

    projectionSrc = "PROJCS[\"WGS 84 / UTM zone 10N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",-123],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32610\"]]"
    dst_filename = 'classifyRes.tiff'

    x_pixels = width  # number of pixels in x
    y_pixels = width  # number of pixels in y

    driver = gdal.GetDriverByName('GTiff')

    geotif = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        3,
        gdal.GDT_Byte, )

    geotif.SetGeoTransform((
        originX,  # 0
        pixelWidth,  # 1
        0,  # 2
        originY,  # 3
        0,  # 4
        pixelHeight))

    geotif.SetProjection(projectionSrc)
    geotif.GetRasterBand(1).WriteArray(labelMap[:, :, 0].astype(np.uint8))
    geotif.GetRasterBand(2).WriteArray(labelMap[:, :, 1].astype(np.uint8))
    geotif.GetRasterBand(3).WriteArray(labelMap[:, :, 2].astype(np.uint8))
    geotif.FlushCache()  # Write to disk.


if __name__ == "__main__":
    main()