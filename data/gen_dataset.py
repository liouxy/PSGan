from __future__ import division
import cv2
import gdal, ogr, os, osr
import numpy as np
import random

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,bandSize):
    if (bandSize==4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')
    
        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_Byte)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1,5):

            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i-1,:,:])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize==1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY =rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn,cols,rows,1,gdal.GDT_Byte)
        outRaster.SetGeoTransform((originX,pixelWidth,0,originY,0,pixelHeight))
        
        outband=outRaster.GetRasterBand(1)
        outband.WriteArray(array)



def main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,bandSize):
    #reversed_arr = array[::-1] # reverse array so the tif looks like the array
    array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,bandSize) # convert array to raster


if __name__ == "__main__":
    rasterOrigin = (-123.25745,45.43013)
    trainCount=0
    testCount=0
    trainDir="/data/psgan/data/quickbird/TIF/train"
    testDir="/data/psgan/data/quickbird/TIF/test"
    for num in range(1,9):
        mul = '/data/psgan/data/quickbird/dataset/%d_mul.tif' % num
        lr = '/data/psgan/data/quickbird/dataset/%d_lr.tif' % num
        lr_u = '/data/psgan/data/quickbird/dataset/%d_lr_u.tif' % num
        pan = '/data/psgan/data/quickbird/dataset/%d_pan.tif' % num
        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_pan = gdal.Open(pan)
        dt_lr_u = gdal.Open(lr_u)
        img_mul = dt_mul.ReadAsArray()
        img_lr = dt_lr.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize

        for count in range(8000):

            x=random.randint(0,XSize-32)
            y=random.randint(0,YSize-32)
            main('%s/%d_mul.tif' %(trainDir ,trainCount), rasterOrigin, 2.4, 2.4,
                 img_mul[:, y*4:(y + 32)*4, x*4:(x + 32)*4], 4)
            main('%s/%d_lr_u.tif' % (trainDir ,trainCount), rasterOrigin, 2.4, 2.4,
                 img_lr_u[:, y*4:(y + 32)*4, x*4:(x + 32)*4], 4)
            main('%s/%d_lr.tif' % (trainDir ,trainCount), rasterOrigin, 2.4, 2.4,
                 img_lr[:, y:(y + 32), x:(x + 32)], 4)
            main('%s/%d_pan.tif' % (trainDir ,trainCount), rasterOrigin, 2.4, 2.4, img_pan[y*4:(y + 32)*4, x*4:(x + 32)*4], 1)
            trainCount+=1
            print ("%d %d"%(num,trainCount))


    for num in range(9,10):
        mul = '/data/psgan/data/quickbird/dataset/%d_mul.tif' % num
        lr = '/data/psgan/data/quickbird/dataset/%d_lr.tif' % num
        lr_u = '/data/psgan/data/quickbird/dataset/%d_lr_u.tif' % num
        pan = '/data/psgan/data/quickbird/dataset/%d_pan.tif' % num
        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_pan = gdal.Open(pan)
        dt_lr_u = gdal.Open(lr_u)
        img_mul = dt_mul.ReadAsArray()
        img_lr = dt_lr.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize
        print XSize,YSize
        row=0
        col=0
        record = open('%s/record.txt'%testDir, "w")
        for y in range(0, YSize, 32):
            if y + 32 > YSize:
                continue
            col = 0
            for x in range(0, XSize, 32):
                if x + 32 > XSize:
                    continue
                main('%s/%d_mul.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_mul[:, y * 4:(y + 32) * 4, x * 4:(x + 32) * 4], 4)
                main('%s/%d_lr_u.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_lr_u[:, y * 4:(y + 32) * 4, x * 4:(x + 32) * 4], 4)
                main('%s/%d_lr.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_lr[:, y:(y + 32), x:(x + 32)], 4)
                main('%s/%d_pan.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_pan[y * 4:(y + 32) * 4, x * 4:(x + 32) * 4], 1)
                testCount += 1
                record.write("%d %d %d\n" % (row, col, testCount))
                col += 1
            row += 1
        record.close()