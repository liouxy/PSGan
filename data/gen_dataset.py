from __future__ import division
import gdal, ogr, os, osr
import numpy as np
import cv2
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
    count=0
    for num in range(9,10):
        mul='/data/psgan/data/quickbird/dataset/%d_mul.tif'%num
        lr = '/data/psgan/data/quickbird/dataset/%d_lr.tif' % num
        lr_u = '/data/psgan/data/quickbird/dataset/%d_lr_u.tif' % num
        pan='/data/psgan/data/quickbird/dataset/%d_pan.tif'%num
        dt_mul=gdal.Open(mul)
        dt_lr=gdal.Open(blur)
        dt_pan=gdal.Open(pan)
        img_mul=dt_mul.ReadAsArray()
        img_blur=dt_blur.ReadAsArray()
        img_pan=dt_pan.ReadAsArray()
        
        for i in range(0,img_mul.shape[1]-128,128):
            for j in range(0,img_mul.shape[2]-128,128):
                main('dataset/test/%d_o.tif'%count,rasterOrigin,2.4,2.4,img_mul[:,i:i+128,j:j+128],4)
                main('dataset/test/%d_b.tif'%count,rasterOrigin,2.4,2.4,img_blur[:,i:i+128,j:j+128],4)
                main('dataset/test/%d_g.tif'%count,rasterOrigin,2.4,2.4,img_pan[i:i+128,j:j+128],1)
                count+=1
        print ('done%d'%num)
   # cv2.imwrite('blur_1.tif',img_blur.transpose(1,2,0))
   # tmp=cv2.imread('blur_1.tif',-1) 

    #print tmp[234][456]
    #print img_blur.transpose(1,2,0)[234][456]
