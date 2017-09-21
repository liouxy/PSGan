from __future__ import division
import gdal, ogr, os, osr
import numpy as np
import cv2
def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):

    cols = array.shape[2]
    rows = array.shape[1]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    
    outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_Byte,[ 'TILED=YES', 'COMPRESS=PACKBITS' ])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    for i in range(1,5):

        outband = outRaster.GetRasterBand(i)
        outband.WriteArray(array[i-1,:,:])
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    #reversed_arr = array[::-1] # reverse array so the tif looks like the array
    array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array) # convert array to raster

def linear(img):
    img_new=np.zeros(img.shape)
    sum_=img.shape[1]*img.shape[2]
    print sum_
    for i in range(0,img.shape[0]):
        num=np.zeros(5000)
        prob=np.zeros(5000)
        for j in range(0,img.shape[1]):
            for k in range(0,img.shape[2]):
                num[img[i,j,k]]=num[img[i,j,k]]+1
        for tmp in range(0,5000):
            prob[tmp]=num[tmp]/sum_
        Min=0
        Max=0
        min_prob=0.0 
        max_prob=0.0
        while(Min<5000 and min_prob<0.2):
            min_prob+=prob[Min]
            Min+=1
        print min_prob,Min
        while (True):
            max_prob+=prob[Max]
            Max+=1
            if(Max>=5000 or max_prob>=0.98):
                break
        print max_prob,Max
        for m in range(0,img.shape[1]):
            for n in range(0,img.shape[2]):
                if (img[i,m,n]>Max):
                    img_new[i,m,n]=255
                elif(img[i,m,n]<Min):
                    img_new[i,m,n]=0
                else:
                    img_new[i,m,n]=(img[i,m,n]-Min)/(Max-Min)*255
    return img_new

if __name__ == "__main__":
    rasterOrigin = (-123.25745,45.43013)
    for i in range(1,10):
        newMul = '/data/psgan/data/quickbird/dataset/%d_mul.tif' % i
        newLR = '/data/psgan/data/quickbird/dataset/%d_lr.tif' % i
        newLR_U = '/data/psgan/data/quickbird/dataset/%d_lr_u.tif' % i
        dataset = gdal.Open('/data/psgan/data/quickbird/%d_MUL.TIF' % i)

        img=dataset.ReadAsArray()
        img=linear(img)
        img_blur=img.transpose(1,2,0)
        blur_1=cv2.pyrDown(img_blur,dstsize=(int(img_blur.shape[1]/2),int(img_blur.shape[0]/2)))
        blur_2=cv2.pyrDown(blur_1,dstsize=(int(blur_1.shape[1]/2),int(blur_1.shape[0]/2)))
        blur_3=cv2.pyrUp(blur_2,dstsize=(blur_1.shape[1],blur_1.shape[0]))
        img_blur=cv2.pyrUp(blur_3,dstsize=(img_blur.shape[1],img_blur.shape[0]))
        img_blur=img_blur.transpose(2,0,1)
        blur_2 = blur_2.transpose(2,0,1)
        print img_blur.shape,blur_2
        main(newMul,rasterOrigin,2.4,2.4,img)
        main(newLR_U,rasterOrigin,2.4,2.4,img_blur)
        main(newLR, rasterOrigin,2.4,2.4,blur_2)
        print ('done%d'%i)
   # cv2.imwrite('blur_1.tif',img_blur.transpose(1,2,0))
   # tmp=cv2.imread('blur_1.tif',-1) 

    #print tmp[234][456]
    #print img_blur.transpose(1,2,0)[234][456]
