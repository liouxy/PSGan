import os,sys
import tensorflow as tf
import gdal
data_dir='/home/lxy/data/pansharpening/dataset/'
testlist=['%s/test_mmm/%d'%(data_dir,number) for number in range(384)]
trainfiles=['%s/train/%d'%(data_dir,number) for number in range(64000)]
output_dir="/home/lxy/data/psgan/"
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(inputfiles, name):
    num_examples=len(inputfiles)
    filename=os.path.join(output_dir,name+'.tfrecords')
    print ('Writing', filename)
    writer=tf.python_io.TFRecordWriter(filename)
    for (file,i) in zip(inputfiles, range(num_examples)):
        print file,i
        img_name = '%s_%d' % (name, i)
        mul_filename = '%s_mul.tif' % file
        blur_filename = '%s_blur.tif' % file
        pan_filename = '%s_pan.tif' % file

        im_mul_raw = gdal.Open(mul_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        im_blur_raw = gdal.Open(blur_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        im_pan_raw = gdal.Open(pan_filename).ReadAsArray().reshape([128, 128, 1]).tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'im_name': _bytes_feature(img_name),
            'im_mul_raw': _bytes_feature(im_mul_raw),
            'im_blur_raw':_bytes_feature(im_blur_raw),
            'im_pan_raw':_bytes_feature(im_pan_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

convert_to(trainfiles,'train')
convert_to(testlist,'test')



