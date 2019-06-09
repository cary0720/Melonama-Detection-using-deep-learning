
from unet import *
from data import *
from fcn import *
from fcn_vgg import *
import skimage.io as io
import skimage.transform as trans
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with tf.device("/cpu:0"):
    # Setup operations
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(2, 'datasets/', 'image', 'mask', data_gen_args, save_to_dir=None)

    model_name = 'fcn_vgg'  # 'unet' 'fcn' 'fcn_vgg'
    #model = unet()
    #model = fcn()
    model = fcn_vgg()
    model_checkpoint = ModelCheckpoint(model_name+'_best.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=10, epochs=300, callbacks=[model_checkpoint])
    #

    test_path = "datasets/test/"
    image_prefix = 'ISIC_'
    image_name_arr = glob.glob(os.path.join(test_path, "%s*.jpg" % image_prefix))

    for i in range(len(image_name_arr)):
        test = io.imread(image_name_arr[i])/255
        test_resize = trans.resize(test, (256, 256))
        test_resize = np.reshape(test_resize, (1,) + test_resize.shape)
        result_resize = model.predict(test_resize, 1, verbose=1)
        result_resize = np.squeeze(result_resize)
        result = trans.resize(result_resize, test.shape[0:2])
        result_name = image_name_arr[i].replace('test', 'result_test_'+model_name).replace('.jpg', '_segmentation.png')
        print(result.shape)
        result = result > 0.5
        result = result.astype(np.float)
        io.imsave(result_name, result)
    # print('ISIC')
    # results = model.predict_generator(testGene,1001,verbose=1)
    # print(results.shape)
    # print(type(results))
    # saveResult("datasets/result_rest/",results, "datasets/test/")
    # data_gen_args = dict(rotation_range=0,
    # width_shift_range=0,
    # height_shift_range=0,
    # shear_range=0,
    # zoom_range=0,
    # horizontal_flip=False,
    # fill_mode='nearest')

with tf.Session() as sess:
    # Run your code
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print
    sess.run
