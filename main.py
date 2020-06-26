import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from model import *
from data import *
from solve import *

import datetime


solve_cudnn_error()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(4 ,'data/train','afterMask_blue','mask',data_gen_args,save_to_dir = None)

print("Construct a model.")
model = unet()

print("Generate training set.")
model_checkpoint = ModelCheckpoint('unet_DRIVE.hdf5', monitor='loss',verbose=1, save_best_only=True)

print("Fitting")
model.fit(myGene,steps_per_epoch=150,epochs=3,callbacks=[model_checkpoint])

print("Predicting.")
testGene = testGenerator("data/test/image", 20)
results = model.predict(testGene,20,verbose=1)

saveResult("data/result", results)