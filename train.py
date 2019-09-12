from model import *
from data import *



data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(4,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane_{epoch}_{loss:.3f}_{acc:.3f}.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=500,epochs=5,callbacks=[model_checkpoint])


# ### Train with npy file

# In[3]:


#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])


# ### test your model and save predicted results

# In[4]:


# testGene = testGenerator("data/membrane/test")
# model = unet()
# model.load_weights("unet_membrane.hdf5")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)


# In[ ]:



