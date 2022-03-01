from tensorflow.keras.preprocessing.image import ImageDataGenerator

#use tf.keras.preprocessing.image.ImageDataGenerator to generate pictures
#-rescale:considering the image pixels are all integers from 0 to 255，rescale can multiply them by the same value, 
#         usually 1./255, and pixels would turn to the number between 0-1

#read images from ImageDataGenerator.flow_from_directory 
#-directory:path  
#-target_size:picture width and height zoom specified size default（256，256） 
#-batch_size:number of pictures read at a time, 32 default 
#-class_mode:types of classes, default categorical
"""
class_mode:types of classes
              'sparse'：classes['cassette_player','chain_saw','church'] ——> [0, 1, 2]
              'categorical'：classes['cassette_player','chain_saw','church'] ——> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
              'input'：classes['cassette_player','chain_saw','church'] remain the same
              if None：dont return
"""

def train_val_generator(data_dir,target_size, batch_size, class_mode=None,subset='training'):
    train_val_datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.2)
    return  train_val_datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset=subset
    )

def test_generator(data_dir,target_size, batch_size, class_mode=None):
    test_datagen = ImageDataGenerator(rescale=1./255.)
    return  test_datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )

def pred_generator(data_dir,target_size, batch_size, class_mode=None):
    pred_datagen = ImageDataGenerator(rescale=1./255.)
    return pred_datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )
