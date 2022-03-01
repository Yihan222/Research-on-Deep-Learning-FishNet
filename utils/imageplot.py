import  numpy as np
import  matplotlib.pyplot as plt

"""
parameters:
-images:contains a sequence of images
-labels:contains a sequence of tags corresponding to the images
"""

class_names2 = ['cassette_player','chain_saw','church','French_horn','garbage_truck','gas_pump','golf_ball','parachute','springer','tench']

def plot(images,labels,class_names):
    fig,axes = plt.subplots(3,5,figsize=(12,6))
    #n*m flat to 1*nm
    axes = axes.flatten()
    for img,label,ax in zip(images,labels,axes):
        ax.imshow(img)
        #np.argmax return the index of the maximum value in a numpy array, if several maximum values, return the first one
        ax.set_title(class_names[np.argmax(label)])
        ax.axis('off')
    #tight_layout: will automatically adjust the parameters of the sub-image to fill the entire image area
    plt.tight_layout()
    plt.show()

def plot2(images,labels):
    fig,axes = plt.subplots(3,5,figsize=(12,6))
    #n*m flat to 1*nm
    axes = axes.flatten()
    for img,label ,ax in zip(images,labels,axes):
        ax.imshow(img)
        #np.argmax
        ax.set_title(class_names2[np.argmax(label)])
        ax.axis('off')
    plt.tight_layout()
    plt.show()