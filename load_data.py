import numpy as np
import os
import glob
import sys
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.transform import resize
from skimage.io import imread
from math import floor

"""
load_data load images from a directory where they are divided in subdirectories classes and the subdirectory name is the class name.

It has a mandatory argument which is the name of the directory, and another two optional arguments: 
	'--test': indicates the creation of a test set by stratified shuffle split method and size of 10% of the original set.
	'--extendedlabels': indicates that the labels have to be in the categorical form.

It saves the sets of images and labels in the format ".npy" at the current directory.
"""

def main():
	#looking to arguments of load_data.py call
	if len(sys.argv) < 2:
		print("usage: load_data.py directory_name [--test [--extendedlabels]]")
		sys.exit(1)	

	dirname = sys.argv[1]

	create_testset = False
	extendedlabels = False

	if len(sys.argv) > 2:
		if sys.argv[2] == '--test':	
			create_testset = True
		if sys.argv[2] == '--extendedlabels':
			extendedlabels = True
		if len(sys.argv) > 3:
			if sys.argv[3] == '--test':	
				create_testset = True
			if sys.argv[3] == '--extendedlabels':
				extendedlabels = True
	

	#listing classes
	classes = os.listdir(os.path.join(os.getcwd(), dirname))
	classes.sort()

	#listing images paths
	paths = glob.glob(dirname + "/*/*")
	paths.sort()
	n = len(paths)

	#putting all images and labels in arrays
	images = np.zeros(n, dtype = 'object')
	labels = np.zeros(n, dtype='int32')
	i = 0
	for path in paths:
		images[i] = imread(path, as_grey=True)
		class_name = os.path.basename(os.path.dirname(path))
		labels[i] = classes.index(class_name)
		i = i+1		

	#resizing the images
	
	target_shape = (95, 95, 1)
	"""
	new_imgs = np.zeros((len(images), target_shape[0], target_shape[1]))
	for k in range(len(images)):
		new_imgs[k] = resize(images[k], (target_shape[0],target_shape[1])).astype('float32')
	new_shape = (len(images), target_shape[0], target_shape[1], target_shape[2])
	images = np.reshape(new_imgs, new_shape)	
	"""
	
	new_imgs = np.zeros((len(images), target_shape[0], target_shape[1]))
	for k in range(len(images)):
		current = images[k]
		majorside = np.amax(current.shape)
		majorside_idx = np.argmax(current.shape)
		minorside = np.amin(current.shape)

		factor = target_shape[0]/majorside
		minorside_new = floor(minorside*factor)

		if majorside_idx == 0:
			current = resize(current, (target_shape[0],minorside_new))

		if majorside_idx == 1:
			current = resize(current, (minorside_new, target_shape[1]))

		for i in range(current.shape[0]):
			for j in range(current.shape[1]):
				new_imgs[k,i,j] = current[i,j]

	images = np.reshape(new_imgs, (len(images), target_shape[0], target_shape[1], target_shape[2]))
		
	print("Images and labels collected.")


	#spliting data in train and test and extend the labels 
	#FIXME: can be used the funcion "to_categorical" do keras?
	if create_testset == True:
		sss = StratifiedShuffleSplit(n_splits=1,test_size=0.1)
		train_index, test_index = next(sss.split(np.zeros(n), labels))
		X_train, X_test = images[train_index], images[test_index]
		y_train, y_test = labels[train_index], labels[test_index]
		print("Saving train images.")
		np.save("images_train.npy", X_train)
		print("Saving test images.")
		np.save("images_test.npy", X_test)

		if extendedlabels == False: 
			print("Saving train labels.")
			np.save("labels_train.npy", y_train)
			print("Saving test labels.")
			np.save("labels_test.npy", y_test)

		if extendedlabels == True:
			ex_labels_train = np.zeros((len(y_train),len(classes)))			
			for y1 in y_train:
				i = y_train[y1]
				ex_labels_train[y1,i] = 1 
			ex_labels_test = np.zeros((len(y_test),len(classes)))	
			for y2 in y_test:
				i = y_test[y2]
				ex_labels_test[y2,i] = 1
			print("Saving train extended labels.")
			np.save("extended_labels_train.npy", ex_labels_train)
			print("Saving test extended labels.")
			np.save("extended_labels_test.npy", ex_labels_test)	



	if create_testset == False and extendedlabels == True:
		print("Saving images.")
		np.save("images.npy", images)		
		ex_labels = np.zeros((n,len(classes)))
		for y in labels:
			i = labels[y]
			ex_labels[y,i] = 1 
		print("Saving extended labels.")
		np.save("extended_labels.npy", ex_labels)
	
	if create_testset == False and extendedlabels == False:	
		print("Saving images.")
		np.save("images.npy", images)
		print("Saving labels.")
		np.save("labels.npy", labels)
		

	print ('Done.')



if __name__ == "__main__":
	main()


	
