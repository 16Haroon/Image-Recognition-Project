#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)

  images_nparray = np.array(images)
  labels_nparray = np.array(labels)
  # Extract images and labels based on indices in number_list
  images_nparray = images_nparray[number_list]
  labels_nparray = labels_nparray[number_list]

  return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title.
  num_images = len(images)

  # Create a grid for plotting images
  num_cols = 5
  num_rows = (num_images + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))

  for i, ax in enumerate(axes.flat):
    if i < num_images:
      # Plot the image
      ax.imshow(images[i], cmap='gray')
      ax.axis('off')
      # Set the title with the corresponding label
      ax.set_title(f'Label: {labels[i]}', fontsize=10)
    else:
      ax.axis('off')

  plt.tight_layout()
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test_reshaped)


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  total_count = len(actual_values)
  correct_count = 0

  for pred_value, actual_value in zip(results, actual_values):
    if pred_value == actual_value:
      correct_count += 1

  #return Accuracy
  return correct_count / total_count

print("The overall results:")

# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The results of the Gaussian model is " + str(Model1_Overall_Accuracy))

#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
allnumbers_images_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
pred_labels = model_1.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images , pred_labels )


#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model_2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model_2_results, y_test)
print("The results of the KNN model is " + str(Model2_Overall_Accuracy))

pred_labels = model_2.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images , pred_labels )

#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model_3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model_3_results, y_test)
print("The results of the MLP Classifier model is " + str(Model3_Overall_Accuracy))

pred_labels = model_3.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images , pred_labels )

#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
print("The results of data with poison:")

X_train_poison_reshaped = X_train_poison.reshape(X_train.shape[0], -1)

#GaussianNB
model_1.fit(X_train_poison_reshaped, y_train)
model1_results = model_1.predict(X_test_reshaped)
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The results of the Gaussian model is " + str(Model1_Overall_Accuracy))

pred_labels = model_1.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images , pred_labels )

#KNN
model_2.fit(X_train_poison_reshaped, y_train)
model_2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model_2_results, y_test)
print("The results of the KNN model is " + str(Model2_Overall_Accuracy))

pred_labels = model_2.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images , pred_labels )

#MLP Classifier
model_3.fit(X_train_poison_reshaped, y_train)
model_3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model_3_results, y_test)
print("The results of the MLP Classifier model is " + str(Model3_Overall_Accuracy))

pred_labels = model_3.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images , pred_labels )

#Part 12-13
# Denoise the poisoned training data, X_train_poison.
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data.
kpca = KernelPCA(n_components=X_train_poison_reshaped.shape[1], kernel='rbf', gamma=1e-5)

# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

X_train_denoised = kpca.fit_transform(X_train_poison_reshaped)

X_test_transformed = kpca.transform(X_test_reshaped)
allnumbers_images_transfornmed = kpca.transform(allnumbers_images_reshaped)

#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
print("The results of data after denoising:")

#GaussianNB
model_1.fit(X_train_denoised, y_train)
model1_results = model_1.predict(X_test_transformed)
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The results of the Gaussian model is " + str(Model1_Overall_Accuracy))

pred_labels = model_1.predict(allnumbers_images_transfornmed)
print_numbers(allnumbers_images , pred_labels )

#KNN
model_2.fit(X_train_denoised, y_train)
model_2_results = model_2.predict(X_test_transformed)
Model2_Overall_Accuracy = OverallAccuracy(model_2_results, y_test)
print("The results of the KNN model is " + str(Model2_Overall_Accuracy))

pred_labels = model_2.predict(allnumbers_images_transfornmed)
print_numbers(allnumbers_images , pred_labels )

#MLP CLassifier
model_3.fit(X_train_denoised, y_train)
model_3_results = model_3.predict(X_test_transformed)
Model3_Overall_Accuracy = OverallAccuracy(model_3_results, y_test)
print("The results of the MLP Classifier model is " + str(Model3_Overall_Accuracy))

pred_labels = model_3.predict(allnumbers_images_transfornmed)
print_numbers(allnumbers_images , pred_labels )