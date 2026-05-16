# Project

Up until now, we have given you fairly detailed instructions for how to design data analyses to answer specific questions about data -- in particular, how to set up a particular analysis and what steps to take to run it. In this project, you will put that knowledge to use!

Put yourself in the shoes of a data scientist being given a data set and asked to draw conclusions from it. Your job will be to understand what the data is showing you, design the analyses you need, justify those choices, draw conclusions from running the analyses, and explain why they do (or do not) make sense.

We are deliberately not giving you detailed directions on how to solve these problems, but feel free to come to office hours to brainstorm.

## Objectives

Use dataset (path) #3, which are images of digits. See below for the analysis questions we want you to answer.

   
## Path 3: Data Security in Model Training

The first task is to understand the digits dataset from sklearn. This dataset information is found at sklearn.datasets.load_digits — scikit-learn 1.1.3 documentation. You may find this link helpful: (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). There are 10 classes (0-9) and each datapoint is a 8 x 8 image of a digit. 
The first task is to understand this data by importing the dataset and printing out some of the samples. You will need to do the following in Path 3:

1. Complete the code for dataset_searcher and print_number in MinProjectPath3.py
2. print out and plot the numbers of the class [2, 0, 8, 7, 5]

Incomplete sample code is given as a guideline to create and fit different models to the data. Please refer to the sklearn documentation of (1) Gaussian Naive Bayes (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), (2) K-NeighborsClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), and (3) MLPClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), to properly do the following:

3. Get the predicted values of the model with the **Test data** with the Gaussian Naive Bayes model
4. Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?) by finishing the definintion of OverallAccuracy and find the value for the overall accuracy of the Gaussian model
5. Get the predicted values and show the results of the model with the numbers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (use instead of X_test) with the print_number function
6. Repeat steps 3 and 5 with the KNeighbors Classifier and the MLPClassifier
7. Discuss your results. Is there a difference between the performance of the three models?

Now some of the training data is "poisoned." This is shown in the later part of the MinProjectPath3.py code.

8. Describe what is happening to the training data
9. Repeat steps 3-6, but this time, use the poisoned training data to fit the model. Note that the evaluation still should be done for the **test data**. 
10. Discuss how the three model performances have changed after poisoning the training data
11. Discuss what model showed strongest robustness against the poisoning. 

Now try to "denoise" the "poisoned" training data using denoising functions provided from the sklearn library. 
We suggest you to use the KernelPCA method for denoising the corrupted training data. The following link can be helpful for applying KernelPCA method: (https://scikit-learn.org/stable/auto_examples/applications/plot_digits_denoising.html)

12. Describe what is happening to remove the noise.
13. Discuss how Poison Data 1 differs from the denoised data.
14. After denoising the training dataset, repeat steps 3-6 with the denoised training data. Note that the denoised training data is used only for fitting the model, and the evaluation should be done for the **test data**. 
15. Discuss how the three model performances have changed after applying the denoising steps. Have the performances improved? Or was there no significant difference? 
m your analysis, but you must explain why that might be**.

* All Python `.py` code files you wrote to complete the analysis steps.
