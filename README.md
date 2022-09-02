# Project Overview

The purpose of this project is to classify texts into two categories (positive, negative) using ID3, Naive Bayes and Random Forest algorithms.

# Execution Instracions 

A) For the correct operation of the program, installation of the texttable and matplotlib libraries is required.

B) To run the program, the following steps must be followed (all .py files must be in the same folder):
1. Go to the folder containing the .py files through the command prompt.
2. Run the program with the python command Main.py.
3. Enter the path of the file containing the vocabulary, e.g. imdb.vocab.
4. Enter the path of the file that contains the elements of the comments that will be used for training, e.g. train\labeledBow.feat.
5. Enter the path of the file that contains the elements of the comments that will be used for the testing e.g. test\labeledBow.feat.
6. Select algorithm to run.

C) Display results.
1. Display correctness curve and number of training examples used in each iteration of the experiment.
2. Display precision and recall curve.
3. Display F1 curve and number of training examples used in each iteration of the experiment.
4. The results of the algorithm for the test data are in the Results.txt file located in the same location as the .py files.
5. The accuracy and count table of the training examples in the accuracy.txt file located in the same location as the .py files.
6. The precision and recall table in the precision-recall.txt file located in the same location as the .py files.
7. The F1 array of training examples in the F1.txt file located in the same location as the .py files.
