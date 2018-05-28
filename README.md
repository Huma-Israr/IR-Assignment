# IR-Assignment
Text Classification
Appling various Classifier on UCI Dataset (Sentence Classification Data Set: https://archive.ics.uci.edu/ml/datasets/Sentence+Classification ) Using irlib-0.1.1 .

# Applied Three Classifier   
    Rocchio 
    K-Nearest-Neighbors
    Naive Bayes 
# irlib-o.1.1
is Python Based Information Retrieval Libraray.It need to have Python installed on your computer and NLTK.
To install the package 
   1: Install Python  
   2: Install NLTK
   2: Install irlib==0.1.1 
# File Structure
Properly Cloned irlib Contain following main components:
   1. matrix.py
   2. metrics.py
   3. classifier.py
   4. preprocessor.py
   5. configuration.py
   6. superlist.py
 # How to use Code for the Dataset Classification 
 Download the dataset and Test.py is program used to train and test the classifier.download and execute this file. This program need library file to be imporeted also before executing the test program check the path for clssify.conf.it contain necessary configuration for the classfication task.(check the file pathin the test.py). changing classifer name in classify.conf file will execute diffrent classifier (e.g knn to Rocchio).you can also change other parameter in the classify.conf according to your requirment.
 For More Detail on irlib Libraray visit: https://github.com/gr33ndata/irlib
