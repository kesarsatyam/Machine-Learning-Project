# Machine-Learning-Project
Heart Disease Prediction

#Data Preparation

##Database Source:

The dataset is publicly available on the UCI Machine Learning Repository. The dataset provides
the patients’ information. This database contains 76 attributes, but all published experiments
refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that
has been used by ML researchers to this date.
The classification goal is to predict whether the patient has heart disease or not, and it also refers
to the presence of heart disease of the patient in given data. It is integer valued from 0 (no
presence) to 4.
Herat Disease Dataset Access Link: http://archive.ics.uci.edu/ml/datasets/Heart+Disease

###Attribute Information:

###Only 14 attributes used in this dataset-

• Id- patient identification number
• Trestbps- resting blood pressure (in mm Hg on admission to the hospital)
• Chol- serum cholesterol in mg/dl
• Fbs- (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
• Restecg- resting electrocardiographic results
• Thalach- maximum heart rate achieved
• Exang- exercise induced angina (1 = yes; 0 = no)
• Oldpeak- ST depression induced by exercise relative to rest
• Slope- the slope of the peak exercise ST segment
• Ca- number of major vessels (0-3) colored by flourosopy
• Thal- 3 = normal; 6 = fixed defect; 7 = reversable defect
• Pncaden- (sum of 5, 6, and 7)
• cp -chest pain type
• target- 1 or 0

##Python Libraries used in this project

• Panadas – Pandas is a high-level data manipulation tool developed by Wes McKinney. It is
built on the NumPy package and its key data structure is called the DataFrame. DataFrames
allow us to store and manipulate tabular data in rows of observations and columns of
variables.
• NumPy- NumPy is a Python package which stands for 'Numerical Python'. I have used
numpy for fast mathematical computation on arrays and matrices.
• Matplotlib – Matplotlib is a part, rather say a library of python. Using Matplotlib we can
plot graphs, histogram and bar plot and all those things.
matplotlib.pyplot is a collection of command style functions that make matplotlib work like
MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates
a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels,
etc.
• Seaborn- Seaborn is Statistical data visualization. It is a Python data visualization
library based on matplotlib. It provides a high-level interface for drawing attractive
and informative statistical graphics. I have used this library for data visualization.
• TensorFlow – TensorFlow is a machine learning framework being developed by
google to implement complex ML model without any difficulty like I have used in
Artificial Neural Network to train a model.
• Keras – Keras is an open-source neural-network library written in Python and high
level abstraction of TensorFlow. It reduces the syntaxial complexity of TensorFlow.
Instead of creating maps we directly implement the code by using TensorFlow as
backend. I have used this for because it runs seamlessly on CPU and GPU & also for
fast computing in neural network.
• Scikit-learn – Python’s scikit-learn library is easy to code for a machine learning classifier.
In this project, I have used many classifiers algorithms like (Decision Tree, K-Nearest
Neighbors, SVM etc.). After training our model with classification algorithms using scikitlearn, it can be used to predict the target labels given on dataset. In this project this is very important library for me.

###Below are some classifaction models which has already implemented in scikitlearn—

i. from sklearn.model_selection import train_test_split
- train_test_split used for devide training and testing data
ii. from sklearn.metrics import classification_report, accuracy_
score
- to get classification_report and accuracy score you will
get to know about these things in the code part
iii. from sklearn.tree import DecisionTreeClassifier
iv. from sklearn.ensemble import RandomForestClassifier
v. from sklearn.neighbors import KNeighborsClassifier
vi. from sklearn.svm import SVC
vii. from sklearn.naive_bayes import GaussianNB
viii. from sklearn.metrics import accuracy_score
- these are the classifier from scikit-learn library.


#CONCLUSION

 In this project, I have applied classification algorithms to identify if a patient presents a heart
disease or not. In this work, I have also discussed the various algorithms of Machine learning and
their implementation part that how we can use those algorithms to predict the Herat disease prediction
and we have talked about the symptoms of Heart Disease in given Dataset and we can apply these
algorithms like ANN, SVM, KNN, Logistic Regression, Decision Tree and Random Forest in any
dataset which has these type of similarity in that dataset and I have also discussed few measure factors
from which the person can take care of their health like distribution of age and chest pain.
 I have also shared my GitHub link and also attach one video to make you understand my project
well.
 Machine Learning has the ability from which we can predict the future so that we can save so
many lives to just predict the disease and research are going on in the medical field to make this field
more reliable and trustworthy.
