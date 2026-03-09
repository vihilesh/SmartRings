# SmartRings
This is the start of the smart rings project to use them to capture handwriting which are to be later analyzed by computers to determine if the person shows micrographia symptoms. The idea is that we capture handwriting over time and do periodic analysis to detect changes which can be compared against a model that knows if the changes are indicative of Parkinson's disease. 

# Pre-Requisites
Python 3.11 or greater
Visual Studio Code
Numpy
Pandas
Matplotlib
SciPy

Installation
Import project into VisualStudio Code. Add resampled heathly and parkinson's patient files into the data directory. Directory rhd is where resampled writing 
samples from healthy patients and directory rpd is where resampled writing data from Parkinson's patients are present. The code reads the data from the Data
directory and calculates various features of the handwriting like jerk, velocity, amplitude etc for healthy and parkinsons patients and compares them. It then
creates a model, separates out the data into 80% training set and 20% testing set to see how good the model is. 



