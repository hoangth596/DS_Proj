# Disaster Response Pipeline Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Components](#components)
 * [Instructions](#instructions)
 * [Licensing, Authors, Acknowledgements, etc.](#licensing-authors-acknowledgements-etc)
 
### Project Motivation
In this project, I applied my data engineering skills to analyze the disaster data in Figure Eight and create a model for an API to classify disaster news. We built a machine learning pipeline to classify real-world messages sent during a disaster so that they can be routed to the appropriate disaster relief. This project contains a web app that allows paramedics to enter new messages and retrieve classification results in multiple categories. The web app also displays data visualizations.

### File Descriptions
app

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data

|- disaster_categories.csv # categories data to process
|- disaster_messages.csv # messages data to process
|- process_data.py # data cleaning pipeline
|- InsertDatabaseName.db # database to save clean data to


models

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


README.md


### Components
There are three components in this project:

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

    - Loads the messages and categories datasets
    - Merges these two datasets
    - Cleans the data
    - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation` was used to prepare the process_data.py python script.
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used to prepare the train_classifier.py python script.

#### 3. Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The outputs are shown below:

![image](https://user-images.githubusercontent.com/69283201/190128456-c863edf5-e130-432c-aa35-6496332dedf0.png)

![image](https://user-images.githubusercontent.com/69283201/190128550-cfa264d4-4665-4d3c-8e07-b7fe4a7515c0.png)

![image](https://user-images.githubusercontent.com/69283201/190128676-8baec8d8-ad79-40e9-bc0c-9c189dab4585.png)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for starter code for the web app. 
