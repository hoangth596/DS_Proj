# import required libraries
import sys
import re
import pickle
import pandas as pd 
from sqlalchemy import create_engine
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report


nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Parameters:
    database_filepath: filepath to the database
    
    Returns:
        X: Features
        Y: Target
    """

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)

    X = df['message']  # Message Column
    Y = df.iloc[:, 4:] # Classification label

    return X,Y


def tokenize(text):
    """
    Function: split text into words and return the root form of the words
    Args:
        text(str): the message
    Return:
        lemm(list of str): a list of the root form of the message words
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]
    
    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in words]

    return lemm


def build_model():
    """
    Build classifier and tune model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the performance of model and returns classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """

    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Exports the final model as a pickle file
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Build, train, evaluate and save the model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()