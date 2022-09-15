# import libraries
import sys
import os
import pandas as pd 
import numpy as np 
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load datasets from 2 filepaths and merge together.
    
    Parameters:
    messages_filepath: messages csv file
    categories_filepath: categories csv file
    
    Returns:
    df: dataframe containing messages_filepath and categories_filepath merged
    
    """


    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how ='inner', on='id')

    return df


def clean_data(df):
    """
    Clean the dataframe.

    Parameters:
    df : DataFrame

    Returns:
    df : Cleaned dataframe
    """
    # create a dataframe of all the category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.head()

    # extract list of new column names for categories
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    category_colnames = category_colnames.tolist()

    # rename the columns of `categories` dataframe
    categories.columns = category_colnames

    # convert categories value to 1 and 0 only
    for column in categories:
        # set each value to be the last character of the string (1 or 0)
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)
    
    # replace 2s with 1s in related column
    categories['related'] = categories['related'].replace(to_replace=2, value=1)

    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1, join = 'inner')
    
    # drop duplicates rows
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filepath):
    """
    Store the Dataframe in a SQLite database
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()