# import libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories,on='id')
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    categories=categories.iloc[1:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    categories.head()
    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])
    df.head()
    return df


def clean_data(df):
    print(df.duplicated().sum())
    df=df.drop_duplicates() 
    print(df.duplicated().sum())
    return df


def save_data(df, engine):
     df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db')
        print(df.columns)
        save_data(df, engine)
        
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