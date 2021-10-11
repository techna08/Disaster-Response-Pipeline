import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlite3
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * from messages', engine) 
    X=df.message
    Y=df.drop(['id', 'message','original','genre'], axis=1)
    print(df.columns)
    for col in Y.columns:
        Y[col] = Y[col].fillna(method='bfill').apply(str)
        Y[col]  = Y[col].apply(lambda x : int(float(x)))
    X=X.fillna(method='ffill')
    return X,Y

def tokenize(text):
    urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@/.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(urlRegex, text)
    for url in detected_urls:
        message = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleanTokens=[]
    for token in tokens:
        cleanToken=lemmatizer.lemmatize(token).lower().strip()
        if(cleanToken != ''):
            cleanTokens.append(cleanToken)
    return cleanTokens


def build_model():
    dtc =  DecisionTreeClassifier(random_state=0)
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=dtc))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, Y):
    y_pred = model.predict(X_test)
    for columns in Y:
        print(classification_report(Y_test, y_pred, digits=2))


def save_model(model, model_filepath):
    # Saving the trained model.   
    pickle.dump(model, open(model_filepath, 'wb'))
    # Load the pickled model
    loaded_model = pickle.load(open(model_filepath, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result)
    # We can use Test data here directly for predictions
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        np.unique(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test,Y)

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