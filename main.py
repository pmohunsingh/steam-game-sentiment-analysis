import pandas as pd
import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
##################################### Load and merge data ###########################################
def extract_game_name(file_path):
    # Extracting the filename from the file path
    filename = os.path.basename(file_path)
    
    # Splitting the filename at '_' and removing the leading number
    # Assuming format is 'number_gamename.csv'
    parts = filename.split('_')
    if len(parts) > 1:  # Check if the split was successful
        game_name = '_'.join(parts[1:])  # Join all parts except the number
        game_name = game_name.replace('.csv', '')  # Remove the file extension
        return game_name
    else:
        return "Unknown"  # Default value if the filename format is unexpected

## Path to directory where all CSV files live
directory_path = 'C:\\Users\\pmohu\\OneDrive\\game_rvw_csvs'
## Empty list to store dataframes
dataframes_list = []
## Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        game_name = extract_game_name(file_path)
        df = pd.read_csv(file_path, low_memory=False)  
        df['game_name'] = game_name
        dataframes_list.append(df)
## Combine all dataframes
combined_df = pd.concat(dataframes_list, ignore_index=True)

################################# Preprocess Combined DataFrame #######################################

## Display first few rows of DataFrame
print(combined_df.head())
## Get summary of DataFrame
print(combined_df.info())
## Check for duplicates
print(combined_df.duplicated().sum())
## Handle duplicates
combined_df = combined_df.drop_duplicates()
## Check for missing values
print(combined_df.isnull().sum())
## Handle missing values
combined_df = combined_df.dropna()
combined_df['review'] = combined_df['review'].fillna("No review")
## Convert data type(s)
df['votes_up'] = pd.to_numeric(df['votes_up'], errors='coerce')

## Function to clean and preprocess text
def preprocess_text(text):
    ## Convert to lowercase
    text = text.lower()
    
    ## Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    ## Tokenize
    words = word_tokenize(text)
    
    ## Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    ## Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

## Apply preprocessing to the review text column
combined_df['review'] = combined_df['review'].apply(preprocess_text)

## Standardize Numerical Data
scaler = StandardScaler()
combined_df['votes_up'] = scaler.fit_transform(combined_df[['votes_up']])

## Encoding Categorical Variables using One-Hot encoding
combined_df = pd.get_dummies(combined_df, columns=['language'])

###################################### Feature Engineering ##########################################

## Define a function to calculate sentiment score
def get_sentiment_score_textblob(text):
    # Convert to string in case the review is not in text format
    text = str(text)
    return TextBlob(text).sentiment.polarity

## Apply the function to DataFrame
combined_df['sentiment_score'] = combined_df['review'].apply(get_sentiment_score_textblob)

## Save Preprocessed data for further use
combined_df.to_csv('processed_combined_data.csv', index=False)
print(combined_df[['review', 'sentiment_score']].head())