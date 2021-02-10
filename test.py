from csv_postprocessing import CSV_preprocessing


# Package version:
# python 3.9
# numpy 1.19.5
# pandas 1.1.4
# nltk 3.5
# scipy 1.6.0
# matplotlib 3.3.3



# Path to CSV file (File encoding must be UTF-8, ';' separated)
DIR = 'test_dataset.csv'

# Create a class object and specify the path of the CSV file,
data = CSV_preprocessing(dir=DIR, lang='russian', min_len=2)

# Removing invalid keywords
words = ['фгуп','пао','оао','ао','зао']
data.replaceKeywords(label='name', keywords=words, stemming=False)

# Removing invalid keywords
words3 = ['торговая','компания']
data.replaceKeywords(label='about', keywords=words3, stemming=True)

# Select a data column for analysis and delete the entire data row if it contains ignored words
words2 = ['другие', 'страны', 'беларусь', 'украина']
data.removeRowByKeyword(label='region', keywords=words2, stemming=False)

# Select the data column, load the prepared list of marked words.
data.removeRowByBolean(label='region', dir='city_database/city.csv', stemming=False)

# We prepare the text from the data column of interest, for which we need to make an LSA
data.getLSAdata('about')

# Run LSA analysis
data.runLSA(threshold=0.05)

# If the clustering result meets the expectations, then add labels to the additional column of the dataframe
data.addLSAmarks()
print(data.df)

# save the updated dataset in CSV
data.write_csv()




