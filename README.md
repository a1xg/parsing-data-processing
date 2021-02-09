Parsing-data-processing
=====================================================================
A set of tools for processing СSV files, based on a Pandas dataframe, with the possibility of latent semantic analysis of texts.
- A set of tools for processing tabular data of the DSV obtained by parsing web pages.
- Released:
- Delete row of data by keyword
- Deleting a data line by a list of pre-marked keywords from a CSV file
- Removing keywords
- Latent semantic analysis for the selected data column (the clustering result is not
- always correct, requires an experimental approach).
- Assigning an LSA cluster label to data rows.
- save CSV

Quick Start:
===========
Path to CSV file (File encoding must be UTF-8, ';' separated)
test dataset: 200 different companies with contact details,
information about the region, etc. The text information
is not complete, written in different forms using special characters.
```python
DIR = 'test_dataset.csv'
```

Create a class object and specify the path of the CSV file,
the main language of the analyzed data and the minimum word
length, less than which words will be deleted
```python
data = CSV_preprocessing(dir=DIR, lang='russian', min_len=2)
```
Removing invalid keywords
```python
words = ['фгуп','пао','оао','ао','зао']
data.replaceKeywords(label='name', keywords=words, stemming=False)
```
Removing invalid keywords
```python
words3 = ['торговая','компания']
data.replaceKeywords(label='about', keywords=words3, stemming=True)
```
Select a data column for analysis and delete the entire data
row if it contains ignored words
```python
words2 = ['другие', 'страны', 'беларусь', 'украина']
data.removeRowByKeyword(label='region', keywords=words2, stemming=False)
```
Select the data column, load the prepared list of marked words.
For example, a row whose cell contains the word, we want to save, are marked with "1".
Rows containing words marked with "0" will be deleted.
```python
data.removeRowByBolean(label='region', dir='city_database/city.csv', stemming=False)
```
We prepare the text from the data column of interest, for which we need to make an LSF
Warning, rows containing NaN cells will be removed from the dataframe.
Rows, the analyzed cell of which contains only unique words
that do not occur in the dataset will also be deleted,this does not affect the analysis result.
```python
data.getLSAdata('about')
```

Run LSA analysis
```python
data.runLSA(threshold=0.05)
```
The threshold parameter is selected for each data set individually,
and determines the level of association of dendrogram branches at which
the clusters will be cut and selected.
Unfortunately, the quality of clustering depends only on the dataset and
cannot always give a sufficient level of correct answers on the labeled data.


If the clustering result meets the expectations, then add labels to the additional column of the dataframe
```python
data.addLSAmarks()
print(data.df)
```

Save the updated dataset in CSV
```python
data.write_csv()
```
