import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

# A set of tools for processing tabular data of the DSV obtained by parsing web pages.
# Released:
# Delete row of data by keyword
# Deleting a data line by a list of pre-marked keywords from a CSV file
# Removing keywords
# Latent semantic analysis for the selected data column (the clustering result is not
# always correct, requires an experimental approach).
# Assigning an LSA cluster label to data rows.
# save CSV


class CSV_preprocessing:
    def __init__(self, dir, lang, min_len):
        self.dir = dir # CSV directory
        self.lang = lang # The main language of the analyzed texts
        self.df = self.read_csv(path=self.dir) #  Pandas dataframe
        print('labels:', self.df.head())
        # min lenght of words. It is used to filter words by the number
        # of characters, words less than min_len will be removed
        self.min_len = min_len

        self.wdict = []
        self.cleared_text = []
        self.prepared_words = []
        # categories found LSA
        self.category = []

    def replaceKeywords(self, label, keywords, stemming):
        """
        The method removes custom keywords from the parsed text in the specified label column.

        label: str

        keywords: list

        stemming: bool

        The True flag of the stemming parameter will remove words that have partial similarity,
         for example, the word booking in the text will be removed if the word book is in the custom keyword list.
         The False flag for the stemming parameter enables the exact match between the keyword and the word from the text. book != books
        """
        self.df[label] = self.df[label].str.lower()
        mask = r'\b(?:{})' if stemming == True else r'\b(?:{})\b'
        self.df[label] = self.df[label].replace(mask.format('|'.join(keywords)), '', regex=True)

    def removeRowByKeyword(self, label, keywords, stemming):
        '''The method takes a column of a dataframe and deletes the entire row
          if it contains one or more keywords.
        :param label: str
            The parsed dataframe header
        :param keywords: list
            Custom keyword list written in lowercase.
        :param stemming: bool
            This parameter enables or disables keyword stemming.
        :stemming: parameter also switches regular expressions:
            The True flag will allow searching for a fragment of the keyword in the string,
            for example the keywords 'hot' and 'book' can be found in the line 'hotel booking on holidays'
            and the row will be removed.
            The False flag enables exact matches of keywords in the string.
            The keywords 'hotel' 'booking' 'on' 'holidays' can be found in the line 'hotel booking on holidays'
            and the row will be removed.
        '''
        # remove special characters and lowercase the text
        self.df[label] = self.df[label].replace(r'[^\w]', ' ', regex=True).str.lower()
        mask = r'\b(?:{})' if stemming == True else r'\b(?:{})\b'
        ignore_words = self.__stemmer(keywords, self.min_len) if stemming == True else keywords
        self.df = self.df.loc[~self.df[label].str.contains(mask.format('|'.join(ignore_words)))]
        self.df.reset_index(drop=True, inplace=True)

    def removeRowByBolean(self, dir, label, stemming):
        '''The method accepts a CSV file in which each row contains:
         keyword (name) and boolean value(bool_val) 1 or 0 depending on whether
         we want to delete the rows containing the keyword or keep.
         For example, a row whose cell contains the word, we want to save, are marked with "1".
         Rows containing words marked with "0" will be deleted.
         The method searches for keywords only in the specified column.

        :param dir: str
        :param label: str
        :param stemming: bool
        '''
        csv_bool = self.read_csv(dir)
        csv_bool['bool_val'] = csv_bool['bool_val'].astype(str)
        csv_bool = csv_bool.loc[~csv_bool['bool_val'].str.contains('1')]
        # remove special characters and lowercase the text
        csv_bool['name'] = csv_bool['name'].replace(r'[^\w]', ' ', regex=True).str.lower()
        blacklist = csv_bool['name'].tolist()
        self.removeRowByKeyword(label, blacklist, stemming)

    def getLSAdata(self, label):
        '''We carry out preparatory operations for LSA text analysis.
        :label: str

        :threshold: float

        The threshold parameter determines the level at which the dendrogram tree will be
        truncated and classes will be assigned to each element of the dataset.
        The color of the branches of the dendrogram corresponds to the found cluster.
        The threshold should be selected individually for the dataset
        '''
        # remove special characters and lowercase the text.
        dfcopy = self.df.copy()
        dfcopy[label] = dfcopy[label].replace(r'[^\w]', ' ', regex=True).str.lower()
        # Delete rows containing nan cells
        dfcopy.dropna(subset=[label], inplace=True)
        # Tokenizing strings into separate words.
        dfcopy[label] = dfcopy[label].str.split()
        stem_single_words = self.__singleWordDetection(dfcopy[label])
        # Stopwords to be excluded from the examined cell of the dataframe.
        stopwords = nltk.corpus.stopwords.words(self.lang)
        stem_stopwords = self.__stemmer(stopwords, min_len=1)
        # cleared_text Array of sentences cleared of stop words, single words and passed stemming.
        self.cleared_text = self.__sentencePreparation(dfcopy[label], stem_single_words, stem_stopwords)
        # wdict Dictionary of word inclusions, contains word as key and index string inclusions as value.
        # prepared_words Processed array of words post stemming, after clearing of single, stop words.
        self.wdict, self.prepared_words = self.__getWdict(self.cleared_text)
        print('Completed text preparation for LSA')

    def runLSA(self, threshold):
        lsa_data = (self.cleared_text, self.wdict, self.prepared_words)
        lsa = LSA(lsa_data, threshold=threshold)
        self.category = lsa.labels

    def addLSAmarks(self):
        self.df['category'] = self.category

    def __stemmer(self, words_list, min_len):
        """
        The method takes a list of source words and returns a list of words after stemming.
        :param words_list: list
        :param min_len: int
        :return: list
        """
        stemmer = SnowballStemmer(self.lang)
        stemmed_words_list = [stemmer.stem(word) for word in words_list if len(word) > min_len]
        return stemmed_words_list

    def __wordFilter(self, words, ignore_words):
        '''The method takes a list of parsed words and a list of ignored words,
         returns a list cleared of ignored words'''
        filtred_words = [word for word in words if word not in ignore_words]
        return filtred_words

    def __singleWordDetection(self, texts):
        ''' The method accepts lists containing words,
         combines several lists into one,
         carries out stemming of words,
         Returns a list of single words.
        '''
        merged_sentences = []
        [merged_sentences.extend(sentence) for sentence in texts]
        stemmed_words = self.__stemmer(merged_sentences, self.min_len)
        fdist = nltk.FreqDist(stemmed_words)
        single_words = fdist.hapaxes()
        return single_words

    def __sentencePreparation(self, texts, single_words, stopwords):
        '''The method accepts an array of sources, an array of stop words, an array of single words.
            Returns an array of keywords prepared for building the frequency matrix.слов.
        '''
        cleared_words = []
        remove_indices = [] # List of indices for remove.
        for index, wordlist in enumerate(texts):
            stemmed_words = self.__stemmer(wordlist, self.min_len)
            del_stopwords = self.__wordFilter(stemmed_words, stopwords)
            del_single_words = self.__wordFilter(del_stopwords, single_words)
            # For further analysis, add only those lines with more than 1 word
            if len(del_single_words) > 0:
                cleared_words.append(del_single_words)
            else:
                remove_indices.append(index)
                print('Will be deleted:', index, wordlist)
        self.df.drop(labels=remove_indices, axis=0, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("Removed {a} rows with empty analyzed item, with indices: {b}".format(a=len(remove_indices), b=remove_indices))

        return cleared_words

    def __getWdict(self, cleared_text):
        '''The method searches each sentence for words that are in self.prepared_words
           if a word is found, this word is added to the self.wdict dictionary as a key,
           the indices of sentences in which the word occurs are added to the dictionary as values
        '''
        wdict = {}
        prepared_words = []
        for i in range(0, len(cleared_text)):
            for word in cleared_text[i]:
                if word not in prepared_words:
                    prepared_words.append(word)
                    wdict[word] = [i]
                elif word in prepared_words:
                    wdict[word] = wdict[word] + [i]

        return wdict, prepared_words

    def read_csv(self, path):
        '''Method for opening CSV files, returns dataframe with data'''
        dataframe = pd.read_csv(path, sep=';', header=0, encoding="utf-8")
        return dataframe

    def write_csv(self):
        '''CSV writer, accepts a Pandas dataframe with data, and a directory to write the file'''
        dir = os.path.dirname(self.dir)
        name = 'new_' + os.path.basename(self.dir)
        self.df.to_csv(os.path.join(dir, name), sep=';', encoding='utf-8', header=True, index=False)


class LSA:
    '''Latent semantic analysis of texts
     Based on the code from: https://habr.com/ru/post/335668/
    '''
    def __init__(self, lsa_data, threshold):
        # Unpack data for LSA
        self.cleared_text, self.wdict, self.prepared_words = lsa_data
        self.threshold = threshold

        self.labels = []
        self.matrix = self.__freq_matrix()
        self.U, self.S, self.Vt, self.data_word = self.__svd()
        self.show_result()

    def __freq_matrix(self):
        '''Frequency matrix construction and normalization'''
        matrix = np.zeros([len(self.prepared_words), len(self.cleared_text)])
        self.prepared_words.sort()

        for i, k in enumerate(self.prepared_words):
            for j in self.wdict[k]:
                matrix[i, j] += 1

        # TF-IDF matrix
        wpd = np.sum(matrix, axis=0)
        dpw = np.sum(np.asarray(matrix > 0, 'i'), axis=1)
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                m = float(matrix[i, j]) / wpd[j]
                n = np.log(float(cols) / dpw[i])
                matrix[i, j] = round(n * m, 2)

        return matrix

    def __svd(self):
        # Singular value decomposition of a matrix
        U, S, Vt = np.linalg.svd(self.matrix)
        rows, cols = U.shape
        for j in range(0, cols):
            for i in range(0, rows):
                U[i, j] = round(U[i, j], 4)

        res1 = -1 * U[:, 0:1];
        res2 = -1 * U[:, 1:2]
        data_word = []
        for i in range(0, len(self.prepared_words)):  # Preparation input data
            data_word.append([res1[i][0], res2[i][0]])

        return (U, S, Vt, data_word)

    def show_result(self):
        plt.figure()
        dist = pdist(self.data_word, 'euclidean')  # Calculate Euclidean distance
        Z = hierarchy.linkage(dist, method='average')  # allocating clusters
        hierarchy.dendrogram(Z, labels=self.prepared_words, color_threshold=self.threshold, leaf_font_size=8,
                             count_sort=True, orientation='right')

        rows, cols = self.Vt.shape
        for j in range(0, cols):
            for i in range(0, rows):
                self.Vt[i, j] = round(self.Vt[i, j], 4)
        res3 = (-1 * self.Vt[0:1, :])
        res4 = (-1 * self.Vt[1:2, :])

        data_docs = [];
        row_indices = []
        for i in range(0, len(self.cleared_text)):
            row_indices.append(str(i))
            data_docs.append([res3[0][i], res4[0][i]])

        dist = pdist(data_docs, 'euclidean')
        Z = hierarchy.linkage(dist, method='average')

        self.labels = fcluster(Z, t=self.threshold, criterion='distance')
        print('CLUSTERS: \n', self.labels)
        plt.figure()
        hierarchy.dendrogram(Z, labels=row_indices, color_threshold=self.threshold, leaf_font_size=8, count_sort=True, orientation='right')
        plt.show()



