""" TEXT VECTORIZATION
Bag of Words model

Algorithm of vectorization that allows to vectorize text by losing the order of the words.
Each word of the text becomes a column(feature) of the matrix (text of 100.000 words becomes a 100.000 columns matrix),
repeated words are merged into a single feature.

Bag of words:           | first | hello | world | again | new |
first hello world =>    |  1    |  1    |  1    |  0    |  0  |
hello world again =>    |  0    |  1    |  1    |  1    |  0  |
new world hello =>      |  0    |  1    |  1    |  0    |  1  |

Each string is vectorized into a vector of 0 and 1, where 1 means that the word is present in the string.

TFIDF (Term Frequency - Inverse Document Frequency)
More complex algoritmh based on this that able to take control of the frequency of the words in the text.

"""

from sklearn.feature_extraction.text import CountVectorizer

X = [
    'first hello world',
    'hello world again',
    'world world hello',
]

vectorizer = CountVectorizer()

vectorizer.fit(X)

X = vectorizer.transform(X)

# Array of the features identified in the text
# ['again' 'first' 'hello' 'world']
print(vectorizer.get_feature_names_out())
""" A matrix-like representation of the text
[[0 1 1 1]
 [1 0 1 1]
 [0 0 1 2]]
 """
print(X.todense())
# Raw representation of the text
print(X)
