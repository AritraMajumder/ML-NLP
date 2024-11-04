import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#import nltk
#ltk.download('stopwords')

stemmer = PorterStemmer()
def stemming(content):
    stemmed = re.sub('[^a-zA-Z]',' ',content) #substitute 1 with 2 in 3. 1 is not in alphabetical
    stemmed = stemmed.lower()                 #just taking alphabets and excluding else
    stemmed = stemmed.split()
    stemmed = [stemmer.stem(word) for word in stemmed if not word in stopwords.words('english')]#if not stopwords then find stem
    stemmed = ' '.join(stemmed)
    return stemmed #string of all stemmed words joined by ' '