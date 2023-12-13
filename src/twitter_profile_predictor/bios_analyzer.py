###########################################
# Loading the module
###########################################

######################
# Tokenize
######################
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# 0 - downloading stopwords and punctuation
nltk.download('punkt')
nltk.download('stopwords')

stopwords_en = stopwords.words('english')
stopwords_fr = stopwords.words('french')

######################
# Identify keywords
######################

import pandas as pd
from ast import literal_eval

# 0. Helping functions
##############

def listation(literal):
        """Get a list of words and bi-words in str format and return the list format.
        e.g.: "'ab',('cd','ef')"  -> ["ab",("cd","ef")]
        
        Args:
            literal (str): the string representing the list
        """
        listed=[]
        literal = literal.replace(' ','').split(',')
        for k in range(len(literal)) :
            if (literal[k][0]=='(') :
                listed.append(literal_eval(literal[k]+","+literal[k+1]))
            elif (literal[k][-1]!=')') :
                listed.append(literal[k])
        return(listed)

def process_biwords(x) :
    """Transform tuple biwords in x into str biwords. If the input format is wrong return False.
    e.g.: ["ab",("cd","ef")] -> ["ab","cd ef"]
    e.g.: ("ab","cd") -> "ab cd"
    e.g.: 3 -> False

    Args:
        x (str, list): the element containing biwords
    """
    if type(x) == list :
        for k in range(len(x)) : # if x is a list
            if type(x[k]) == tuple :
                x[k] = x[k][0]+' '+x[k][1] # if biword
            else :
                x[k] = x[k]

    elif type(x) == tuple :
        x = x[0]+' '+x[1]   #si biword    # x is biword
        
    elif type(x) == str :
        x = x

    else : 
        x = False

    return(x)

# 1. Loading keywords dicts
##############

import pkg_resources

# Assuming this code is in module_file.py
data_path = pkg_resources.resource_filename(__name__, 'data/Keywords_professions_statuses.xlsx')

def load_professions_keyword_dict() :
    """
    Load the .xslx file in the data, and create a dict where each professional keyword is mapped to a profession
    """
    # Load profession dataframe
    df_map_word_professions = pd.read_excel(data_path, sheet_name='Professions')
    
    # format the string to list
    df_map_word_professions['Keywords'] = df_map_word_professions['Keywords'].apply(listation)

    # build a map from keywords to professions
    map_ProKeywords_to_professions ={}
    for k in range(len(df_map_word_professions)):
        for key in df_map_word_professions['Keywords'][k] :
            map_ProKeywords_to_professions[key] = df_map_word_professions['Professions'][k]
    
    return(map_ProKeywords_to_professions)

def load_statuses_keyword_dict() :
    """
    Load the .xslx file in the data, and create a dict where each status keyword is mapped to a status
    """
    # Load profession dataframe
    df_map_word_status = pd.read_excel(data_path, sheet_name='Titres')

    # format the string to lists
    df_map_word_status['Keywords'] = df_map_word_status['Keywords'].apply(listation)

    # build a map from keywords to statuses
    map_StatKeywords_to_statuses={}
    for k in range(len(df_map_word_status)):
        for key in df_map_word_status['Keywords'][k] :
            map_StatKeywords_to_statuses[key] = df_map_word_status['Titres'][k]
    
    return(map_StatKeywords_to_statuses)

map_ProKeywords_to_professions = load_professions_keyword_dict()
map_StatKeywords_to_statuses = load_statuses_keyword_dict()


###########################################
# Module
###########################################

class bios_analyzer() :
    """
    A class processing a bios string, offering methods to analyze the string as a Twitter(X) bios
    """
    def __init__(self, bios="") :
        
        self.bios = bios

    def tokenize(self) :
        """Return self.tokens : the tokenize version in the bios (list of words with no punctuation nor stopwords).
        """
        # 1 - remove punctuation

        self.tokens = word_tokenize(self.bios.lower().translate(str.maketrans('', '', string.punctuation)), language='french')

        # 2 - Remove the stop word

        self.tokens = [token for token in self.tokens if ((token not in stopwords_en) and (token not in stopwords_fr))]

        return(self.tokens)
    
    def bi_tokenize(self) :
        """Return self.bi_tokens : the bi-tokens in the bios (list of tuple of following tokens).
        """
        if not(hasattr(self,'tokens')) :
            self.tokenize()
        
        self.bi_tokens = list(nltk.bigrams(self.tokens))

        return(self.bi_tokens)
    
    def full_tokenize(self) :
        """Return self.full_tokens : the list of tokens and bi-tokens in the bios.
        """
        if not(hasattr(self,'tokens')) :
            self.tokenize()
        if not(hasattr(self,'bi_tokens')) :
            self.bi_tokenize()
        self.full_tokens = self.tokens + self.bi_tokens

    def get_professions(self) :
        """Return self.professions : the list of the professions declared in the bios.
        """
        
        # Build tokens from bios
        if not(hasattr(self,'full_tokens')) :
            self.full_tokenize()

        # Identify professions in tokens
        self.professions=[]
        for token in self.full_tokens :
            try : 
                self.professions.append(map_ProKeywords_to_professions[token])
            except:
                a=0
        
        self.professions = list(set(self.professions))

        return(self.professions)

    def get_statuses(self) :
        """Return self.statuses : the list of the statuses declared in the bios.
        """
        
        # Build tokens from bios
        if not(hasattr(self,'full_tokens')) :
            self.full_tokenize()

        # Identify statuses in tokens
        self.statuses=[]
        for token in self.full_tokens :
            try : 
                self.statuses.append(map_StatKeywords_to_statuses[token])
            except:
                a=0
        
        self.statuses = list(set(self.statuses))

        return(self.statuses)
