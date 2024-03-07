###########################################
# Loading the module
###########################################

######################
# Language recognition
######################

import nltk
import cld2
import langdetect
langdetect.DetectorFactory.seed = 0
import langid

# Define language recognition models
def detlang(text):
    """
    Detect the language of the given text using the langdetect library.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: The detected language code.
    """
    try:
        return langdetect.detect(text)
    except:
        return 0

def detlang_id(text):
    """
    Detect the language of the given text using the langid library.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: The detected language code.
    """
    try:
        return langid.classify(text)[0]
    except:
        return 0

def detlang_cld(text):
    """
    Detect the language of the given text using the cld2 library.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: The detected language code.
    """
    try:
        results = cld2.detect(text)
        if results.is_reliable:
            # Extract language code
            return results.details[0].language_code
        else:
            return 0
    except:
        return 0

def lang(text, mainfrench=True):
    """
    Use 3 different language models to identify the language of the text by majority vote.

    Args:
        text (str): The text to identify the language of.
        mainfrench (bool): True if the expected language is French.

    Returns:
        str or bool: The detected language code or False if no unique language is found.
    """
    langdet = detlang(text)
    langid = detlang_id(text)
    langcld = detlang_cld(text)

    if mainfrench:
        if langdet == 'fr' or langid == 'fr' or langcld == 'fr':
            return 'fr'  # WARNING: favor French because of prior of Twitter corpus

    if langdet == langid:
        return langdet
    elif langdet == langcld:
        return langdet
    elif langid == langcld:
        return langid
    else:
        return False

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
    """
    Convert a string representation of a list of words and bi-words into a list format.

    Args:
        literal (str): The string representing the list.

    Returns:
        list: The list of words and bi-words.
    """
    listed = []
    literal = literal.replace(' ', '').split(',')
    for k in range(len(literal)):
        if literal[k][0] == '(':
            listed.append(literal_eval(literal[k] + "," + literal[k + 1]))
        elif literal[k][-1] != ')':
            listed.append(literal[k])
    return listed

def process_biwords(x):
    """
    Transform tuple bi-words in x into string bi-words.

    Args:
        x (str, list): The element containing bi-words.

    Returns:
        str or list: The transformed bi-words.
    """
    if type(x) == list:
        for k in range(len(x)):
            if type(x[k]) == tuple:
                x[k] = x[k][0] + ' ' + x[k][1]
            else:
                x[k] = x[k]

    elif type(x) == tuple:
        x = x[0] + ' ' + x[1]

    elif type(x) == str:
        x = x

    else:
        x = False

    return x

# 1. Loading keywords dicts
##############

import pkg_resources

# Assuming this code is in module_file.py
data_path = pkg_resources.resource_filename(__name__, 'data/df_words_twitterUsersAnalysis.xlsx')

def load_all_dicts():
    """
    Load all the keyword dataframes and transform them into dictionaries.

    Returns:
        tuple: A tuple containing the dictionaries for each keyword category.
    """
    # Load keyword dataframes (classes to keywords)
    df_map_word_professions = pd.read_excel(data_path, sheet_name='Professions')

    df_map_word_prostatus = pd.read_excel(data_path, sheet_name='Professional Statuses')
    df_map_word_actorstatus = pd.read_excel(data_path, sheet_name='Actor Type Status')
    df_map_word_groupstatus = pd.read_excel(data_path, sheet_name='Group Status')
    df_map_word_universitystatus = pd.read_excel(data_path, sheet_name='University Status')
    df_map_word_allstatus = pd.read_excel(data_path, sheet_name='Status')

    df_map_word_age = pd.read_excel(data_path, sheet_name='Age')
    df_map_word_gender = pd.read_excel(data_path, sheet_name='Genre')

    df_map_word_topic = pd.read_excel(data_path, sheet_name='Topics')

    # format list of keywords (str to list)
    df_map_word_professions['Keywords'] = df_map_word_professions['Keywords'].apply(listation)

    df_map_word_prostatus['Keywords'] = df_map_word_prostatus['Keywords'].apply(listation)
    df_map_word_actorstatus['Keywords'] = df_map_word_actorstatus['Keywords'].apply(listation)
    df_map_word_groupstatus['Keywords'] = df_map_word_groupstatus['Keywords'].apply(listation)
    df_map_word_universitystatus['Keywords'] = df_map_word_universitystatus['Keywords'].apply(listation)
    df_map_word_allstatus['Keywords'] = df_map_word_allstatus['Keywords'].apply(listation)

    df_map_word_age['Keywords'] = df_map_word_age['Keywords'].apply(listation)
    df_map_word_gender['Keywords'] = df_map_word_gender['Keywords'].apply(listation)

    df_map_word_topic['Keywords'] = df_map_word_topic['Keywords'].apply(listation)

    # transform the keywords dataframe to dictionaries (keywords to classes)
    pro_key = {}
    for k in range(len(df_map_word_professions)):
        for key in df_map_word_professions['Keywords'][k]:
            pro_key[key] = df_map_word_professions['English'][k]

    prostatus_key = {}
    for k in range(len(df_map_word_prostatus)):
        for key in df_map_word_prostatus['Keywords'][k]:
            prostatus_key[key] = df_map_word_prostatus['English'][k]

    actorstatus_key = {}
    for k in range(len(df_map_word_actorstatus)):
        for key in df_map_word_actorstatus['Keywords'][k]:
            actorstatus_key[key] = df_map_word_actorstatus['English'][k]

    groupstatus_key = {}
    for k in range(len(df_map_word_groupstatus)):
        for key in df_map_word_groupstatus['Keywords'][k]:
            groupstatus_key[key] = df_map_word_groupstatus['English'][k]

    universitystatus_key = {}
    for k in range(len(df_map_word_universitystatus)):
        for key in df_map_word_universitystatus['Keywords'][k]:
            universitystatus_key[key] = df_map_word_universitystatus['English'][k]

    allstatus_key = {}
    for k in range(len(df_map_word_allstatus)):
        for key in df_map_word_allstatus['Keywords'][k]:
            allstatus_key[key] = df_map_word_allstatus['English'][k]

    age_key = {}
    for k in range(len(df_map_word_age)):
        for key in df_map_word_age['Keywords'][k]:
            age_key[key] = df_map_word_age['Age'][k]

    gender_key = {}
    for k in range(len(df_map_word_gender)):
        for key in df_map_word_gender['Keywords'][k]:
            gender_key[key] = df_map_word_gender['English'][k]

    topic_key = {}
    for k in range(len(df_map_word_topic)):
        for key in df_map_word_topic['Keywords'][k]:
            topic_key[key] = df_map_word_topic['English'][k]

    ## PCS groups
    professions_to_PCSgroups = df_map_word_professions.set_index('English')['Group'].to_dict()
    # 2.2- Status
    status_to_stutustype = df_map_word_allstatus.set_index('English')['Type'].to_dict()

    return (
        pro_key,
        prostatus_key,
        actorstatus_key,
        groupstatus_key,
        universitystatus_key,
        allstatus_key,
        age_key,
        gender_key,
        topic_key,
        professions_to_PCSgroups,
        status_to_stutustype
    )

map_Keywords_to_professions, map_Keywords_to_prostatuses, map_Keywords_to_actorstatuses, map_Keywords_to_groupstatuses, map_Keywords_to_universitystatuses, map_Keywords_to_allstatuses, map_Keywords_to_ages, map_Keywords_to_gender, map_Keywords_to_topics, map_professions_to_PCSgroups, map_status_to_stutustype = load_all_dicts()


###########################################
# Module
###########################################

class bios_analyzer:
    """
    A class processing a bios string, offering methods to analyze the string as a Twitter(X) bios.
    """

    def __init__(self, bios=""):
        """
        Initialize the bios_analyzer object.

        Args:
            bios (str, optional): The bios string to analyze. Defaults to "".
        """
        self.bios = bios

    def tokenize(self, bios=None):
        """
        Tokenize the bios string by removing punctuation and stopwords.

        Args:
            bios (str, optional): The bios string to tokenize. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of tokens in the bios string.
        """
        if bios is not None:
            self.bios = bios

        # 1 - remove punctuation
        self.tokens = word_tokenize(self.bios.lower().translate(str.maketrans('', '', string.punctuation)), language='french')
        # 2 - Remove the stop words
        self.tokens = [token for token in self.tokens if ((token not in stopwords_en) and (token not in stopwords_fr))]

        return self.tokens

    def bi_tokenize(self, bios=None):
        """
        Tokenize the bios string into bi-tokens.

        Args:
            bios (str, optional): The bios string to tokenize. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of bi-tokens in the bios string.
        """
        if bios is not None:
            self.bios = bios
            self.tokenize()
        elif not hasattr(self, 'tokens'):
            self.tokenize()

        self.bi_tokens = list(nltk.bigrams(self.tokens))

        return self.bi_tokens

    def full_tokenize(self, bios=None):
        """
        Tokenize the bios string into tokens and bi-tokens.

        Args:
            bios (str, optional): The bios string to tokenize. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of tokens and bi-tokens in the bios string.
        """
        if bios is not None:
            self.bios = bios
            self.tokenize()
            self.bi_tokenize()
        if not hasattr(self, 'tokens'):
            self.tokenize()
        if not hasattr(self, 'bi_tokens'):
            self.bi_tokenize()

        self.full_tokens = self.tokens + self.bi_tokens

        return self.full_tokens

    def get_professions(self, bios=None):
        """
        Return the list of professions declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of professions declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()

        # Identify professions in tokens
        self.professions = []
        for token in self.full_tokens:
            try:
                self.professions.append(map_Keywords_to_professions[token])
            except:
                pass

        self.professions = list(set(self.professions))

        return self.professions

    def get_prostatus(self, bios=None):
        """
        Return the list of professional statuses declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of professional statuses declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()

        # Identify statuses in tokens
        self.prostatus = []
        for token in self.full_tokens:
            try:
                self.prostatus.append(map_Keywords_to_prostatuses[token])
            except:
                pass

        self.prostatus = list(set(self.prostatus))

        return self.prostatus

    def get_actorstatuses(self, bios=None):
        """
        Return the list of actor statuses declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of actor statuses declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()

        # Identify statuses in tokens
        self.actorstatuses = []
        for token in self.full_tokens:
            try:
                self.actorstatuses.append(map_Keywords_to_actorstatuses[token])
            except:
                pass

        self.actorstatuses = list(set(self.actorstatuses))
        return self.actorstatuses

    def get_groupstatuses(self, bios=None):
        """
        Return the list of group statuses declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of group statuses declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()
        # Identify statuses in tokens
        self.groupstatuses = []
        for token in self.full_tokens:
            try:
                self.groupstatuses.append(map_Keywords_to_groupstatuses[token])
            except:
                pass

        self.groupstatuses = list(set(self.groupstatuses))
        return self.groupstatuses

    def get_universitystatuses(self, bios=None):
        """
        Return the list of university statuses declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of university statuses declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()
        # Identify statuses in tokens
        self.universitystatuses = []
        for token in self.full_tokens:
            try:
                self.universitystatuses.append(map_Keywords_to_universitystatuses[token])
            except:
                pass

        self.universitystatuses = list(set(self.universitystatuses))
        return self.universitystatuses

    def get_allstatuses(self, bios=None):
        """
        Return the list of all statuses declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of all statuses declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()
        # Identify statuses in tokens
        self.allstatuses = []
        for token in self.full_tokens:
            try:
                self.allstatuses.append(map_Keywords_to_allstatuses[token])
            except:
                pass

        self.allstatuses = list(set(self.allstatuses))
        return self.allstatuses

    def get_ages(self, bios=None):
        """
        Return the list of ages declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of ages declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()
        # Identify ages in tokens
        self.ages = []
        for token in self.full_tokens:
            try:
                self.ages.append(map_Keywords_to_ages[token])
            except:
                pass

        self.ages = list(set(self.ages))
        if len(self.ages) > 1:
            self.ages = []

        return self.ages

    def get_gender(self, bios=None):
        """
        Return the gender declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The gender declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()
        # Identify gender in tokens
        self.gender = []
        for token in self.full_tokens:
            try:
                self.gender.append(map_Keywords_to_gender[token])
            except:
                pass

        self.gender = list(set(self.gender))

        if "Woman" in self.gender:
            self.gender = ["Woman"]

        return self.gender

    def get_topics(self, bios=None):
        """
        Return the list of topics declared in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The list of topics declared in the bios.
        """
        if bios is not None:
            self.full_tokenize(bios=bios)
        # Build tokens from bios
        if not(hasattr(self, 'full_tokens')):
            self.full_tokenize()
        # Identify topics in tokens
        self.topics = []
        for token in self.full_tokens:
            try:
                self.topics.append(map_Keywords_to_topics[token])
            except:
                pass

        self.topics = list(set(self.topics))
        return self.topics

    def get_lang(self, bios=None, mainfrench=True):
        """
        Use 3 different language models to identify the language of the bios by majority vote.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.
            mainfrench (bool, optional): If True, it means that French is the expected language and hence only one model out of 3 identifying French will be considered enough. Defaults to True.

        Returns:
            string: The recognized language standard code (e.g., "en", "fr", "sp").
        """
        if bios is not None:
            self.bios = bios

        self.language = lang(self.bios, mainfrench)
        return self.language

    def get_PCSgroup(self, bios=None):
        """
        Return the PCS group corresponding to the professions in the bios.

        Args:
            bios (str, optional): The bios string to analyze. If not provided, the bios string provided at initialization will be used.

        Returns:
            list: The PCS group corresponding to the profession.
        """
        if bios is not None:
            self.get_professions(bios=bios)
        if not(hasattr(self, 'professions')):
            self.get_professions()

        self.PCSgroup = []
        for pro in self.professions:
            self.PCSgroup.append(map_professions_to_PCSgroups[pro])
        self.PCSgroup = list(set(self.PCSgroup))

        return(self.PCSgroup)

###########################################
    # Independent functions
###########################################
    


def tokenize(bios) :
    """Tokenize the bios by removing punctuation and stopwords.

    Args:
        bios (str): The bios to be tokenized.

    Returns:
        list: A list of words with no punctuation or stopwords.
    """
    # 1 - remove punctuation
    tokens = word_tokenize(bios.lower().translate(str.maketrans('', '', string.punctuation)), language='french')

    # 2 - Remove the stop word
    tokens = [token for token in tokens if ((token not in stopwords_en) and (token not in stopwords_fr))]

    return(tokens)

def bi_tokenize(bios):
    """Return the bi-tokens in the bios (list of tuple of following tokens).

    Args:
        bios (list): A list of bios to tokenize.

    Returns:
        list: A list of bi-tokens (tuples of following tokens).
    """
    tokens = tokenize(bios)
    bi_tokens = list(nltk.bigrams(tokens))
    return bi_tokens

def full_tokenize(bios):
    """Return the list of tokens and bi-tokens in the bios.

    Args:
        bios (str): The input bios to tokenize.

    Returns:
        list: The list of tokens and bi-tokens extracted from the bios.
    """
    tokens = tokenize(bios)
    bi_tokens = bi_tokenize(bios)
    full_tokens = tokens + bi_tokens
    return full_tokens

def get_professions(bios=None, tokens=None):
    """Return the list of professions declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized list of bios. If provided, the function will use these tokens.

    Returns:
        list: A list of professions identified in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)

    # Identify professions in tokens
    professions = []
    for token in tokens:
        try:
            professions.append(map_Keywords_to_professions[token])
        except:
            pass

    professions = list(set(professions))

    return professions

def get_prostatus(bios=None, tokens=None):
    """Return the list of professional statuses declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized bios. If provided, the function will use these tokens instead of tokenizing the bios.

    Returns:
        list: A list of professional statuses found in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)

    # Identify statuses in tokens
    prostatus = []
    for token in tokens:
        try:
            prostatus.append(map_Keywords_to_prostatuses[token])
        except:
            pass

    prostatus = list(set(prostatus))

    return prostatus

def get_actorstatuses(bios=None, tokens=None):
    """Return the list of actor statuses declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized bios. If provided, the function will use these tokens instead of tokenizing the bios.

    Returns:
        list: A list of actor statuses found in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)

    # Identify statuses in tokens
    actorstatuses = []
    for token in tokens:
        try:
            actorstatuses.append(map_Keywords_to_actorstatuses[token])
        except:
            pass

    actorstatuses = list(set(actorstatuses))
    return actorstatuses

def get_groupstatuses(bios=None, tokens=None):
    """
    Return the list of group statuses declared in the bios.

    Parameters:
    - bios (str): The bios to analyze. If provided, the function will tokenize the bios.
    - tokens (list): The pre-tokenized list of bios. If provided, the function will use these tokens.

    Returns:
    - list: The list of group statuses found in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)

    # Identify statuses in tokens
    groupstatuses = []
    for token in tokens:
        try:
            groupstatuses.append(map_Keywords_to_groupstatuses[token])
        except:
            pass

    groupstatuses = list(set(groupstatuses))
    return groupstatuses

def get_universitystatuses(bios = None, tokens = None) :
    """Return the list of the university statuses declared in the bios.

    Args:
        bios (str): The bios to analyze. If not provided, tokens must be provided.
        tokens (list): The pre-tokenized bios. If not provided, bios will be tokenized.

    Returns:
        list: The list of university statuses found in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)

    # Identify statuses in tokens
    universitystatuses = []
    for token in tokens:
        try:
            universitystatuses.append(map_Keywords_to_universitystatuses[token])
        except:
            pass

    universitystatuses = list(set(universitystatuses))
    return universitystatuses

def get_allstatuses(bios=None, tokens=None):
    """Return the list of all statuses declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized bios. If provided, the function will use these tokens instead of tokenizing the bios.

    Returns:
        list: A list of all unique statuses found in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)
    
    # Identify statuses in tokens
    allstatuses = []
    for token in tokens:
        try:
            allstatuses.append(map_Keywords_to_allstatuses[token])
        except:
            pass
                
    allstatuses = list(set(allstatuses))
    return allstatuses

def get_ages(bios=None, tokens=None):
    """Return the list of ages declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized list of bios. If provided, the function will use these tokens.

    Returns:
        list: A list of ages declared in the bios.

    """
    if bios is not None:
        tokens = tokenize(bios)

    # Identify ages in tokens
    ages = []
    for token in tokens:
        try:
            ages.append(map_Keywords_to_ages[token])
        except:
            pass

    ages = list(set(ages))
    if len(ages) > 1:
        ages = []

    return ages

def get_gender(bios=None, tokens=None):
    """Return the list of genders declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized list of bios. If provided, the function will use these tokens.

    Returns:
        list: A list of genders declared in the bios.

    """
    if bios is not None:
        tokens = full_tokenize(bios)
        
    # Identify genders in tokens
    genders = []
    for token in tokens:
        if token in map_Keywords_to_gender:
            genders.append(map_Keywords_to_gender[token])
                
    genders = list(set(genders))
    if "Woman" in genders:
        genders = ["Woman"]
    
    return genders

def get_topics(bios=None, tokens=None):
    """Return the list of topics declared in the bios.

    Args:
        bios (str): The bios to analyze. If provided, the function will tokenize the bios.
        tokens (list): The pre-tokenized list of tokens. If provided, the function will use these tokens instead of tokenizing the bios.

    Returns:
        list: A list of topics identified in the bios.

    """
    if bios is not None:
        tokens = full_tokenize(bios)

    # Identify topics in tokens
    topics = []
    for token in tokens:
        if token in map_Keywords_to_topics:
            topics.append(map_Keywords_to_topics[token])

    topics = list(set(topics))
    return topics

def get_PCSgroup(bios=None, tokens=None, professions=None):
    """
    Return the PCS group corresponding to the profession.

    Parameters:
    - bios (str): The bios text to analyze. If provided, it will be tokenized and used to determine the professions.
    - tokens (list): The pre-tokenized list of words. If provided, it will be used to determine the professions.
    - professions (list): The list of professions. If provided, it will be used directly.

    Returns:
    - PCSgroup (list): The list of PCS groups corresponding to the given professions.
    """
    if bios is not None:
        tokens = full_tokenize(bios)
        professions = get_professions(tokens=tokens)
    if tokens is not None:
        professions = get_professions(tokens=tokens)

    PCSgroup = []
    for pro in professions:
        PCSgroup.append(map_professions_to_PCSgroups[pro])
    PCSgroup = list(set(PCSgroup))

    return PCSgroup


class df_bios_analyzer():
    """
    A class for processing a dataframe of bios strings, offering methods to analyze the strings as Twitter bios.

    Attributes:
        df (pandas.DataFrame): The dataframe containing the bios strings.
        description_column (str): The column name in the dataframe that contains the bios strings.
        bios_analyzer (bios_analyzer): An instance of the bios_analyzer class for analyzing individual bios strings.

    """

    def __init__(self, df, description_column):
        """
        Initialize the df_bios_analyzer with a dataframe and the name of the column containing the bios strings.

        Args:
            df (pandas.DataFrame): The dataframe containing the bios strings.
            description_column (str): The column name in the dataframe that contains the bios strings.
        """
        self.df = df
        self.description_column = description_column

    def tokenize(self):
        """
        Tokenize the bios strings in the dataframe.
        The results are stored in a new column 'tokens'.
        """
        self.df['tokens'] = self.df[self.description_column].apply(tokenize)

    def bi_tokenize(self):
        """
        Perform bi-tokenization on the bios strings in the dataframe.
        The results are stored in a new column 'bi_tokens'.
        """
        self.df['bi_tokens'] = self.df[self.description_column].apply(bi_tokenize)

    def full_tokenize(self):
        """
        Perform full tokenization on the bios strings in the dataframe.
        The results are stored in a new column 'full_tokens'.
        """
        self.df['full_tokens'] = self.df[self.description_column].apply(full_tokenize)

    def get_professions(self, tokens_column=None):
        """
        Extract professions from the bios strings in the dataframe. If a column of tokens is provided, 
        it will use these tokens instead of the bios strings.
        The results are stored in a new column 'professions'.

        Args:
            tokens_column (str, optional): The column name in the dataframe that contains the tokens. Defaults to None.
        """
        if tokens_column is not None:
            self.df['professions'] = self.df[tokens_column].apply(lambda x : get_professions(tokens=x))
        else:
            self.df['professions'] = self.df[self.description_column].apply(lambda x : get_professions(bios=x))


    def get_prostatus(self, tokens_column=None):
        """
        Extract professional statuses from the bios strings in the dataframe.
        The results are stored in a new column 'prostatus'.
        """
        if tokens_column is not None:
            self.df['prostatus'] = self.df[tokens_column].apply(lambda x : get_prostatus(tokens=x))
        else:
            self.df['prostatus'] = self.df[self.description_column].apply(lambda x : get_prostatus(bios=x))

    def get_actorstatuses(self, tokens_column=None):
        """
        Extract actor statuses from the bios strings in the dataframe.
        The results are stored in a new column 'actorstatus'.
        """
        if tokens_column is not None:
            self.df['actorstatus'] = self.df[tokens_column].apply(lambda x : get_actorstatuses(tokens=x))
        else:
            self.df['actorstatus'] = self.df[self.description_column].apply(lambda x : get_actorstatuses(bios=x))

    def get_groupstatuses(self, tokens_column=None):
        """
        Extract group statuses from the bios strings in the dataframe.
        The results are stored in a new column 'groupstatus'.
        """
        if tokens_column is not None:
            self.df['groupstatus'] = self.df[tokens_column].apply(lambda x : get_groupstatuses(tokens=x))
        else:
            self.df['groupstatus'] = self.df[self.description_column].apply(lambda x : get_groupstatuses(bios=x))

    def get_universitystatuses(self, tokens_column=None):
        """
        Extract university statuses from the bios strings in the dataframe.
        The results are stored in a new column 'universitystatus'.
        """
        if tokens_column is not None:
            self.df['universitystatus'] = self.df[tokens_column].apply(lambda x : get_universitystatuses(tokens=x))
        else:
            self.df['universitystatus'] = self.df[self.description_column].apply(lambda x : get_universitystatuses(bios=x))

    def get_allstatuses(self, tokens_column=None):
        """
        Extract all types of statuses from the bios strings in the dataframe.
        The results are stored in a new column 'allstatus'.
        """
        if tokens_column is not None:
            self.df['allstatus'] = self.df[tokens_column].apply(lambda x : get_allstatuses(tokens=x))
        else:
            self.df['allstatus'] = self.df[self.description_column].apply(lambda x : get_allstatuses(bios=x))

    def get_ages(self, tokens_column=None):
        """
        Extract ages from the bios strings in the dataframe.
        The results are stored in a new column 'age'.
        """
        if tokens_column is not None:
            self.df['age'] = self.df[tokens_column].apply(lambda x : get_ages(tokens=x))
        else:
            self.df['age'] = self.df[self.description_column].apply(lambda x : get_ages(bios=x))

    def get_gender(self, tokens_column=None):
        """
        Extract gender from the bios strings in the dataframe.
        The results are stored in a new column 'gender'.
        """
        if tokens_column is not None:
            self.df['gender'] = self.df[tokens_column].apply(lambda x : get_gender(tokens=x))
        else:
            self.df['gender'] = self.df[self.description_column].apply(lambda x : get_gender(bios=x))

    def get_topics(self, tokens_column=None):
        """
        Extract topics from the bios strings in the dataframe.
        The results are stored in a new column 'topic'.
        """
        if tokens_column is not None:
            self.df['topic'] = self.df[tokens_column].apply(lambda x : get_topics(tokens=x))
        else:
            self.df['topic'] = self.df[self.description_column].apply(lambda x : get_topics(bios=x))

    def get_lang(self, mainfrench=True):
        """
        Determine the language of the bios strings in the dataframe.

        Args:
            mainfrench (bool, optional): If true it means that French is the expected language and hence only one model on 3 identifying French will be considered enough. Defaults to True.

        The results are stored in a new column 'lang'.
        """
        self.df['lang'] = self.df[self.description_column].apply(lambda x : lang(text=x, mainfrench=mainfrench))

    def get_PCSgroup(self, tokens_column=None, professions_column=None):
        """
        Extract the PCS group corresponding to the professions in the bios strings in the dataframe.
        The results are stored in a new column 'PCSgroup'.
        """
        if tokens_column is not None:
            self.df['PCSgroup'] = self.df[tokens_column].apply(lambda x : get_PCSgroup(tokens=x))
        if professions_column is not None:
            self.df['PCSgroup'] = self.df[professions_column].apply(lambda x : get_PCSgroup(profession=x))
        else:
            self.df['PCSgroup'] = self.df[self.description_column].apply(lambda x : get_PCSgroup(bios=x))

    def get_all(self, mainfrench=True):
        """
        Perform all analyses on the bios strings in the dataframe.

        Args:
            tokens_column (str, optional): The column name in the dataframe that contains the tokens. Defaults to None.
            mainfrench (bool, optional): If true it means that French is the expected language and hence only one model on 3 identifying French will be considered enough. Defaults to True.
        """
        self.tokenize()
        self.bi_tokenize()
        self.full_tokenize()
        tokens_column = 'full_tokens'
        self.get_professions(tokens_column)
        self.get_prostatus(tokens_column)
        self.get_actorstatuses(tokens_column)
        self.get_groupstatuses(tokens_column)
        self.get_universitystatuses(tokens_column)
        self.get_allstatuses(tokens_column)
        self.get_ages(tokens_column)
        self.get_gender(tokens_column)
        self.get_topics(tokens_column)
        self.get_lang(mainfrench=mainfrench)

        return (self.df)

###########################################
# Getting list of keywords
###########################################
    
def get_pro_kewords() :
    """Return the list of professions keywords.
    """
    # Load keyword dataframes (classes to keywords)
    df_map_word_professions = pd.read_excel(data_path, sheet_name='Professions')
    # Remove the 'Professions' column
    df_map_word_professions = df_map_word_professions.drop(columns=['Professions'])
    # Rename the 'English' column to 'Professions'
    df_map_word_professions = df_map_word_professions.rename(columns={'English': 'Professions'})
    # Set 'Professions' as the index
    df_map_word_professions.set_index('Professions', inplace=True)

    return(df_map_word_professions)

def get_status_kewords() :
    """Return the list of status keywords.
    """
    # Load keyword dataframes (classes to keywords)
    df_map_word_allstatus = pd.read_excel(data_path, sheet_name='Status')
    # Remove the 'Status' column
    df_map_word_allstatus = df_map_word_allstatus.drop(columns=['Status'])
    # Rename the 'English' column to 'Status'
    df_map_word_allstatus = df_map_word_allstatus.rename(columns={'English': 'Status'})
    # Set 'Status' as the index
    df_map_word_allstatus.set_index('Status', inplace=True)

    return(df_map_word_allstatus)

def get_age_kewords() :
    """Return the list of age keywords.
    """
    # Load keyword dataframes (classes to keywords)
    df_map_word_age = pd.read_excel(data_path, sheet_name='Age')
    # Remove the 'Status' column
    df_map_word_age = df_map_word_age.drop(columns=['Status'])
    # Set 'Age' as the index
    df_map_word_age.set_index('Age', inplace=True)

    return(df_map_word_age)

def get_gender_kewords() :
    """Return the list of gender keywords.
    """
    # Load keyword dataframes (classes to keywords)
    df_map_word_gender = pd.read_excel(data_path, sheet_name='Genre')
     # Remove the 'Genre' column
    df_map_word_gender = df_map_word_gender.drop(columns=['Genre'])
    # Rename the 'English' column to 'Gender'
    df_map_word_gender = df_map_word_gender.rename(columns={'English': 'Gender'})
    # Set 'Gender' as the index
    df_map_word_gender.set_index('Gender', inplace=True)
    return(df_map_word_gender)

def get_topic_kewords() :
    """Return the list of topic keywords.
    """
    # Load keyword dataframes (classes to keywords)
    df_map_word_topic = pd.read_excel(data_path, sheet_name='Topics')
    # Remove the 'Topics' column
    df_map_word_topic = df_map_word_topic.drop(columns=['Sujet'])
    # Rename the 'English' column to 'Topics'
    df_map_word_topic = df_map_word_topic.rename(columns={'English': 'Topics'})
    # Set 'Topics' as the index
    df_map_word_topic.set_index('Topics', inplace=True)
    return(df_map_word_topic)