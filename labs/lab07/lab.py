# lab.py


import pandas as pd
import numpy as np
import os
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = '^.{2}\[.{2}]'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = '^\(858\) \d{3}-\d{4}$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes 5t?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = '^[\w\s?]{5,9}\?$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = '^\$([^abc\$]*\$)?[aA]+[bB]+[cC]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = '[a-zA-Z0-9_]+\.py$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = '^[a-z]+_[a-z]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = '^_.*\_$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = '^[^Oi1]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = '^((CA)-\d{2}-(LAX|SAN)-\d{4}|(NY)-\d{2}-([A-Z]{3})-\d{4})'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    string = string.lower().replace('a','')
    string = re.sub(r'[^\w+]+','',string)
    pattern = '.{3}'
    prog = re.compile(pattern)
    return prog.findall(string)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s):
    email_pattern = '[\w]+@[\w]+[\w.]+'
    prog_email = re.compile(email_pattern)
    email = prog_email.findall(s)

    ssn_pattern = '\d{3}-\d{2}-\d{4}'
    prog_ssn = re.compile(ssn_pattern)
    ssn  = prog_ssn.findall(s)

    bitcoin_pattern = '[a-zA-Z0-9]{34}'
    prog_bitcoin = re.compile(bitcoin_pattern)
    bitcoin  = prog_bitcoin.findall(s)

    street_pattern = '\d+ [a-zA-Z ]+'
    prog_street = re.compile(street_pattern)
    street  = prog_street.findall(s)


    return email,ssn,bitcoin,street
# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(reviews_ser, review):


    tfidf_dict = {'word':[],'cnt':[],'tf':[],'idf':[],'tfidf':[]}
    
    for word in np.unique(review.split()):
        pattern = fr'\b{word}\b'
        prog = re.compile(pattern)
        cnt = len(prog.findall(review))

        tf = cnt/ len(review.split())
        idf = np.log(len(reviews_ser) / reviews_ser.str.contains(pattern).sum() )
        tfidf = tf*idf

        tfidf_dict['word'].append(word)
        tfidf_dict['cnt'].append(cnt)
        tfidf_dict['tf'].append(tf)
        tfidf_dict['idf'].append(idf)
        tfidf_dict['tfidf'].append(tfidf)

    df = pd.DataFrame(tfidf_dict).set_index('word')

    return df
def relevant_word(out):
    return out['tfidf'].idxmax()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def hashtag_list(tweet_text):
    hashtag_pattern = r'#(\S+)'
    hashtags = tweet_text.str.findall(hashtag_pattern)
    return hashtags


def most_common_hashtag(tweet_lists):
    hashtag_counts = {}

    # Count the occurrences of each hashtag in the entire Series
    for hashtags_list in tweet_lists:
        for hashtag in hashtags_list:
            hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1

    # Define a function to get the most common hashtag in a list
    def get_most_common_in_list(hashtags_list):
        if not hashtags_list:
            return np.NAN
        return max(hashtags_list, key=lambda x: hashtag_counts[x])

    # Apply the function to each list of hashtags in the Series
    result = tweet_lists.apply(get_most_common_in_list)

    return result
# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def create_features(ira):
    df = pd.DataFrame()
    df['num_hashtags'] = hashtag_list(ira.loc[:,'text']).apply(len)
    df['mc_hashtags'] = most_common_hashtag(hashtag_list(ira.loc[:,'text']))

    tag_pattern = r'@'
    df['num_tags'] = ira['text'].str.findall(tag_pattern).apply(len)

    link_pattern = r'(http|https)(\S+)'
    df['num_links'] = ira['text'].str.findall(link_pattern).apply(lambda x: len(x) if len(x)!=0 else 0)

    retweet_pattern = r'^RT'
    df['is_retweet'] = ira['text'].str.findall(retweet_pattern).apply(lambda x: True if len(x)!=0 else False)

    df['text'] = ira['text'].str.replace(retweet_pattern,'',regex=True)\
        .str.replace(r'@\S+','',regex=True).str.replace(link_pattern,'',regex=True)\
        .str.replace( r'#(\S+)','',regex=True).str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)\
        .str.lower().str.replace(r'\s+', ' ',regex=True).str.strip()

    df = df[['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']]
    return df