# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return 



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


#https://books.toscrape.com/catalogue/page-2.html
def extract_book_links(text):
    soup = bs4.BeautifulSoup(text)

    
    booklist = soup.find_all('article', class_ = 'product_pod')
    book_url = []
    for i in booklist:
        star = i.find('p',class_='star-rating')['class'][1]
        price = i.find('p',class_='price_color').text
        price = ''.join(c for c in price if c.isdigit() or c == '.')

        if star in ['Four','Five'] and float(price)<50:
            url = i.find('a')['href']
            book_url.append(url)
    return book_url

def get_product_info(text, categories):
    
    category_dict = {}
    soup = bs4.BeautifulSoup(text)
    table = soup.find('table',class_='table table-striped')
    row = table.find_all('tr')
    category = soup.find('ul', class_='breadcrumb').find_all('li')[-2].text.strip()

    if category in categories:
        category_dict['UPC'] = row[0].find('td').text
        category_dict['Product Type'] = row[1].find('td').text
        category_dict['Price (excl. tax)'] = row[2].find('td').text
        category_dict['Price (incl. tax)'] = row[3].find('td').text
        category_dict['Tax'] = row[4].find('td').text
        category_dict['Availability'] = row[5].find('td').text
        category_dict['Number of reviews'] = row[6].find('td').text
        category_dict['Category'] = category

        '''if category in categories:
            if category in category_dict:
                category_dict[category].append(book)
            else:
                category_dict[category] = [book]
        else:
                category_dict[category] = None'''
        category_dict['Rating'] = soup.find('div',class_='col-sm-6 product_main').find('p',class_='star-rating')['class'][1]
        descripion = soup.find('article', class_ = 'product_page').find_all('p')[3].text
        category_dict['Description'] = descripion
        category_dict ['Title'] = soup.find('div',class_='col-sm-6 product_main').find('h1').text
    else:
        return None
    return category_dict

def download_page(i):
    url = f'https://books.toscrape.com/catalogue/page-{i}.html'
    request = requests.get(url)
    return request.text
def scrape_books(k, categories):
    
    dfs = []
    for i in range(1,k+1):
        page = download_page(i)
        
        valid_book = extract_book_links(page)
        for book_url in valid_book:
            book_url = f'http://books.toscrape.com/catalogue/{book_url}'
            request = requests.get(book_url).text
            book = get_product_info(request,categories)
            if book:
                dfs.append(book)

    df = pd.DataFrame(dfs)
    return df

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):

    format_url =f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey=9y7uI2yTidBwy6fDtxOy4Gb6H2Z41Zi6'
    family_tree = requests.get(format_url).json()
    data = [i for i in family_tree['historical'] if i['date'][:7] == f'{year}-{month:02}']
    
    df = pd.DataFrame(data)
    

    return df
#Estimated Total Transaction Volume (in dollars)} = {Volume (number of shares traded)} x {Average Price} $$
def stock_stats(history):
    history = history.sort_values(by='date').reset_index(drop=True)
    open_price = history.loc[0,'open']
    close_price = history.loc[len(history)-1,'close']

    percent_change = round((open_price-close_price)/open_price*100 ,2)
    
    total_volume=0
    # Total Transaction Volume  = {Volume (number of shares traded)} x {Average Price} 
    for i in range(history.shape[0]):
        open = history.loc[i,'open']
        close = history.loc[i,'close']
        volume  = (open - close ) /2 * history.loc[i,'volume'] / 100000000
        total_volume += volume
    round_total_volume = str(round(total_volume,2))+' B'
    if percent_change>=0:
        return ('+'+str(percent_change)+'%',round_total_volume)
    else:
        return (str(percent_change)+'%',round_total_volume)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------
visited = list()
next = list()
def create_dict(node):
    node_dict = {'id':[node['id']],\
                    'by':[node.get('by','')],\
                    'text':[node.get('title','')],\
                    'parent': [node.get('parent')],\
                    'time': [pd.Timestamp.fromtimestamp(node.get('time',0))]}
    return node_dict
#18344932

def read_comment(storyid):
    global visited 
    global next 
    node_dict = {}
    format_url = lambda code: f'https://hacker-news.firebaseio.com/v0/item/{code}.json'
    node = requests.get(format_url(storyid)).json()

    if 'dead' not in node.keys():
        if len(next) !=0:
            node_dict = {'id':[node['id']],'by':[node.get('by','')],'text':[node.get('text','')],\
                    'parent': [node.get('parent')],'time': [pd.Timestamp.fromtimestamp(node.get('time',0))]}
            visited =  [node['id']] + visited
            next.pop(0)
            if 'kids'  in node.keys():
                next[:0] = node['kids']
                df = pd.DataFrame(node_dict)
                return df.append(get_comments(next[0]))
            else:
                if len(next) == 0:
                    return pd.DataFrame(node_dict)
                else:
                    df = pd.DataFrame(node_dict)
                    return df.append(get_comments(next[0]))
                
        else:
            visited =   [node['id']] + visited

            if 'kids'   in node.keys():
                df = pd.DataFrame()
                next[:0] = node['kids']
                return df.append(get_comments(next[0]))
                
            else:
                node_dict = create_dict(storyid)
                return pd.DataFrame(node_dict)
            
               
    elif  node['id'] in visited or 'dead' in node.keys():
        next.pop(0)
        if len(next) == 0:
            df = pd.DataFrame()
            return df
        else:
            df = pd.DataFrame()
            return df.append(get_comments(next[0]))
    
def get_comments(storyid):
    lst = read_comment(storyid)
    output = lst.reset_index(drop=True)
    return output
