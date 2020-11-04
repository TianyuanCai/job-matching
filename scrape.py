# -*- coding: utf-8 -*-
"""
We collect job description information from indeed.com
Based on the user rating of compatibility and interest on each job description
We personalize the recommendation of the jobs over time

Application: this can be integrated into our current flask web app

Oct 29, 2020
Haowei Liu, Tianyuan Cai
"""

import itertools
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.nlp import max_sum_sim, mmr

# Execution
title_set = ['data+analyst']
city_test = ['San+Francisco']
city_set = ['San+Francisco', 'Seattle', 'Los+Angeles', 'New+York', 'Boston']
columns = ['city', 'company_name', 'category', 'job_title', 'qualified',
           'tech_compatible', 'major_compatible', 'app_link', 'indeed_link']

start_range = range(0, 50, 10)

# keyword extraction config
n_gram_range = (1, 2)


def format_job_link(title, city, start):

    link = f'https://www.indeed.com/jobs?as_and=' \
           f'{title}&as_not=&as_ttl=&as_cmp=&jt' \
           f'=all&st=&as_src=&salary=&radius=50&l=' \
           f'{city}&fromage=any&limit=10&sort=&psf=advsrch&start={str(start)}'
    return link


def clean_str(text):
    text = re.sub(r'\n|[0-9]{1,}|-|_', ' ', text)
    text = re.sub(r'([A-Z][a-z])', r' \1', text)
    text = re.sub(r'\s{1,}', ' ', text)

    return text

df = pd.DataFrame(columns=columns)

for title, city, start in tqdm(itertools.product(title_set, city_set,
                                                 start_range)):
    jobs, postings, app_links, qualified, lang, majors, companies, \
    descriptions = [], [], [], [], [], [], [], []

    page = requests.get(format_job_link(title, city, start))
    time.sleep(5)
    soup = BeautifulSoup(page.text, 'html.parser')

    for div in soup.find_all(name='div', attrs={'class': 'row'}):
        # Company names
        company = div.find_all(name='span', attrs={'class': 'company'})
        if len(company) > 0:
            companies.append(company[0].text.strip())
        else:
            sec_try = div.find_all(name='span', attrs={
                'class': 'result-link-source'})
            for span in sec_try:
                companies.append(span.text.strip())

        # here we go through each job listing
        for a in div.find_all(name='a', attrs={'data-tn-element': 'jobTitle'}):

            job_posting = 'https://indeed.com' + a['href']
            postings.append(job_posting)

            # Job listing detail page
            job_page = requests.get(job_posting)
            time.sleep(5)

            job_soup = BeautifulSoup(job_page.text, 'html.parser')

            # Skip error page
            if job_soup.find(name='title') and job_soup.find(name='title').text == 'Error 404 Not Found':
                jobs.append('invalid URL')
                app_links.append('invalid URL')
                qualified.append('invalid URL')
                lang.append('invalid URL')
                majors.append('invalid URL')
                descriptions.append('invalid URL')
                break

            # Jobs titles
            jobs.append(''.join(a.text.strip()))

            # get the entire job description
            desc = job_soup.find(name='div', attrs={'id': 'jobDescriptionText'})
            if desc:
                job_description = desc.text
            else:
                job_description = "Missing job description"
                print("Missing job description")
            descriptions.append(clean_str(job_description))

            # Application link
            application_links = job_soup.find_all(name='a', attrs={
                'class': 'icl-Button icl-Button--primary ' 'icl-Button--md'})
            application_links.sort(key=lambda x: x['href'], reverse=True)

            if application_links:
                for link in application_links:
                    if link['href'].find('promo/resume') == -1:
                        app_links.append(link['href'])
                        break
                    else:
                        app_links.append('https://indeed.com' + a['href'])
            else:
                app_links.append('https://indeed.com' + a['href'])

            # Whether within 2 YOE and Bachelor
            # TODO: YOE can be customized
            data = str(job_soup.findAll(text=True)).lower()
            if (re.search('(bachelor|ba)[^a-z0-9]',
                          data) is not None or re.search(
                '(master|ms|ma|phd)[^a-z0-9]', data) is None):
                if re.search('[1-2].{1,10}(Y|y)ear', data) is not None:
                    qualified.append('Bachelor Degree')
                else:
                    qualified.append('Need YOE')
            elif re.search('[1-2].{1,10}(Y|y)ear', data) is not None:
                qualified.append('Need Advanced Degree')
            else:
                qualified.append('Need More Than 2 YOE')

            # Compatible language
            if re.search('(r|sql|python)[^a-z0-9]', data) is not None:
                lang.append('True')
            else:
                lang.append('False')

            # TODO: identify a list of majors
            if re.search('(economics)[^a-z0-9]', data) is not None:
                majors.append('True')
            else:
                majors.append('False')

            # TODO: US work authorization? can be extracted from description

    listings = [('job_title', jobs),
                ('company_name', companies),
                ('indeed_link', postings),
                ('app_link', app_links),
                ('qualified', qualified),
                ('tech_compatible', lang),
                ('major_compatible', majors),
                ('description', descriptions)]
    d_listings = dict(listings)
    temp_df = pd.DataFrame.from_dict(dict(listings))
    temp_df = temp_df.assign(city=str(city))
    temp_df = temp_df.assign(category=str(title))
    df = df.append(temp_df, ignore_index=True)

# TODO: whether to include unique application link
df = df.drop_duplicates(
    subset=['city', 'company_name', 'category', 'job_title', 'qualified',
            'tech_compatible', 'major_compatible'])
df = df.sort_values(
    by=['qualified', 'tech_compatible', 'major_compatible', 'city',
        'company_name', 'category'], ascending=[0, 0, 0, 0, 0, 1])
df.to_csv('da_scraped_lists.csv', encoding='utf-8', index=False)

