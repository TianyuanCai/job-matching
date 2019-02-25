# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# test
URL = "https://www.indeed.com/jobs?q=data+scientist&l=New+York&start=10"
page = requests.get(URL)
soup = BeautifulSoup(page.text, "html.parser")
print(soup.prettify())

# Execution
max_results_per_city = 10
title_set = ["data+scientist", "data+analyst", "product+analyst"]
city_set = ['San+Francisco', 'Seattle', 'Portland', 'Los+Angeles', 'New+York','Chicago', 'Pittsburgh', 'Austin']
# city_set = ['San+Francisco']
columns = ["city", "company_name", "category", "job_title", "qualified", "app_link", "indeed_link"]
df = pd.DataFrame(columns = columns)

for title in title_set:
	for city in city_set:
		for start in range(0, max_results_per_city, 10):
			jobs = []
			postings = []
			app_links = []
			qualified = []
			companies = []

			page = requests.get('https://www.indeed.com/jobs?q='+title + '&l=' + str(city) + '&start=' + str(start))
			soup = BeautifulSoup(page.text, "html.parser")

			for div in soup.find_all(name="div", attrs={"class":"row"}):
				# Company names
				company = div.find_all(name="span", attrs={"class":"company"})
				if len(company) > 0:
					companies.append(company[0].text.strip())
				else:
					sec_try = div.find_all(name="span", attrs={"class":"result-link-source"})
					for span in sec_try:
						companies.append(span.text.strip())

				for a in div.find_all(name="a", attrs={"data-tn-element":"jobTitle"}):
					# Jobs titles
					jobs.append(''.join(a.text.strip()))
					postings.append('http://indeed.com' + a['href'])

					# Job listing detail page
					job_page = requests.get('https://indeed.com' + a['href'])
					job_soup = BeautifulSoup(job_page.text, "html.parser")

					# Application link
					application_links = job_soup.find_all(name = "a", attrs = {"class":"icl-Button icl-Button--primary icl-Button--md"})
					application_links.sort(key = lambda x: x['href'], reverse = True)
					if application_links:
						for link in application_links:
							if link['href'].find('promo/resume') == -1:
								app_links.append(link['href'])
								break
							else:
								app_links.append("Unknown")
					else:
						app_links.append("Unknown")

					# Whether within 2 YOE and Bachlor
					data = str(job_soup.findAll(text = True))
					if re.search('[1-2].{1,10}(Y|y)ear', data) is not None:
						if (re.search('((B|b)achelor)|BA',data) is not None or re.search('((M|m)aster)|MA|(PhD|phd|PHD)', data) is None):
							qualified.append("True")
						else:
							qualified.append("Need Advanced Degrees")
					elif (re.search('((B|b)achelor)|BA',data) is not None or re.search('((M|m)aster)|MA|(PhD|phd|PHD)', data) is None):
						qualified.append("Need YOE")
					else:
						qualified.append("False")

			listings = [("job_title", jobs),
						("company_name", companies),
						("indeed_link", postings), 
						("app_link", app_links), 
						("qualified", qualified)]
			temp_df = pd.DataFrame.from_items(listings)
			temp_df = temp_df.assign(city = str(city))
			temp_df = temp_df.assign(category = str(title))
			df = df.append(temp_df, ignore_index = True)
			time.sleep(1)

df = df[["city", "company_name", "category", "job_title", "qualified", "app_link", "indeed_link"]]

df.to_csv("H:/git_repo/indeed_scraping/results.csv", encoding='utf-8')
