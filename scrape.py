# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# test
#URL = "https://www.indeed.com/jobs?q=data+scientist&l=New+York&start=10"
#page = requests.get(URL)
#soup = BeautifulSoup(page.text, "html.parser")
#print(soup.prettify())

# Execution
max_results = 100
title_set = ["data+scientist", "data+analyst", "product+analyst"]
city_set = ["San+Francisco", "Seattle", "Los+Angeles", "New+York", "Boston"]
columns = ["city", "company_name", "category", "job_title", "qualified", "tech_compatible", "major_compatible", "app_link", "indeed_link"]
df = pd.DataFrame(columns = columns)

for title in title_set:
	for city in city_set:
		for start in range(0, max_results, 10):
			jobs, postings, app_links, qualified, plang, majors, companies = [], [], [], [], [], [], []
			page = requests.get("https://www.indeed.com/jobs?as_and=" + title + 
					   "&as_phr=&as_any=bachelor%2C+BA&as_not=&as_ttl=&as_cmp=&jt=all&st=&as_src=&salary=&radius=50&l="+ city + 
					   "&fromage=any&limit=10&sort=&psf=advsrch&start=" + str(start))
			time.sleep(5)
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
					jobs.append("".join(a.text.strip()))
					postings.append("https://indeed.com" + a["href"])

					# Job listing detail page
					job_page = requests.get("https://indeed.com" + a["href"])
					time.sleep(5)
					job_soup = BeautifulSoup(job_page.text, "html.parser")

					# Application link
					application_links = job_soup.find_all(name = "a", attrs = {"class":"icl-Button icl-Button--primary icl-Button--md"})
					application_links.sort(key = lambda x: x["href"], reverse = True)
					if application_links:
						for link in application_links:
							if link["href"].find("promo/resume") == -1:
								app_links.append(link["href"])
								break
							else:
								app_links.append("Unknown")
					else:
						app_links.append("Unknown")

					# Whether within 2 YOE and Bachlor
					data = str(job_soup.findAll(text = True)).lower()
					if (re.search("(bachelor|ba)[^a-z0-9]", data) is not None or re.search("(master|ma|phd)[^a-z0-9]", data) is None):
						if re.search("[1-2].{1,10}(Y|y)ear", data) is not None:
							qualified.append("True")
						else:
							qualified.append("Need YOE")
					elif re.search("[1-2].{1,10}(Y|y)ear", data) is not None:
						qualified.append("Need Advanced Degree")
					else:
						qualified.append("False")

					# Compatible technologies & major
					plang.append("True") if re.search("(r|sql|python)[^a-z0-9]", data) is not None else plang.append("False")
					majors.append("True") if re.search("(economics)[^a-z0-9]", data) is not None else majors.append("False")

			listings = [("job_title", jobs),
						("company_name", companies),
						("indeed_link", postings), 
						("app_link", app_links), 
						("qualified", qualified), 
						("tech_compatible", plang), 
						("major_compatible", majors)]
			temp_df = pd.DataFrame.from_items(listings)
			temp_df = temp_df.assign(city = str(city))
			temp_df = temp_df.assign(category = str(title))
			df = df.append(temp_df, ignore_index = True)

df = df[["city", "company_name", "category", "job_title", "qualified", "tech_compatible", "major_compatible", "app_link", "indeed_link"]]
df = df.sort_values(by = ["qualified", "tech_compatible", "major_compatible"], ascending = [0, 0, 0])
df.to_csv("H:/git_repo/job_scraping/output.csv", encoding="utf-8", index=False)