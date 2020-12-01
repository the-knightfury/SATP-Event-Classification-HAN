import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
countries = ["bangladesh", "india", "pakistan", "bhutan", "nepal", "maldives", "afghanistan", "srilanka"]
years = ["2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011", "2010"]


# Saving News Content and The Country wise location in these 2 lists
news_contents = []
news_location = []

for i in years:
    print(i)
    # Iterating over each country
    for k in countries:
        # Iterating over each month
        print(k)
        for j in months:
            print(j)

            url = "https://www.satp.org/terrorist-activity/"+ k + "-" + j + "-" + i

            # Request
            r1 = requests.get(url)

            # We'll save in coverpage the cover page content
            coverpage = r1.content

            # Soup creation
            soup1 = BeautifulSoup(coverpage, 'html.parser')

            # News identification
            coverpage_news = soup1.find_all('div', class_='more')


            news_size = len(coverpage_news)

            for paragraph_num in range(news_size):
                paragraph = coverpage_news[paragraph_num].get_text()
                paragraph = re.sub("^\s+|\s+$", "", paragraph, flags=re.UNICODE)
                paragraph = " ".join(re.split("\s+", paragraph, flags=re.UNICODE)[:-2])
                news_location.append(k)
                news_contents.append(paragraph)


        print(k," ",len(news_contents))


news_data = pd.DataFrame({'location': news_location,
                          'news': news_contents})

print(news_data.shape)

# news_data.to_csv ('./Satp_Data/news_satp_2010.csv', index = False, header=True)





