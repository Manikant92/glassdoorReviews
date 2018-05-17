# glassdoorCompanyReviews and Word Cloud for company reviews
Scraping glassdoor company reviews and then reviews using word cloud and nltk.

Scrape any company glassdoor reviews (I scraped Accenture's). To scrape reviews of company which you wish to, just change url variable value with the company url of your choice. glassdoorReviewsScrapper.py is the script which you need to run to scrape the company reviews.

Scraped data will be stored in dataframe and then saves to CSV file in your local project repository.

Load that csv file into reviewsAnalysisWc.py with CSV file name (you don't have to load, just change the csv file name in that script). And just run the script, it does everything for you by creating word cloud images in your local project repository.

I will try to optimize the code fakeAgent package (a package which generates randomized user agents) and test it and push the changes here once done. For now, I used userAgents.py file which generates randomized userAgent headers in order not to get blocked by website (normally, when you use actual real useragent while scraping any website, it may tend to detect you as bot and ask for captch. To avoid, I used random user agents).

Also, will automate the process in the code by taking url input from user and doing the rest process. Will also change automate csv file name, after update it will just take file name you provided while running glassdoorReviewsScrapper.py script.

Currently, you need to run two scripts separately. Will update code changes for all these scripts into main class file, then you can simply run one script, it will do everything for you.

Also, for now, I am taking 100 page reviews for any company. If you want to scrape all pages, then follow the comments provided in glassdoorReviewsScrapper.py file. However, if you follow that to scrape all pages, it will take hours to scrape the data. Accenture glassdoor reviews has 20000+ reviews which is of 2000+ pages. For scraping 100 pages with automating url generators, it took me around 7 mins.

Script produces word clouds for Pros, Cons and Adivce reviews separately. Unigrams and Bigrams word clouds are included. If you wish to play, you can form Trigrams (just update ngrams = 3 in reviewsAnalysisWc.py). If you wish to play with word clouds with different image shapes (refer under reviewAnalysis_JupyterNB folder to sampleAccentureshape.png file), you can follow the comments provided in reviewsAnalysisWc file. I have provided the code for it as well. 

You can play with forming different dataframes like word cloud for former employess and current employees as you wish to. For that you need tweak/separate dataframe into two dataframes and play with it.

Create virtual environment in your local and install requirements.txt file

pip install -r requirements.txt

advwc_bigrams.png - Bigrams word cloud image for top 30 advice reviews.
advwc_unigrams.png - Unigrams word cloud image for top 30 advice reviews.
conwc_bigrams.png - Bigrams word cloud image for top 30 con reviews.
conwc_unigrams.png - Unigrams word cloud image for top 30 con reviews.
poswc_bigrams.png - Bigrams word cloud image for top 30 pros reviews.
poswc_unigrams.png - Unigrams word cloud image for top 30 pros reviews.

I have analysed reviews using Jupyter notebook before building the code for word clouds. Attached the Notebook file under reviewAnalysis_JupyterNB folder. 

Happy Coding!
