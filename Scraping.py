import pandas as pd
import tweepy
consumer_key = "XXXXXXXXX"
consumer_secret = "XXXXXXXXX"
access_token = "XXXXXXXXX"
access_token_secret = "XXXXXXXXX"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

text_query = '2020 US Election'
count = 150
try:
    # Creation of query method using parameters
    tweets = tweepy.Cursor(api.search, q=text_query).items(count)

    # Pulling information from tweets iterable object
    tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]

    # Creation of dataframe from tweets list
    # Add or remove columns as you remove tweet information
    tweets_df = pd.DataFrame(tweets_list)

except BaseException as e:
    print('failed on_status,', str(e))
    time.sleep(3)