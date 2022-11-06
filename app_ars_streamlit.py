import requests
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm
import base64
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """



def main():
    html_temp = """
	<div style="background-color:grey;"><p style="color:black;font-size:40px;padding:9px"><b>Live Twitter Sentiment Analysis<b></p></div>
	"""
    st.markdown(html_temp,unsafe_allow_html=True)

    html_temp1 = """
	<div><p style="color:Blue;font-size:20px;padding:9px"><b>Choose a Topic that you want to get Sentiment Analysis on<b></p></div>
	"""
    st.markdown(html_temp1,unsafe_allow_html=True)
    #st.subheader('<p style="font-family:sans-serif; color:Green; font-size: 42px;">Choose a Topic that you want to get Sentiment Analysis on</p>')
     # Use the below credentials to authenticate the API.

    auth = tweepy.OAuthHandler("Enter the OuthHandler Code", "Enter the OuthHandler Code")
    auth.set_access_token("Enter Access Token","Enter Access Token")
    api = tweepy.API(auth)
    ################################################################
    df =pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT","User_Location"])
    df1 =pd.DataFrame(columns=["User","Tweet"])

     # Write a Function to extract tweets:
    def get_tweets(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
            #time.sleep(0.1)
            #my_bar.progress(i)
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_Location"] = tweet.user.location
            

            #csv=df.to_csv("TweetDataset.csv",index=False)
            #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            #href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass

    
       

    def get_tweets1(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
            #time.sleep(0.1)
            #my_bar.progress(i)
            
            df1.loc[i,"User"] = tweet.user.name
            df1.loc[i,"Tweet"] = tweet.text
            #df.to_csv("TweetDataset.csv",index=False)
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass

    #function to clean tweet
    lemmatizer = WordNetLemmatizer()
    def clean_tweet(tweet):
        punc= ' '.join(re.sub('(@[a-zA-Z0-9]+)|([^0-9a-zA-Z \t])|(\w+:\/\/\S+)|(RT)', ' ',tweet.lower()).split())
        # Now just remove any stopwords and lemmatize the words
        return ' '.join([lemmatizer.lemmatize(word) for word in punc.split() if word.lower() not in STOPWORDS])

    #function to assign polarity
    def analyze_sentiment(tweet):
        analysis=TextBlob(tweet)
        if analysis.sentiment.polarity >0 :
            return"Positive"
        elif analysis.sentiment.polarity ==0:
            return "Neutral"
        else :
            return "Negative"     

    from PIL import Image

    

    #Function to Pre-process data for Worlcloud
    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordCloud 
        ###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new

    #image =Image.open('twitter-gifs.gif')
    #st.image(image,caption='Twitter for Analytics',use_column_width=True)   
    #st.image(
           # "twitter-gifs.gif", # I prefer to load the GIFs using GIPHY
            #width=400, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
        #)
   
    
    file_ = open("twitter-gifs.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
    )

    
    #collect data from user
    Topic = str()
    Topic=str(st.text_input("Enter the Topic you want (Press Enter once done)"))
    
    if len(Topic)>0:
        file_ = open("RD07.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
        )
        # call the function to get the data
        with st.spinner('Please wait while the tweets are being executed'):
            get_tweets(Topic,Count=199)
        st.success('Tweets have been extracted!')
    
        # call function to get clean tweets
        df['clean_tweet']=df['Tweet'].apply(lambda x:clean_tweet(x))

        # call function to get sentiments
        df['Sentiment']=df['Tweet'].apply(lambda x:analyze_sentiment(x))  
        

        option = st.sidebar.radio(
        'Navigate through various features of the app!',
        ('See the extracted data.','Count Plot for Tweet Sentiment','Count Plot for User Verification','Piechart for Sentiment','WordCloud for Positive Tweets'
        ,'WordCloud for Negative Tweets',)
        )  
        if option==('See the extracted data.'):
            with st.spinner('Please wait for  the data to be displayed'):
                st.success('The extracted data:')

        # write summary of the tweets
        st.write("Total Tweets extracted for the topic '{}' are :{}".format(Topic,len(df.Tweet)))
        st.write("Total Positive Tweets are:{} ".format(len(df[df["Sentiment"]=="Positive"])))
        st.write("The total number of Negative tweets:{}".format(len(df[df['Sentiment']=='Negative'])))
        st.write("The number of Neutral tweets are:{}".format(len(df[df['Sentiment']=='Neutral'])))
        st.write(df.head(50))
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv.)'
        st.write('The file contains the raw data , applied NLP Techniques, clean data and sentiments')
        st.markdown(href, unsafe_allow_html=True)

      
        # see the extracted data
        
            

        # get the count plot
        if option==('Count Plot for Tweet Sentiment'):
            st.write('Countplot is being generated')
            st.write(sns.countplot(df['Sentiment']))
            st.pyplot()

        # get piechart
        if option==('Piechart for Sentiment'):
            st.success('Generating a Pie Chart')
            a=len(df[df['Sentiment']=='Positive'])
            b=len(df[df['Sentiment']=='Negative'])
            c=len(df[df['Sentiment']=='Neutral'])
            d=np.array([a,b,c])
            explode = (0.1,0.0,0.1)
            st.write(plt.pie(d,shadow=True,explode=explode,labels=['Positive','Negative','Neutral'],autopct='%1.2f%%'))
            st.pyplot()

        if option==("Count Plot for User Verification"):
            st.success("Generating A Count Plot (Verified and unverified Users)")
            st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
            st.write(sns.countplot(df["Sentiment"],hue=df.IsVerified))
            st.pyplot()

        if option==('WordCloud for Positive Tweets'):
                st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
                stopwords = set(STOPWORDS)
                text_positive = ' '.join(text for text in  df['clean_tweet'][df['Sentiment']=='Positive'])
                all_words_positive = prepCloud(text_positive,Topic)
                # combining the image with the dataset
                Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

                # We use the ImageColorGenerator library from Wordcloud 
                # Here we take the color of the image and impose it over our wordcloud
                image_colors = ImageColorGenerator(Mask)

                # Now we use the WordCloud function from the wordcloud library 
                wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_positive)

                plt.figure(figsize=(10,20))

                # Here we recolor the words from the dataset to the image's color
                # recolor just recolors the default colors to the image's blue color
                # interpolation is used to smooth the image generated 
                st.write(plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming"))

                st.pyplot()

        if option==('WordCloud for Negative Tweets'):
                st.success("Generating A WordCloud for all Negative Tweets about {}".format(Topic))
                stopwords = set(STOPWORDS)
                text_negative = ' '.join(text for text in  df['clean_tweet'][df['Sentiment']=='Negative'])
                all_words_negative = prepCloud(text_negative,Topic)
                wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(all_words_negative)
                # combining the image with the dataset
                Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

                # We use the ImageColorGenerator library from Wordcloud 
                # Here we take the color of the image and impose it over our wordcloud
                image_colors = ImageColorGenerator(Mask)

                # Now we use the WordCloud function from the wordcloud library 
                wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                plt.figure(figsize=(10,20))

                # Here we recolor the words from the dataset to the image's color
                # recolor just recolors the default colors to the image's blue color
                # interpolation is used to smooth the image generated 
                st.write(plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming"))

                st.pyplot()

    st.sidebar.header("App Details")
    st.sidebar.info('The App will scrape live tweets with the help of Tweepy API and create a dataframe.')    
    st.sidebar.info('Analysis of the raw data and visualisations are provided to extract meaning')    
    st.sidebar.text("App built with Twitter API and Streamlit")
    



    if st.button("Exit"):
        st.balloons()

        


if __name__ == '__main__':
    main()

  
