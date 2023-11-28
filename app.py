import cleantext
import pandas as pd
import streamlit as st
import nltk
nltk.download('punkt')
#from gensim.summarization.summarizer import summarize
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from textblob import TextBlob

st.title("NLP- D590 Final Project - Fall 2023")
st.header('NLP App with Streamlit,TextBlob,Gensim Using Python ')
st.subheader("Sentiment Analysis, Summarization, Clean Text")
st.subheader("Sentiment Analysis")


def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result



with st.expander('Enter text for Sentiment Analysis'):
    #st.subheader("Sentiment Analysis")
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))
st.subheader("Text Preprocessing")
with st.expander('Enter text for Preprocessing'):

    pre = st.text_input('Clean Text:')

    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True, stemming=True
                                 ))
st.subheader("Summarize Text")
with st.expander('Enter text to Summarize'):

    message = st.text_area('Enter Content Here:')
    summary_options = st.selectbox("Choice of your summarizer", ("gensim", "sumy"))
    if st.button("Summarize"):
        if summary_options == 'gensim':
            st.text("Using Gensim...")
            summary_result = sumy_summarizer(message)
        elif summary_options == 'sumy':
            st.text("Using Sumy..")
            summary_result = sumy_summarizer(message)
        else:
            st.warning("Using default summarizer")
            st.text("Using Gensim")
            summary_result = sumy_summarizer(message)

        st.success(summary_result)


st.subheader("Upload CSV for Sentiment Analysis on Amazon reviews")
with st.expander("Upload Amazon review Data set to do Sentiment Analysis"):


    upl = st.file_uploader('Upload your file to do Sentiment Analysis')

    def score(x):
        blob1 = TextBlob(str(x))
        return blob1.sentiment.polarity


    def analyze(x):

        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:

        df = pd.read_csv(upl)
        df['score'] = df['reviewText'].apply(score)
        #df['sentiment_calc'] = df['review'].apply(sentiment_calc)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download the sentiment analysis results as CSV",
            data=csv,
            file_name='Sentiment_Analysis.csv',
            mime='text/csv',
        )