import streamlit as st
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from wordcloud import WordCloud, STOPWORDS
import base64
from sklearn.feature_extraction.text import CountVectorizer

# Importar dataset
url_dataset = 'https://github.com/soilmo/meliuz/blob/main/meliuz.xlsx?raw=true'
@st.cache(show_spinner=False)
def importar_base(url):
    df = pd.read_excel(url, engine='openpyxl')
    return df

# Funções para word cloud -------------------------------------
def freq_onegrams(text):
    
    WNL = nltk.WordNetLemmatizer()
    # Lowercase and tokenize
    text = text.lower()
    # Remove single quote early since it causes problems with the tokenizer.
    text = text.replace("'", "")
    # Remove numbers from text
    digits = '0123456789'
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)

    #set the stopwords list
    stopwords_wc = set(STOPWORDS)

    # If you want to remove any particular word form text which does not contribute much in meaning
    customised_words_bi = [',',';','a','o','O','as','os','e','para','por','?','!','Não','nao','Nao','não','E','.','-','/',
                        '..','...','<','>','(',')',':','&','$','%','§','pra', ' ','a','b','c','d','e','f','g','h','i','j',
                        'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ano','hoje','ontem','yoy','"','the','and','to','is','are','on','in','it','of','ir','group','ex',
                        '*','"','dia','“','”','etc','eh','fm',
                        'ja','já','R$','r$','+','-','that','per','mt','with','by','pq','cent',
                        'br','us','hj','dp','mto','=','share','volumar','mm','1x1','sp','en-export',
                        'port','n-export','iv','rgb','width','ha','dele','dela','desse','outro','da','de','do',
                       'das','dos','deles','ela','ele','eles','elas','um','uma']

    new_stopwords = stopwords_wc.union(customised_words_bi)

    #text_content = [word for word in text_content if word not in new_stopwords]
    text_content = [word for word in text1 if word not in new_stopwords]

    # After the punctuation above is removed it still leaves empty entries in the list.
    text_content = [s for s in text_content if len(s) != 0]

    # Best to get the lemmas of each word to reduce the number of similar words
    text_content = [WNL.lemmatize(t) for t in text_content]

    #Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    bag_of_words = vectorizer.fit_transform(text_content)
    #vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq, new_stopwords

def freq_bigrams(text):
    
    WNL = nltk.WordNetLemmatizer()
    # Lowercase and tokenize
    text = text.lower()
    # Remove single quote early since it causes problems with the tokenizer.
    text = text.replace("'", "")
    # Remove numbers from text
    digits = '0123456789'
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)

    #set the stopwords list
    stopwords_wc = set(STOPWORDS)

    # If you want to remove any particular word form text which does not contribute much in meaning
    customised_words_bi = [',',';','a','o','O','as','os','e','para','por','?','!','Não','nao','Nao','não','E','.','-','/',
                        '..','...','<','>','(',')',':','&','$','%','§','pra', ' ','a','b','c','d','e','f','g','h','i','j',
                        'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ano','hoje','ontem','yoy','"','the','and','to','is','are','on','in','it','of','ir','group','ex',
                        '*','"','dia','“','”','etc','eh','fm',
                        'ja','já','R$','r$','+','-','that','per','mt','with','by','pq','cent',
                        'br','us','hj','dp','mto','=','share','volumar','mm','1x1','sp','en-export',
                        'port','n-export','iv','rgb','width','ha','dele','dela','desse','outro','da','de','do',
                       'das','dos','deles','ela','ele','eles','elas','um','uma']

    new_stopwords = stopwords_wc.union(customised_words_bi)

    #text_content = [word for word in text_content if word not in new_stopwords]
    text_content = [word for word in text1 if word not in new_stopwords]

    # After the punctuation above is removed it still leaves empty entries in the list.
    text_content = [s for s in text_content if len(s) != 0]

    # Best to get the lemmas of each word to reduce the number of similar words
    text_content = [WNL.lemmatize(t) for t in text_content]

    nltk_tokens = nltk.word_tokenize(text)  
    bigrams_list = list(nltk.bigrams(text_content))
    dictionary2 = [' '.join(tup) for tup in bigrams_list]

    #Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    #vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq, new_stopwords

def freq_trigrams(text):
    
    WNL = nltk.WordNetLemmatizer()
    # Lowercase and tokenize
    text = text.lower()
    # Remove single quote early since it causes problems with the tokenizer.
    text = text.replace("'", "")
    # Remove numbers from text
    digits = '0123456789'
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)

    #set the stopwords list
    stopwords_wc = set(STOPWORDS)

    # If you want to remove any particular word form text which does not contribute much in meaning
    customised_words_bi = [',',';','a','o','O','as','os','e','para','por','?','!','Não','nao','Nao','não','E','.','-','/',
                        '..','...','<','>','(',')',':','&','$','%','§','pra', ' ','a','b','c','d','e','f','g','h','i','j',
                        'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ano','hoje','ontem','yoy','"','the','and','to','is','are','on','in','it','of','ir','group','ex',
                        '*','"','dia','“','”','etc','eh','fm',
                        'ja','já','R$','r$','+','-','that','per','mt','with','by','pq','cent',
                        'br','us','hj','dp','mto','=','share','volumar','mm','1x1','sp','en-export',
                        'port','n-export','iv','rgb','width','ha','dele','dela','desse','outro','da','de','do',
                       'das','dos','deles','ela','ele','eles','elas','um','uma']

    new_stopwords = stopwords_wc.union(customised_words_bi)

    #text_content = [word for word in text_content if word not in new_stopwords]
    text_content = [word for word in text1 if word not in new_stopwords]

    # After the punctuation above is removed it still leaves empty entries in the list.
    text_content = [s for s in text_content if len(s) != 0]

    # Best to get the lemmas of each word to reduce the number of similar words
    text_content = [WNL.lemmatize(t) for t in text_content]

    nltk_tokens = nltk.word_tokenize(text)  
    bigrams_list = list(nltk.trigrams(text_content))
    dictionary2 = [' '.join(tup) for tup in bigrams_list]

    #Using count vectoriser to view the frequency of bigrams
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(ngram_range=(3, 3))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    #vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq, new_stopwords


# Criar link para download
@st.cache(persist=True, max_entries = 20, ttl = 1800, show_spinner=False)
def get_table_download_link(df, arquivo):
    
    csvfile = df.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = arquivo +".csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download da tabela de frequência de palavras</a>'

    return href

# Title
st.title("Análises Meliuz - Glass Door")
# Período de análise
st.header("Período de análise")

# Importar base
df = importar_base(url_dataset)
df = df.sort_values(by=['data'])
st.warning("Data mais antiga: " + str(df['data'].iloc[0]))

dt_i = st.date_input("Qual o dia inicial?", datetime.datetime.now())
dt_i = dt_i.strftime('%Y-%m-%d')

dt_f = st.date_input("Qual o dia final?", datetime.datetime.now())
dt_f = dt_f.strftime('%Y-%m-%d')

# Filtrar
filtro_1 = df['data']>=dt_i
filtro_2 = df['data']<=dt_f
df_dt = df[(filtro_1) & (filtro_2)]

st.success("Base importada. Total de " + str(df_dt.shape[0])+ " avaliações nesse período")

dict_categorias = {
    'Título da Avaliação':'titulo',
    'Prós':'pros',
    'Contras':'contras',
    'Conselhos a Presidência':'conselho'
}

# Textual
st.header("Análise textual")
# Escolher segmento
categoria = st.selectbox("Qual categoria quer olhar?", options = ['Título da Avaliação','Prós','Contras','Conselhos a Presidência'])

df_filtrado =df_dt

if df_filtrado.shape[0]>0:
    
    with st.beta_expander("Montar análises de um termos"):
    
        texto = ''
        for i in df[dict_categorias[categoria]]:
            texto = texto + " " + str(i)
        # Tokens
        words_freq, new_stopwords = freq_onegrams(texto)
            
        # Mapa de palavras ---------
        #if st.checkbox("Mapa de palavras"):
        st.markdown("Mapa de palavras")
             
        # Generate word cloud
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 200
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=new_stopwords).generate_from_frequencies(words_dict)
        st.image(wordCloud.to_array())

        #if st.checkbox("Frequência de palavras"):
        st.markdown("Frequência de palavras")
        df_freq = pd.DataFrame({
            'termo':'',
            'qtd':''
        }, index = [0])

        for i in words_freq:
            aux = {
                'termo':i[0],
                'qtd':i[1]
            }
            df_freq = df_freq.append(aux, ignore_index = True)

        df_freq = df_freq.iloc[1:,]
        arquivo = 'freq_termos_um_' + categoria
        url_base = get_table_download_link(df_freq, arquivo)
        st.markdown(url_base, unsafe_allow_html=True)
        st.write(df_freq.set_index(['termo']))



    with st.beta_expander("Montar análises de dois termos"):
    
        texto = ''
        for i in df[dict_categorias[categoria]]:
            texto = texto + " " + str(i)
        # Tokens
        words_freq, new_stopwords = freq_bigrams(texto)
            
        # Mapa de palavras ---------
        #if st.checkbox("Mapa de palavras"):
        st.markdown("Mapa de palavras")
             
        # Generate word cloud
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 200
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=new_stopwords).generate_from_frequencies(words_dict)
        st.image(wordCloud.to_array())

        #if st.checkbox("Frequência de palavras"):
        st.markdown("Frequência de palavras")
        df_freq = pd.DataFrame({
            'termo':'',
            'qtd':''
        }, index = [0])

        for i in words_freq:
            aux = {
                'termo':i[0],
                'qtd':i[1]
            }
            df_freq = df_freq.append(aux, ignore_index = True)

        df_freq = df_freq.iloc[1:,]
        arquivo = 'freq_termos_bi_' + categoria
        url_base = get_table_download_link(df_freq, arquivo)
        st.markdown(url_base, unsafe_allow_html=True)
        st.write(df_freq.set_index(['termo']))

    with st.beta_expander("Montar análises de três termos"):
    
        texto = ''
        for i in df[dict_categorias[categoria]]:
            texto = texto + " " + str(i)
        # Tokens
        words_freq, new_stopwords = freq_trigrams(texto)
            
        # Mapa de palavras ---------
        st.markdown("Mapa de palavras")
             
        # Generate word cloud
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 200
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=new_stopwords).generate_from_frequencies(words_dict)
        st.image(wordCloud.to_array())

        st.markdown("Frequência de palavras")
        df_freq = pd.DataFrame({
            'termo':'',
            'qtd':''
        }, index = [0])

        for i in words_freq:
            aux = {
                'termo':i[0],
                'qtd':i[1]
            }
            df_freq = df_freq.append(aux, ignore_index = True)

        df_freq = df_freq.iloc[1:,]
        arquivo = 'freq_termos_tri_' + categoria
        url_base = get_table_download_link(df_freq, arquivo)
        st.markdown(url_base, unsafe_allow_html=True)
        st.write(df_freq.set_index(['termo']))
        
