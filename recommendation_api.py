from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gensim
from konlpy.tag import Okt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
import json
from gensim.models import Word2Vec
import os

app = FastAPI()

# Load the Word2Vec model and setup environment
# 학습시킨 모델 -(1)
model = Word2Vec.load("E:/Dockggu/word2vec_model_v1")
# Java 환경 설치되어 있어야 함 -(2)
os.environ["JAVA_HOME"] = r'C:/Program Files/Java/jdk-21/bin/server'

# Morphological analyzer and stopwords
okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# Load the DataFrame
# 전처리 완료한 csv 파일 -(3)
df = pd.read_csv('E:/Dockggu/book_data_edit.csv')
df['title_vector'] = df['title_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
df['category_vector'] = df['category_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
df['description_vector'] = df['description_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
df['author_vector'] = df['author_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

class BookRecommendation(BaseModel):
    book_title: str

# Function to recommend books
def recommend_books(book_title, top_n=5):
    if book_title in df['title'].values:
        input_book = df[df['title'] == book_title].iloc[0]
        input_title_vector = input_book['title_vector']
        input_author_vector = input_book['author_vector']
        input_category_vector = input_book['category_vector']
        input_description_vector = input_book['description_vector']
        fetched_author = input_book['author']
    else:
        api_key = "" # 알라딘 API 키 -(4)
        url = f"http://www.aladin.co.kr/ttb/api/ItemSearch.aspx?ttbkey={api_key}&Query={book_title}&QueryType=Title&MaxResults=10&start=1&SearchTarget=Book&output=js&Version=20131101"
        response = requests.get(url)
        response_json = json.loads(response.text)
        if 'item' in response_json and response_json['item']:
            fetched_title = response_json['item'][0]['title']
            fetched_author = response_json['item'][0]['author']
            fetched_description = response_json['item'][0]['description']
            fetched_category = response_json['item'][0]['categoryName']
            
        else:
            raise HTTPException(status_code=404, detail="Book not found")

        input_title_vector = convert_text_to_vector(fetched_title, model, okt, stopwords)
        input_author_vector = convert_text_to_vector(fetched_author, model, okt, stopwords)
        input_description_vector = convert_text_to_vector(fetched_description, model, okt, stopwords)
        input_category_vector = convert_text_to_vector(fetched_category, model, okt, stopwords)

    # Calculate average similarity
    def calculate_average_similarity(row):
        title_similarity = calculate_cosine_similarity(input_title_vector, row['title_vector'])
        author_similarity = calculate_cosine_similarity(input_author_vector, row['author_vector'])
        category_similarity = calculate_cosine_similarity(input_category_vector, row['category_vector'])
        description_similarity = calculate_cosine_similarity(input_description_vector, row['description_vector'])
        return np.mean([title_similarity, author_similarity, category_similarity, description_similarity])

    df['average_similarity'] = df.apply(calculate_average_similarity, axis=1)
    top_5_similar_books = df.sort_values(by='average_similarity', ascending=False).iloc[1:6]

    recommendations = []
    for index, row in top_5_similar_books.iterrows():
        recommendations.append({
            "title": row['title'],
            # 추가
            "author": row['author']  # Add author information
        })

    return recommendations

# Calculate average similarity
def calculate_cosine_similarity(vector1, vector2):
    # NumPy 배열로 변환
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]

def calculate_average_similarity(row):
    title_similarity = calculate_cosine_similarity(input_title_vector, row['title_vector'])
    author_similarity = calculate_cosine_similarity(input_author_vector, row['author_vector'])
    category_similarity = calculate_cosine_similarity(input_category_vector, row['category_vector'])
    description_similarity = calculate_cosine_similarity(input_description_vector, row['description_vector'])
    return np.mean([title_similarity, author_similarity, category_similarity, description_similarity])

# Text to vector conversion
def text_to_vector(text, model, okt, stopwords):
    tokenized_sentence = okt.morphs(text, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if word not in stopwords]
    word_vectors = [model.wv[word] for word in stopwords_removed_sentence if word in model.wv]

    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)

def convert_text_to_vector(text, model, okt, stopwords):
    vector = text_to_vector(text, model, okt, stopwords)
    return vector

@app.post("/recommendation/")
async def get_recommendations(book_recommendation: BookRecommendation):
    try:
        recommendations = recommend_books(book_recommendation.book_title)
        return {"recommendations": recommendations}
    except HTTPException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

