# -*- coding: utf-8 -*-
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

#키워드 추출 함수 
def extract_keywords(dialogue):
    # KoNLPy 형태소 분석기 초기화
    okt = Okt()

    # 형태소 분석 후 명사만 추출
    nouns_list = [' '.join(okt.nouns(doc)) for doc in dialogue]

    # TF-IDF 벡터화를 수행 (TF-IDF 모델 생성)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(nouns_list)

    # 단어 목록 및 각 단어의 TF-IDF 값 출력
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    # TextRank 그래프 생성
    graph = nx.Graph()
    for doc in dialogue:
        tokens = okt.nouns(doc)
        graph.add_nodes_from(tokens)
        graph.add_edges_from([(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)])

    # TextRank 알고리즘을 적용하여 중요한 단어 추출
    pagerank_scores = nx.pagerank(graph)

    # 총 문장 수 확인
    total_sentences = len(dialogue)

    # n_keywords를 총 문장 수의 약 25%로 설정
    n_keywords = int(total_sentences * 0.25)

    # TF-IDF 값이 높은 순으로 정렬하여 상위 키워드 선택
    top_tfidf_keywords = [feature_names[i] for i in tfidf_scores.argsort()[::-1][:n_keywords]]

    # TextRank 알고리즘을 적용하여 중요한 단어 추출
    top_pagerank_keywords = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:n_keywords]

    # 두 방법에서 추출한 키워드에 각각 가중치를 부여하여 혼합
    tfidf_weight = 0.3
    pagerank_weight = 0.7

    # Convert feature_names to a list
    feature_names_list = list(feature_names)

    weighted_keywords = {
        keyword: (
            tfidf_weight * tfidf_scores[feature_names_list.index(keyword)]
            if keyword in feature_names_list else 0
        ) + (
            pagerank_weight * pagerank_scores[keyword]
            if keyword in pagerank_scores else 0
        )
        for keyword in set(top_tfidf_keywords + top_pagerank_keywords)
    }

    # 가중치가 높은 순으로 정렬하여 상위 키워드 선택
    combined_keywords = sorted(weighted_keywords, key=weighted_keywords.get, reverse=True)[:n_keywords]

    return combined_keywords

#키워드 빈도수 계산 함수 
def calculate_keyword_frequency(combined_keywords, keywords_list):
    # 빈도수를 저장할 딕셔너리 초기화
    category_keyword_count = {category: 0 for category in keywords_list}

    # Combined Keywords에 일치하는 키워드가 있는지 확인하고 해당 카테고리의 키워드 개수를 누적하여 저장
    for keyword in combined_keywords:
        for category, keywords in keywords_list.items():
            if keyword in keywords:
                category_keyword_count[category] += 1

    return category_keyword_count

