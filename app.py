from flask import Flask, request, jsonify
import numpy as np
import random
import json
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from flask_cors import CORS
from clustering.genetic_algorithm import GeneticAlgorithmForClustering

from Extract_Keyword.keyword import extract_keywords, calculate_keyword_frequency

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(JSONEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)
app.json_encoder = JSONEncoder

# Flask의 기본 로거 설정
logging.basicConfig(level=logging.DEBUG)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.DEBUG)

@app.route('/cluster-data', methods=['POST'])
def cluster_data():
    app.logger.info("클러스터링 데이터 요청 받음")
    
    try:
        received_data = request.json['data']
        app.logger.debug(f"받은 데이터: {received_data}")

        # `member_id`와 나머지 데이터를 분리합니다.
        processed_data = []
        member_ids = []
        for item in received_data:
            member_id = item['member_id']
            member_ids.append(member_id)
            processed_item = [item['age'], item['gender'], item['interest_1'], item['interest_2'], item['interest_3'], item['interest_4'], item['interest_5']]
            processed_data.append(processed_item)
        
        app.logger.debug(f"처리된 데이터: {processed_data}")
        app.logger.debug(f"멤버 IDs: {member_ids}")
        
        # 데이터 스케일링
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(processed_data))
        print("스케일링")
        for row in scaled_data:
            print("[", ", ".join(f"{x:.2f}" for x in row), "],")
        # app.logger.debug(f"스케일링된 데이터: {scaled_data}")
    
        
        # t-SNE를 사용하여 차원 축소
        tsne_reducer = TSNE(n_components=2, random_state=42)
        reduced_data = tsne_reducer.fit_transform(scaled_data)
        app.logger.debug(f"차원 축소 결과: {reduced_data}")
        
        # 첫 번째 항에 member_ids 추가함
        combined_data = [[member_id, *coords] for member_id, coords in zip(member_ids, reduced_data)]
        app.logger.debug(f"조합된 데이터: {combined_data}")
        
                        # NearestNeighbors를 사용해 k-거리 계산
        def estimate_eps(data):
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(data)
            distances, indices = neighbors_fit.kneighbors(data)
            distances = np.sort(distances, axis=0)[:, 3]
            return np.percentile(distances, 90)  # 90번째 백분위수를 사용
        
        estimated_eps = estimate_eps(reduced_data)
        app.logger.debug(f"추정된 EPS: {estimated_eps}")

        # 클러스터링
        genetic_clustering = GeneticAlgorithmForClustering(combined_data, estimated_eps)
        results = genetic_clustering.run_algorithm();
        print(results)
        
        #모든 group_id를 Python의 int로 변환
        for result in results:
            result['group_id'] = int(result['group_id'])
        app.logger.info(f"최종 클러스터링 결과: {results}")
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"클러스터링 중 오류 발생: {e}")
        return jsonify({'error': '클러스터링 처리 중 오류 발생'}), 500

#키워드 리스트
keywords_list = {
    "스포츠 활동": [
        "축구", "농구", "테니스", "수영", "등산", "날씨",
        "게이트볼", "걷기", "골프", "체조", "당구", "볼링", 
        "스트레칭", "요가", "탁구", "낚시", "자전거", "댄스스포츠", 
        "에어로빅", "플라잉디스크", "배드민턴", "복싱", "런닝", 
        "풋살", "스쿼시", "야구"
    ],
    "사회 및 기타 활동": [
        "자원봉사", "선거", "클럽", "동호회", "커뮤니티",
        "교회", "성당", "전화", "노인정", "가족", "절", 
        "봉사활동", "템플스테이", "강연", "성경", "토론", 
        "나들이", "김장", "경로당", "동창회", "예배", 
        "심리상담", "동물보호", "기부", "재활", "헌혈", "홈리스"
    ],
    "문화예술 관람 활동": [
        "공연", "전시회", "음악회", "영화", "박물관",
        "노래교실", "서예", "춤", "연주", "그림", 
        "갤러리", "카드 게임", "문화센터", "연극", 
        "여행", "도서관", "소설", "뮤지컬", "공예", 
        "합창", "희곡", "건축물", "가요제", "시", 
        "대중문화", "컬쳐소핑"
    ],
    "취미오락 활동": [
        "요리", "그림", "악기", "독서", "만화", "역사",
        "화투", "고스톱", "장기", "바둑", "책", "화초", 
        "퍼즐", "종이접기", "정원(가꾸기)", "수다(떨기)", "체스", 
        "게임", "악기 연주", "강아지", "사진 찍기", 
        "텃밭", "평생교육", "경매", "야구장", "포커", 
        "경마", "고양이", "작곡"
    ],
    "휴식활동": [
        "스파", "산책", "명상", "온천", "바베큐",
        "감상", "사우나", "라디오", "티비", "신문", 
        "소풍", "티타임", "마사지", "꽃구경", 
        "트로트", "유튜브", "인터넷", "낮잠", 
        "카톡", "낚시", "휴양지", "캠핑", "놀이공원", "호캉스"
    ]
}

@app.route('/dialogue-data', methods=['POST'])
def process_dialogue_data():
    # JSON 형식으로 전달된 대화 데이터 받기
    dialogue_data = request.json

    # 대화 데이터에서 대화 부분 추출
    dialogue = dialogue_data['dialogue']

    # 키워드 추출
    extracted_keywords = extract_keywords(dialogue)

    # 키워드 빈도수 계산
    keyword_frequency = calculate_keyword_frequency(extracted_keywords, keywords_list)

    # 결과를 JSON 형태로 반환
    return jsonify({
        'extracted_keywords': extracted_keywords,
        'keyword_frequency': keyword_frequency
    })



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
