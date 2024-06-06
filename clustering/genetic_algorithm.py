from clustering.dbscan_a import DBSCANClusterer
import numpy as np

class GeneticAlgorithmForClustering:
    def __init__(self, data, input_eps=0.3, input_min_samples=5, num_individuals=100, cluster_size=5, crossover_rate=0.85, mutation_rate=0.01):
        self.data = np.array(data)
        self.num_individuals = num_individuals
        self.cluster_size = cluster_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population(input_eps, input_min_samples, data)

    def initialize_population(self, eps, min_samples, data):
        # 좌표 데이터 추출 및 타입 변환
        coordinates = np.array([point[1:] for point in data], dtype=float)

        # DBSCAN 클러스터러 생성 및 예측
        dbscan = DBSCANClusterer(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coordinates)

        # 클러스터 별 인덱스 매핑
        clusters = {}
        for label in np.unique(labels):
            if label != -1:
                cluster_indices = np.where(labels == label)[0]
                clusters[label] = cluster_indices

        # 전체 개체군 저장
        total_population = []

        # 개체 생성
        for _ in range(self.num_individuals):
            selected_indices = set()
            for cluster_indices in clusters.values():
                num_to_select = max(1, len(cluster_indices) // 5)
                selected_from_cluster = np.random.choice(cluster_indices, num_to_select, replace=False)
                selected_indices.update(selected_from_cluster)

            # 필요한 추가 인덱스 선택
            if len(selected_indices) < 20:
                additional_needed = 20 - len(selected_indices)
                available_indices = list(set(range(len(data))) - selected_indices)
                additional_selected = np.random.choice(available_indices, additional_needed, replace=False)
                selected_indices.update(additional_selected)

            # 개체군 추가
            total_population.append(list(selected_indices))

        return total_population



    def calculate_fitness(self, individual):
        """
        개체의 적합도를 계산하는 함수.
        :param individual: 개체 (클러스터 센터포인트의 인덱스 배열)
        :param data: 클러스터링할 전체 데이터 세트 (예상되는 데이터 형태: 각 행이 [id, x, y] 포맷)
        :return: 적합도 점수
        """
        # fitness = 0
        # for center_index in individual:
        #     # 클러스터 중심점 좌표 추출, float으로 변환
        #     center_point = np.array(data[center_index][1:], dtype=np.float64)
        #     # 데이터 포인트 좌표 추출 및 데이터 타입 변환
        #     points = np.array([item[1:] for item in data], dtype=np.float64)
        #     # 유클리드 거리 계산
        #     distances = np.linalg.norm(points - center_point, axis=1)
        #     closest_indices = np.argsort(distances)[:self.cluster_size]
        #     fitness -= np.sum(distances[closest_indices])
        # return fitness
        
        fitness = 0
        for center_index in individual:
            center_point = self.data[center_index, 1:]
            distances = np.linalg.norm(self.data[:, 1:] - center_point, axis=1)
            fitness -= np.sum(np.sort(distances)[:self.cluster_size])
        return fitness

    def select(self, fitnesses):
        fitnesses = np.array(fitnesses)
        if len(fitnesses) != self.num_individuals:
            raise ValueError(f"Expected {self.num_individuals} fitness values, but got {len(fitnesses)}")

        # 계산된 확률값의 안정성을 보장
        fitnesses = fitnesses - np.min(fitnesses)
        probabilities = np.exp(fitnesses) / np.sum(np.exp(fitnesses))

        # probabilities의 합이 1이 되도록 조정
        probabilities /= probabilities.sum()

        # 오류가 여기서 발생하면 문제는 probabilities의 크기가 num_individuals와 다름을 의미
        selected_indices = np.random.choice(self.num_individuals, size=self.num_individuals, replace=False, p=probabilities)
        return [self.population[i] for i in selected_indices]



    
    
    def crossover(self, parent1, parent2):
        # 부모 배열의 길이를 확인하여 적절한 조치를 취합니다.

        if len(parent1) < 1 or len(parent2) < 1:

            return parent1, parent2

        # 교차점을 선택합니다.
        crossover_point = np.random.randint(1, len(parent1))

        # 부모 배열이 리스트가 아니라 numpy 배열일 경우, 리스트로 변환
        child1 = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
        child2 = list(parent2[:crossover_point]) + list(parent1[crossover_point:])
        
            # 중복 제거 및 누락된 요소 보충
        def fix_child(original, cross_section):
            fixed_child = list(dict.fromkeys(original))  # 중복 제거
            missing_elements = [item for item in self.data if item not in fixed_child]
            np.random.shuffle(missing_elements)
            fixed_child += missing_elements[:len(original) - len(fixed_child)]
            return fixed_child
        return child1, child2

    def mutate(self, individual):
        """
        개체의 변이를 수행하는 함수.
        :param individual: 변이할 개체
        :param data: 클러스터링할 전체 데이터 세트
        :return: 변이된 개체
        """
  
        if individual is None or not isinstance(individual, (list, np.ndarray)) or len(individual) == 0:
            print("Error: 'individual' is None or empty.")
            return individual
        
        individual_set = set(individual)  # 중복을 피하기 위해 집합 사용
        individual_list = list(individual)  # 리스트로 변환하여 인덱싱 가능하게 함
        
        if self.data is None or len(self.data) == 0:
            print("Error: 'self.data' is None or empty.")
            return individual
        
        if isinstance(individual, np.ndarray):
            individual_list = individual.tolist()  # NumPy 배열을 리스트로 변환
        elif isinstance(individual, list):
            individual_list = individual  # 이미 리스트인 경우 그대로 사용
        else:
            print("Warning: 'individual' is neither a NumPy array nor a list.")
            return individual  # 스칼라나 다른 형태의 데이터일 경우 그대로 반환하거나 다른 처리 수행
 
        for i in range(len(individual_list)):
            try:
                if np.random.rand() < self.mutation_rate:
                    available_choices = [item for item in range(len(self.data)) if item not in individual_set]
                    if not available_choices:  # 모든 가능한 변이가 이미 사용되었을 경우
                        continue
                    choice = np.random.choice(available_choices)
                    individual_set.remove(individual_list[i])
                    individual_set.add(choice)
                    individual_list[i] = choice
            except Exception as e:
                print(f"Error occurred : {str(e)}")
                # 관련 변수 상태 출력
                break  # 또는 적절한 예외 처리

        return individual_list
    
    def run_algorithm(self):
        """
        유전 알고리즘을 실행하는 함수.
        :param data: 클러스터링할 전체 데이터 세트
        :return: 최적의 클러스터 포인트 리스트
        """
        best_fitness = float('-inf')
        generations = 0
        member_ids = [point[0] for point in self.data]  # 멤버 ID 추출
        for generation_count in range (600):  # 최대 500세대 실행
            try:
                fitnesses = [self.calculate_fitness(ind) for ind in self.population]
                if not fitnesses:  # Check if fitnesses list is empty
                    break
                
                best_current_fitness = max(fitnesses)
                if best_fitness == float('-inf'):
                    best_fitness = best_current_fitness - 1 
                if best_current_fitness > best_fitness:
                    if (best_current_fitness - best_fitness) / abs(best_fitness) < 0.01:
                        generations += 1
                    else:
                        generations = 0
                    best_fitness = best_current_fitness
            
                if generations >= 50:
                    break

                self.population = self.select(fitnesses)

                if not self.population:  # Ensure population is not empty after selection
                    break
                new_population = []
                for i in range(0, len(self.population), 2):
                
                    if i+1 >= len(self.population):  # Check if there's a pair to crossover
                        new_population.append(self.population[i])  # Add the last unpaired individual
                        break
        
                    parent1, parent2 = self.population[i], self.population[i+1]
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(self.mutate(child1))
                    new_population.append(self.mutate(child2))
 

                self.population = new_population
            except Exception as e:
                print(f"Error occurred at generation {generation_count + 1}: {str(e)}")
                # 관련 변수 상태 출력
                break  # 또는 적절한 예외 처리
        
        print(f"Total generations executed: {generation_count + 1}")  # 실행된 총 세대 수 출력
        print(f"Final best fitness: {best_fitness}")

        best_individual = self.population[np.argmax(fitnesses)]
        print("best",best_individual)
        results = []
                 # 멤버 ID와 위치 추출
        member_ids = [int(point[0]) for point in self.data]  # 멤버 ID를 정수형으로 보장
        positions = np.array([point[1:] for point in self.data])
        
            # 클러스터 중심의 위치 가져오기
        cluster_center_points = positions[best_individual]
        
        # 모든 점에 대해 각 중심점과의 거리 계산
        distances = np.linalg.norm(positions[:, np.newaxis, :] - cluster_center_points, axis=2)
        
        results = []
        used_ids = set()  # 이미 선택된 멤버 ID를 저장하는 집합
        for i, center_index in enumerate(best_individual):
            center_member_id = member_ids[center_index]
            sorted_indices = np.argsort(distances[:, i])
            closest_indices = [center_index] if center_member_id not in used_ids else []
            used_ids.add(center_member_id)  # 중심점 멤버 ID 추가
            
            for idx in sorted_indices:
                current_member_id = member_ids[idx]
                if current_member_id not in used_ids and len(closest_indices) < 5:
                    closest_indices.append(idx)
                    used_ids.add(current_member_id)  # 선택된 멤버 ID를 사용된 집합에 추가
                if len(closest_indices) >= 5:  # 필요한 멤버 수가 충족되면 반복 중지
                    break

            # 인덱스를 멤버 ID로 매핑
            group_member_ids = [member_ids[idx] for idx in closest_indices]
            results.append({'group_id': i + 1, 'member_ids': group_member_ids})

        return results
                    
