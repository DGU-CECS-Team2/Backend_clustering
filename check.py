def check_duplicate_members(groups):
    from collections import defaultdict

    # 각 멤버 ID가 어느 그룹에 속하는지 추적
    member_to_group = defaultdict(list)
    
    # 모든 그룹의 member_ids를 순회하면서 각 멤버 ID의 등장 그룹을 기록
    for group in groups:
        for member_id in group["member_ids"]:
            member_to_group[member_id].append(group["group_id"])
    
    # 중복이 발견된 멤버 ID와 해당 그룹 리스트를 저장
    duplicates = {member_id: group_ids for member_id, group_ids in member_to_group.items() if len(group_ids) > 1}
    
    # 중복된 멤버 ID가 없으면 아무 것도 반환하지 않음
    if not duplicates:
        return "No duplicates found."
    
    # 중복 정보 출력
    duplicate_info = []
    for member_id, group_ids in duplicates.items():
        duplicate_info.append(f"Member ID {member_id} is in groups {group_ids}.")
        
    return "\n".join(duplicate_info)


# 예시 데이터 사용
groups = [
    {
        "group_id": 1,
        "member_ids": [
            3,
            61,
            12,
            48,
            60
        ]
    },
    {
        "group_id": 2,
        "member_ids": [
            4,
            76,
            98,
            1,
            84
        ]
    },
    {
        "group_id": 3,
        "member_ids": [
            9,
            42,
            46,
            22,
            32
        ]
    },
    {
        "group_id": 4,
        "member_ids": [
            14,
            91,
            94,
            53,
            21
        ]
    },
    {
        "group_id": 5,
        "member_ids": [
            31,
            57,
            56,
            30,
            77
        ]
    },
    {
        "group_id": 6,
        "member_ids": [
            37,
            63,
            29,
            16,
            81
        ]
    },
    {
        "group_id": 7,
        "member_ids": [
            38,
            92,
            50,
            55,
            26
        ]
    },
    {
        "group_id": 8,
        "member_ids": [
            47,
            89,
            87,
            44,
            67
        ]
    },
    {
        "group_id": 9,
        "member_ids": [
            52,
            66,
            82,
            70,
            64
        ]
    },
    {
        "group_id": 10,
        "member_ids": [
            58,
            54,
            23,
            13,
            20
        ]
    },
    {
        "group_id": 11,
        "member_ids": [
            62,
            79,
            2,
            93,
            15
        ]
    },
    {
        "group_id": 12,
        "member_ids": [
            6,
            88,
            17,
            41,
            99
        ]
    },
    {
        "group_id": 13,
        "member_ids": [
            97,
            35,
            95,
            100,
            86
        ]
    },
    {
        "group_id": 14,
        "member_ids": [
            72,
            73,
            90,
            71,
            68
        ]
    },
    {
        "group_id": 15,
        "member_ids": [
            74,
            27,
            33,
            25,
            10
        ]
    },
    {
        "group_id": 16,
        "member_ids": [
            49,
            65,
            43,
            85,
            28
        ]
    },
    {
        "group_id": 17,
        "member_ids": [
            36,
            34,
            59,
            45,
            80
        ]
    },
    {
        "group_id": 18,
        "member_ids": [
            40,
            51,
            96,
            69,
            78
        ]
    },
    {
        "group_id": 19,
        "member_ids": [
            83,
            8,
            19,
            11,
            39
        ]
    },
    {
        "group_id": 20,
        "member_ids": [
            5,
            75,
            7,
            18,
            24
        ]
    }
]
# 함수 실행 및 결과 출력
result = check_duplicate_members(groups)
print(result)