"""매체 가중치 분류."""

from __future__ import annotations

import re

# 매체 그룹 정의
MEDIA_GROUPS: dict[str, list[str]] = {
    # 그룹1: 중앙일간지/공중파 (가중치 1.0)
    "major": [
        "조선일보", "중앙일보", "동아일보", "한겨레", "경향신문", "한국일보",
        "서울신문", "세계일보", "문화일보", "국민일보",
        "KBS", "MBC", "SBS", "YTN", "JTBC", "TV조선", "채널A", "MBN",
        "연합뉴스", "뉴스1", "뉴시스",
    ],
    # 그룹2: 경제지/전문지 (가중치 0.8)
    "economic": [
        "매일경제", "한국경제", "서울경제", "파이낸셜뉴스", "머니투데이",
        "이데일리", "아시아경제", "헤럴드경제", "디지털타임스", "전자신문",
        "아주경제", "이투데이", "비즈니스워치", "더벨", "딜사이트",
        "한경비즈니스", "이코노미스트", "시사저널", "주간조선", "신동아",
        "대학저널", "베리타스알파", "대학신문", "교수신문", "한국대학신문",
        # 추가 (2024)
        "대한경제", "에너지경제", "비즈니스포스트", "브릿지경제", "한스경제",
        "녹색경제신문", "조선비즈", "아이뉴스24", "지디넷코리아", "이뉴스투데이",
    ],
    # 그룹3: 지역/인터넷 (가중치 0.5)
    "regional": [
        "경인일보", "인천일보", "경기일보", "부산일보", "국제신문",
        "대전일보", "충청일보", "광주일보", "전남일보", "강원일보",
        "제주일보", "매일신문", "영남일보", "전북일보", "충청투데이",
        "오마이뉴스", "프레시안", "미디어오늘", "더팩트", "뉴스핌",
        "데일리안", "쿠키뉴스", "메디게이트", "청년의사", "메디칼타임즈",
        "노컷뉴스", "CBS", "뉴스타파", "인사이트", "위키트리",
        # 추가 (2024)
        "데일리한국", "아시아타임즈", "브레이크뉴스", "중도일보", "국제뉴스",
        "뉴데일리", "아시아투데이", "메트로신문", "뉴스프리존", "천지일보",
        "일요신문", "신아일보", "한국강사신문", "E동아", "경남도민신문",
        "내일신문", "임팩트온", "경북일보", "전북도민일보",
    ],
    # 그룹4: 국민대신문 (가중치 0.3)
    "internal": [
        "국민대신문", "국민대학교", "kookmin",
    ],
}

# 그룹별 가중치
MEDIA_WEIGHTS: dict[str, float] = {
    "major": 1.0,
    "economic": 0.8,
    "regional": 0.5,
    "internal": 0.3,
    "unknown": 0.4,  # 기본 가중치
}


def classify_media(press_name: str) -> str:
    """매체명을 그룹으로 분류.

    Parameters
    ----------
    press_name : str
        매체명.

    Returns
    -------
    str
        그룹명 (major/economic/regional/internal/unknown).
    """
    if not press_name:
        return "unknown"

    press_lower = press_name.lower().strip()

    for group, media_list in MEDIA_GROUPS.items():
        for media in media_list:
            if media.lower() in press_lower or press_lower in media.lower():
                return group

    return "unknown"


def get_media_weight(press_name: str) -> float:
    """매체명의 가중치 반환.

    Parameters
    ----------
    press_name : str
        매체명.

    Returns
    -------
    float
        매체 가중치 (0.3 ~ 1.0).
    """
    group = classify_media(press_name)
    return MEDIA_WEIGHTS.get(group, MEDIA_WEIGHTS["unknown"])


def calculate_exposure_score(
    press_count: int,
    press_list: str,
    base_weight: float = 1.0,
) -> float:
    """미디어 노출도 점수 계산.

    Parameters
    ----------
    press_count : int
        보도 횟수.
    press_list : str
        언론사 목록 (쉼표 구분).
    base_weight : float
        기본 가중치.

    Returns
    -------
    float
        미디어 노출도 점수.
    """
    if press_count <= 0:
        return 0.0

    if not press_list:
        return press_count * MEDIA_WEIGHTS["unknown"] * base_weight

    media_names = [m.strip() for m in press_list.split(",") if m.strip()]

    if not media_names:
        return press_count * MEDIA_WEIGHTS["unknown"] * base_weight

    # 각 매체별 가중치 합산
    total_weight = sum(get_media_weight(media) for media in media_names)

    return total_weight * base_weight
