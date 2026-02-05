"""중대성 매트릭스 시각화."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정."""
    # 폰트 매니저 재빌드
    fm._load_fontmanager(try_read_cache=False)

    # 시스템 폰트 경로
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",  # Windows 맑은고딕
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]

    font_found = False
    for font_path in font_paths:
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            font_found = True
            break

    if not font_found:
        # 기본 폰트 사용
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False


# ESG 영역별 색상 (H/E/S/G)
ESG_COLORS: dict[str, str] = {
    "H": "#e67e22",    # 대학경영: 주황
    "E": "#2ecc71",    # 환경: 녹색
    "S": "#3498db",    # 사회: 파랑
    "G": "#9b59b6",    # 지배구조: 보라
    "HEM": "#e67e22",  # 대학경영 (구버전 호환)
    "unknown": "#95a5a6",  # 기타: 회색
}


def get_esg_category(indicator: str, category_map: dict[str, str] | None = None) -> str:
    """진단항목의 ESG 카테고리 반환.

    Parameters
    ----------
    indicator : str
        진단항목명.
    category_map : dict, optional
        진단항목-ESG 매핑.

    Returns
    -------
    str
        ESG 카테고리 (E/S/G/HEM/unknown).
    """
    if category_map and indicator in category_map:
        return category_map[indicator]

    # 키워드 기반 추론
    indicator_lower = indicator.lower()

    if any(kw in indicator_lower for kw in ["환경", "에너지", "탄소", "폐기물", "녹색", "기후"]):
        return "E"
    elif any(kw in indicator_lower for kw in ["사회", "인권", "안전", "다양성", "지역", "학생", "교육"]):
        return "S"
    elif any(kw in indicator_lower for kw in ["지배구조", "윤리", "투명", "감사", "이사회"]):
        return "G"
    else:
        return "HEM"


def plot_materiality_matrix(
    df: "pd.DataFrame",
    output_path: str | Path | None = None,
    category_map: dict[str, str] | None = None,
    title: str = "ESG Materiality Matrix",
    figsize: tuple[int, int] = (14, 10),
    show_labels: bool = True,
    show_quadrants: bool = True,
    filter_predefined: bool = True,
) -> plt.Figure:
    """중대성 매트릭스 시각화.

    Parameters
    ----------
    df : pd.DataFrame
        중대성 점수 DataFrame.
        필수 컬럼: indicator, issue_exposure, impact_materiality, total_exposure
    output_path : str | Path, optional
        저장 경로.
    category_map : dict, optional
        진단항목-ESG 매핑.
    title : str
        차트 제목.
    figsize : tuple
        그림 크기.
    show_labels : bool
        레이블 표시 여부.
    show_quadrants : bool
        사분면 구분선 표시 여부.
    filter_predefined : bool
        True면 23개 정의된 ESG 지표만 표시.

    Returns
    -------
    plt.Figure
        생성된 Figure.
    """
    setup_korean_font()

    # 정의된 23개 ESG 지표만 필터링
    if filter_predefined:
        predefined_indicators = [
            "ESG 비교과 프로그램 개발 및 운영", "ESG 교육과정 및 교수학습 방법",
            "ESG 관련 캠페인", "ESG 관련 학생 동아리", "지역사회 ESG 파트너십",
            "지역사회 ESG 교육 프로그램", "에너지 사용 절감 및 체계적 관리",
            "청정 재생에너지", "탄소배출량", "물 사용량", "폐기물 배출",
            "유해 폐기물관리", "음식물 쓰레기 저감", "캠퍼스 차량",
            "교직원 복지향상 노력", "다양성과 형평성 추진체계", "취약계층 지원강화",
            "작업장의 건강 및 안전", "인권침해 예방 및 인권보장", "ESG 위원회 조직",
            "ESG 운영위원회의 다양성", "부패예방 및 청렴 강화", "회계의 투명성 관리",
        ]
        df = df[df["indicator"].isin(predefined_indicators)].copy()

    fig, ax = plt.subplots(figsize=figsize)

    # ESG 카테고리별 색상 할당
    colors = []
    for indicator in df["indicator"]:
        cat = get_esg_category(indicator, category_map)
        colors.append(ESG_COLORS.get(cat, ESG_COLORS["unknown"]))

    # 점 크기 계산 (노출도 기반)
    if "total_exposure" in df.columns:
        exposure = df["total_exposure"].values
        # Min-Max 정규화 후 50~500 범위로 스케일링
        if exposure.max() > exposure.min():
            sizes = 50 + (exposure - exposure.min()) / (exposure.max() - exposure.min()) * 450
        else:
            sizes = np.full(len(exposure), 200)
    else:
        sizes = np.full(len(df), 200)

    # 산점도
    scatter = ax.scatter(
        df["issue_exposure"],
        df["impact_materiality"],
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors="white",
        linewidth=1.5,
    )

    # 사분면 구분선 (0~5 스케일, 중앙 2.5)
    if show_quadrants:
        ax.axhline(y=2.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # 사분면 레이블
        ax.text(3.75, 4.7, "Q1: 핵심 중대 주제", fontsize=10, ha="center", style="italic", alpha=0.7)
        ax.text(1.25, 4.7, "Q2: 영향 우선", fontsize=10, ha="center", style="italic", alpha=0.7)
        ax.text(1.25, 0.3, "Q3: 모니터링", fontsize=10, ha="center", style="italic", alpha=0.7)
        ax.text(3.75, 0.3, "Q4: 이슈 우선", fontsize=10, ha="center", style="italic", alpha=0.7)

    # 레이블 표시
    if show_labels:
        for idx, row in df.iterrows():
            x = row["issue_exposure"]
            y = row["impact_materiality"]
            label = row["indicator"]

            # 긴 레이블 축약
            if len(label) > 15:
                label = label[:12] + "..."

            # 점 위치에서 약간 오프셋
            offset_x = 0.1
            offset_y = 0.1

            ax.annotate(
                label,
                (x, y),
                xytext=(x + offset_x, y + offset_y),
                fontsize=8,
                alpha=0.9,
                ha="left",
            )

    # 축 설정 (0~5 스케일)
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 5.2)
    ax.set_xlabel("Issue Exposure (이슈 노출도)", fontsize=12)
    ax.set_ylabel("Impact Materiality (영향 중대성)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 범례 (H/E/S/G)
    legend_elements = [
        plt.scatter([], [], c=ESG_COLORS["H"], s=100, label="H (대학경영)"),
        plt.scatter([], [], c=ESG_COLORS["E"], s=100, label="E (환경)"),
        plt.scatter([], [], c=ESG_COLORS["S"], s=100, label="S (사회)"),
        plt.scatter([], [], c=ESG_COLORS["G"], s=100, label="G (지배구조)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        title="ESG 영역",
        fontsize=9,
    )

    # 점 크기 범례 (미디어 노출도)
    size_legend_ax = fig.add_axes([0.78, 0.15, 0.15, 0.15])
    size_legend_ax.set_xlim(0, 1)
    size_legend_ax.set_ylim(0, 1)
    size_legend_ax.axis("off")
    size_legend_ax.text(0.5, 0.95, "미디어 노출도", fontsize=9, ha="center", fontweight="bold")

    for i, (s, label) in enumerate([(50, "낮음"), (175, "중간"), (300, "높음")]):
        y_pos = 0.7 - i * 0.25
        size_legend_ax.scatter([0.3], [y_pos], s=s, c="gray", alpha=0.6, edgecolors="white")
        size_legend_ax.text(0.55, y_pos, label, fontsize=8, va="center")

    # 그리드
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()

    # 저장
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"📊 차트 저장: {output_path}")

    return fig


def plot_top_indicators(
    df: "pd.DataFrame",
    top_n: int = 10,
    output_path: str | Path | None = None,
    category_map: dict[str, str] | None = None,
) -> plt.Figure:
    """상위 중대 지표 바 차트.

    Parameters
    ----------
    df : pd.DataFrame
        중대성 점수 DataFrame.
    top_n : int
        표시할 상위 지표 수.
    output_path : str | Path, optional
        저장 경로.
    category_map : dict, optional
        진단항목-ESG 매핑.

    Returns
    -------
    plt.Figure
        생성된 Figure.
    """
    setup_korean_font()

    # 복합 점수로 정렬
    df = df.copy()
    df["composite"] = df["issue_exposure"] + df["impact_materiality"]
    df = df.nlargest(top_n, "composite")

    fig, ax = plt.subplots(figsize=(12, 8))

    indicators = df["indicator"].tolist()
    y_pos = np.arange(len(indicators))
    width = 0.35

    # 색상 할당
    colors_iss = []
    colors_imp = []
    for indicator in indicators:
        cat = get_esg_category(indicator, category_map)
        base_color = ESG_COLORS.get(cat, ESG_COLORS["unknown"])
        colors_iss.append(base_color)
        # 영향 중대성은 약간 밝은 색상
        colors_imp.append(base_color + "99")

    # 바 차트 (0~5 스케일)
    bars1 = ax.barh(y_pos - width/2, df["issue_exposure"], width,
                     label="Issue Exposure", color=colors_iss, alpha=0.8)
    bars2 = ax.barh(y_pos + width/2, df["impact_materiality"], width,
                     label="Impact Materiality", color=colors_imp, alpha=0.6)

    ax.set_xlabel("Score (0-5)", fontsize=12)
    ax.set_title(f"Top {top_n} Material Indicators", fontsize=14, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(indicators, fontsize=10)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 5.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"📊 차트 저장: {output_path}")

    return fig
