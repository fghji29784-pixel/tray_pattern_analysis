# -*- coding: utf-8 -*-
"""
tray_pattern_analyzer.py
────────────────────────────────────────────────────────────────────────
트레이(12x12=144셀) 내부의 온도 / OCV 공간 패턴을 찾아내고,
"이런 패턴을 가진 트레이가 몇 개인지"를 일반인이 이해할 수 있게 요약한다.

설계 원칙
  - Python 3.8 32bit 호환: scipy/sklearn 미사용, numpy 직접 구현
  - 랏-트레이 복합키 (Lot ID, TRAY ID) 로 모든 트레이별 계산 수행
  - 온도 오류 셀 제거: T1/T2/T3 중 하나라도 22℃ 미만이면 그 셀 제외
  - 라벨(판정등급)은 '색칠/참고용'일 뿐, 판정 기준으로 쓰지 않는다(순환 방지)

분석 단계
  Phase 0  데이터 로드 · 위치매핑 · 품질필터(22℃) · 제외리포트
  Phase 1  4개 축(위치·온도·OCV·시간) 관계 매트릭스
  Phase 2  '위치효과' 분산 게이트 (위치가 분산의 몇 %인가) ← go/no-go
  Phase 3  트레이별 구배(방향·세기) → 패턴 분류 → 패턴별 트레이 개수

산출물
  - PNG 대시보드 (제외요약 / 온도패턴 / OCV패턴 / 분산게이트)
  - 패턴별 대표 트레이 히트맵 카드 PNG
  - 요약 Excel (여러 시트)
────────────────────────────────────────────────────────────────────────
사용법:
  python tray_pattern_analyzer.py --input "데이터.xlsx" --outdir "results"
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일 저장
import matplotlib.pyplot as plt

# 한글 폰트: 설치된 것 중 첫 번째를 자동 선택 (환경마다 이름이 달라 깨지는 문제 방지)
def _set_korean_font():
    import matplotlib.font_manager as fm
    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Nanum Gothic",
                  "Gulim", "Dotum", "Batang", "Gungsuh", "AppleGothic",
                  "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            plt.rcParams["font.sans-serif"] = [c] + plt.rcParams.get("font.sans-serif", [])
            print("[폰트] 한글 폰트 사용:", c)
            return c
    print("[폰트][경고] 한글 폰트를 찾지 못했습니다 → 한글이 깨질 수 있습니다.")
    print("           설치된 폰트 예:", sorted(available)[:10])
    return None

_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════════
#  0. 설정 : 실제 엑셀 헤더에 맞춘 컬럼 이름
# ══════════════════════════════════════════════════════════════════════
COL = {
    "lot":     "Lot ID",
    "tray":    "TRAY ID_Charge #05",
    "cellno":  "CELL NO_Charge #05",
    "cellpos": "Cell 위치",
    "label":   "판정등급",
    "docv":    "Delta OCV_Delta OCV #07",
    "ocv1":    "OCV_PRIVT OCV #01",
    "ocv2":    "OCV_PRIVT OCV #02",
    "ocv3":    "OCV_PRIVT OCV #03",
    "t1":      "PRIVT OCV #01 온도",
    "t2":      "PRIVT OCV #02 온도",
    "t3":      "PRIVT OCV #03 온도",
    "st1":     "Start Time_PRIVT OCV #01",
    "st2":     "Start Time_PRIVT OCV #02",
    "st3":     "Start Time_PRIVT OCV #03",
    # 전공정(방전7) + 방전 직후 OCV7 (선택 컬럼: 있으면 열이력/완화 분석 활성화)
    "dis_tmin": "DisCharge #07 최저 온도",
    "dis_tavg": "DisCharge #07 평균온도",
    "dis_tmax": "DisCharge #07 최고 온도",
    "ocv7":     "OCV #07 OCV",
    "st7":      "OCV #07 시작시간",
}

GRID = 12                 # 12 x 12
N_CELLS = GRID * GRID     # 144
TEMP_MIN = 22.0           # 22℃ 미만 = 센서오류로 간주하여 제외
MIN_VALID_CELLS = 100     # 트레이당 유효셀이 이보다 적으면 패턴분석 제외
PATTERN_R2_MIN = 0.15     # 구배 설명력이 이보다 낮으면 '패턴 약함(균일)'
PATTERN_R2_STRONG = 0.30  # 이 이상이면 '강한(신뢰할 만한) 패턴'으로 별도 집계
RADIAL_MARGIN = 1.3       # 방사형(중앙/외곽)은 선형보다 이 배 이상 우세할 때만 인정
OUTLIER_K = 3.5           # 트레이 내 튀는 셀(이상치) 판정: |값-중앙값| > K·MAD
DEFECT_LABELS = {"E", "NG", "불량"}   # 불량 등급 표기(있으면 색칠에만 사용)

# 측정 시점과 온도컬럼, 화면 표기(경과일). OCV1 → +2일 → OCV2 → +1일 → OCV3
TIMEPOINTS = [("OCV1", "t1", "OCV1\n(기준)"),
              ("OCV2", "t2", "OCV2\n(+2일)"),
              ("OCV3", "t3", "OCV3\n(+3일)")]


# ══════════════════════════════════════════════════════════════════════
#  Phase 0 : 로드 · 위치매핑 · 품질필터
# ══════════════════════════════════════════════════════════════════════
def load_data(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    optional = {"st1", "st2", "st3", "cellpos", "label",
                "dis_tmin", "dis_tavg", "dis_tmax", "ocv7", "st7"}  # 없어도 분석 가능
    req_missing = [COL[k] for k in COL if k not in optional and COL[k] not in df.columns]
    opt_missing = [COL[k] for k in optional if COL[k] not in df.columns]
    if req_missing:
        print("[경고] 분석에 꼭 필요한 컬럼을 찾지 못했습니다 (이름 확인 필요):")
        for m in req_missing:
            print("   -", m)
        print("발견된 컬럼:", list(df.columns))
    if opt_missing:
        print("[정보] 선택 컬럼이 없어 해당 부분만 생략합니다:", opt_missing)
    return df


def cellno_to_rc(n):
    """CELL NO(1~144) -> (row 0~11, col 0~11).
    col = (n-1)//12, row = (n-1)%12  (열-우선; CLAUDE 프로젝트 규약)."""
    if pd.isna(n):
        return (np.nan, np.nan)
    n = int(round(n))
    if n < 1 or n > N_CELLS:
        return (np.nan, np.nan)
    idx = n - 1
    col = idx // GRID
    row = idx % GRID
    return (row, col)


def parse_cellpos(s):
    """'Cell 위치'가 'A01' 형태이면 (row,col)로 파싱, 아니면 (nan,nan)."""
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip().upper()
    if len(s) < 2 or not s[0].isalpha():
        return (np.nan, np.nan)
    col = ord(s[0]) - ord("A")           # A->0 ... L->11
    digits = "".join(ch for ch in s[1:] if ch.isdigit())
    if digits == "":
        return (np.nan, np.nan)
    row = int(digits) - 1
    if not (0 <= col < GRID and 0 <= row < GRID):
        return (np.nan, np.nan)
    return (row, col)


def add_positions(df):
    """row/col 추가. 기본은 CELL NO 기반, 'Cell 위치'가 있으면 교차검증 로그."""
    rc = df[COL["cellno"]].apply(cellno_to_rc)
    df["row"] = [r for r, c in rc]
    df["col"] = [c for r, c in rc]

    # 교차검증 (Cell 위치가 파싱 가능하면 일치율 출력)
    if COL["cellpos"] in df.columns:
        rc2 = df[COL["cellpos"]].apply(parse_cellpos)
        r2 = np.array([r for r, c in rc2], dtype=float)
        c2 = np.array([c for r, c in rc2], dtype=float)
        ok = ~np.isnan(r2)
        if ok.sum() > 0:
            match = np.mean(
                (df["row"].values[ok] == r2[ok]) & (df["col"].values[ok] == c2[ok])
            )
            print("[위치 교차검증] 'Cell 위치' 파싱가능 %d행, CELL NO 매핑과 일치율 %.1f%%"
                  % (ok.sum(), 100 * match))
            if match < 0.9:
                print("   ⚠ 일치율이 낮습니다 → 위치 매핑 규약을 재확인하세요.")
    return df


def to_num(df, key):
    return pd.to_numeric(df[COL[key]], errors="coerce")


def quality_filter(df):
    """22℃ 필터 + 위치/OCV 결측 처리. (유효 df, 제외리포트 dict) 반환."""
    n_total = len(df)

    for k in ["t1", "t2", "t3", "ocv1", "ocv2", "ocv3", "docv"]:
        df[k] = to_num(df, k)

    # 온도 오류: 세 시점 중 하나라도 22℃ 미만
    temp_min = df[["t1", "t2", "t3"]].min(axis=1)
    temp_missing = df[["t1", "t2", "t3"]].isna().any(axis=1)
    cold = temp_min < TEMP_MIN

    pos_missing = df["row"].isna() | df["col"].isna()
    ocv_missing = df[["ocv1", "ocv3"]].isna().any(axis=1)

    bad = cold | temp_missing | pos_missing | ocv_missing
    df_valid = df.loc[~bad].copy()

    report = {
        "전체 셀": n_total,
        "온도<22℃ (센서오류 추정)": int(cold.sum()),
        "온도 결측": int(temp_missing.sum()),
        "위치 결측": int(pos_missing.sum()),
        "OCV 결측": int(ocv_missing.sum()),
        "── 최종 제외(중복 제거)": int(bad.sum()),
        "분석 대상": int((~bad).sum()),
    }
    return df_valid, report


def tray_key(df):
    """랏-트레이 복합키 문자열."""
    return df[COL["lot"]].astype(str) + " / " + df[COL["tray"]].astype(str)


# ══════════════════════════════════════════════════════════════════════
#  Phase 1 : 4개 축 관계 매트릭스
# ══════════════════════════════════════════════════════════════════════
def build_time_features(df):
    """측정 시각 -> 경과시간(시간). 시간 컬럼이 없거나 파싱 실패하면 NaN.
    (시간은 관계표의 참고지표로만 쓰이므로 없어도 전체 분석은 정상 동작)."""
    have_time = all(COL[k] in df.columns for k in ["st1", "st3"])
    for k in ["st1", "st2", "st3"]:
        if COL[k] in df.columns:
            df[k + "_dt"] = pd.to_datetime(df[COL[k]], errors="coerce")
        else:
            df[k + "_dt"] = pd.NaT
    if have_time and df["st1_dt"].notna().any() and df["st3_dt"].notna().any():
        df["경과시간_1to3(h)"] = (df["st3_dt"] - df["st1_dt"]).dt.total_seconds() / 3600.0
    else:
        df["경과시간_1to3(h)"] = np.nan
        print("[정보] 측정 시각 컬럼이 없거나 비어 있음 → '경과시간'은 생략(나머지 분석 정상).")
    return df


def relationship_matrix(df, outpath):
    """위치·온도·OCV·시간 대표 스칼라 간 상관행렬 히트맵."""
    df = df.copy()
    df["중심거리r"] = np.sqrt((df["col"] - 5.5) ** 2 + (df["row"] - 5.5) ** 2)
    df["냉각량T1-T3"] = df["t1"] - df["t3"]

    feats = {
        "가로위치(열)": "col",
        "세로위치(행)": "row",
        "중심거리": "중심거리r",
        "온도T1": "t1",
        "온도T3": "t3",
        "냉각량(T1-T3)": "냉각량T1-T3",
        "ΔOCV": "docv",
        "OCV1": "ocv1",
        "OCV3": "ocv3",
        "경과시간": "경과시간_1to3(h)",
    }
    labels = list(feats.keys())
    mat = np.full((len(labels), len(labels)), np.nan)
    cols_data = {}
    for name, c in feats.items():
        cols_data[name] = pd.to_numeric(df[c], errors="coerce").values

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            x, y = cols_data[a], cols_data[b]
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() > 10 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                mat[i, j] = np.corrcoef(x[m], y[m])[0, 1]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(mat[i, j]) > 0.5 else "black")
    ax.set_title("① 위치·온도·OCV·시간 관계표 (상관계수, -1~+1)\n"
                 "숫자가 0에 가까우면 관계 약함", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="상관계수")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return pd.DataFrame(mat, index=labels, columns=labels)


# ══════════════════════════════════════════════════════════════════════
#  Phase 2 : '위치효과' 분산 게이트 (go/no-go)
# ══════════════════════════════════════════════════════════════════════
def variance_components(df, value_col, keys):
    """
    분산을 (위치고정효과 / 트레이랜덤 / 잔차)로 분해.
    z_pt ≈ μ + α_p(위치) + β_t(트레이) + ε
    반환: 각 성분이 전체 분산에서 차지하는 % dict.
    위치% 가 작으면 → 전역 위치보정은 효과 없음(게이트 통과).
    """
    d = df[["row", "col", value_col]].copy()
    d["pos"] = d["row"].astype(int) * GRID + d["col"].astype(int)
    d["tray"] = keys.values
    d = d.dropna(subset=[value_col])

    mu = d[value_col].mean()
    pos_mean = d.groupby("pos")[value_col].mean()   # 위치별 평균(트레이 넘나듦)
    tray_mean = d.groupby("tray")[value_col].mean()  # 트레이별 평균

    d["alpha"] = d["pos"].map(pos_mean) - mu
    d["beta"] = d["tray"].map(tray_mean) - mu
    d["resid"] = d[value_col] - mu - d["alpha"] - d["beta"]

    ss_pos = np.sum(d["alpha"] ** 2)
    ss_tray = np.sum(d["beta"] ** 2)
    ss_res = np.sum(d["resid"] ** 2)
    ss_tot = ss_pos + ss_tray + ss_res
    if ss_tot <= 0:
        return None
    return {
        "위치(고정)%": 100 * ss_pos / ss_tot,
        "트레이(랏차이)%": 100 * ss_tray / ss_tot,
        "잔차(개별셀)%": 100 * ss_res / ss_tot,
    }


def plot_variance_gate(vc_temp, vc_ocv, outpath):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups = ["온도", "ΔOCV"]
    parts = ["위치(고정)%", "트레이(랏차이)%", "잔차(개별셀)%"]
    colors = ["#d1495b", "#edae49", "#66a182"]
    data = [vc_temp, vc_ocv]

    bottom = np.zeros(len(groups))
    for pi, p in enumerate(parts):
        vals = [(d[p] if d else 0) for d in data]
        ax.bar(groups, vals, bottom=bottom, label=p, color=colors[pi])
        for gi, v in enumerate(vals):
            if v > 4:
                ax.text(gi, bottom[gi] + v / 2, "%.0f%%" % v,
                        ha="center", va="center", fontsize=11, color="white")
        bottom += vals
    ax.set_ylabel("전체 변동에서 차지하는 비율(%)")
    ax.set_title("② '트레이 안 위치'가 변동을 얼마나 만드나? (핵심 판단)\n"
                 "빨강(위치)이 작으면 → 위치 보정으로 얻을 게 적다는 뜻", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  Phase 3 : 트레이별 구배(방향·세기) → 패턴 분류 → 개수
# ══════════════════════════════════════════════════════════════════════
def fit_trend_surface(rows, cols, z):
    """
    z ≈ a + b*x + c*y + d*r2   (x=열-중심, y=행-중심, r2=x^2+y^2-평균)
    numpy 최소제곱. 반환: (b, c, d, 설명력R2, fitted평면_range, 계수벡터)
    """
    x = cols - 5.5
    y = rows - 5.5
    r2 = x * x + y * y
    r2 = r2 - r2.mean()
    X = np.column_stack([np.ones_like(x), x, y, r2])
    coef, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    fitted = X.dot(coef)
    ss_res = np.sum((z - fitted) ** 2)
    ss_tot = np.sum((z - z.mean()) ** 2)
    r2_score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef[1], coef[2], coef[3], r2_score, (fitted.max() - fitted.min())


# 선형 기울기 8방향 이름 (기울기 벡터 각도 0°=오른쪽, 반시계로 45°씩)
#   h=+오른쪽, v=+위쪽 기준. 대각선(꼭짓점) 패턴 포함.
_DIR8 = {0: "오른쪽이 높음", 1: "오른쪽위가 높음", 2: "위쪽이 높음",
         3: "왼쪽위가 높음", 4: "왼쪽이 높음", 5: "왼쪽아래가 높음",
         6: "아래쪽이 높음", 7: "오른쪽아래가 높음"}


def classify_pattern(b, c, d, r2_score):
    """구배 계수 -> 사람이 읽는 패턴 이름.
    방향 정의: 열(col) 증가=오른쪽, 행(row) 증가=아래쪽 (히트맵 표시 기준).
    선형 기울기가 우세하면 8방향(축4+대각선4), 곡률이 우세하면 중앙/외곽."""
    if r2_score < PATTERN_R2_MIN:
        return "뚜렷한 패턴 없음(균일)"

    xs = np.arange(GRID) - 5.5
    X, Y = np.meshgrid(xs, xs)        # X=열(오른쪽+), Y=행(아래쪽+)
    var_lin = np.var(b * X + c * Y)   # 선형 기울기(대각선 포함)가 만드는 변동
    r2grid = X ** 2 + Y ** 2
    var_rad = np.var(d * (r2grid - r2grid.mean()))  # 곡률(중앙/외곽)이 만드는 변동

    if var_rad >= RADIAL_MARGIN * var_lin:   # 곡률이 '확실히' 우세할 때만 방사형
        return "바깥(외곽)이 높음" if d > 0 else "가운데(중앙)가 높음"

    # 선형 우세 → 기울기 벡터 각도로 8방향. h=오른쪽+, v=위쪽+(c>0=아래高 이므로 v=-c)
    ang = np.degrees(np.arctan2(-c, b))
    sector = int(round(ang / 45.0)) % 8
    return _DIR8[sector]


def robust_outlier_mask(values, k=OUTLIER_K):
    """MAD 기반 이상치(튀는 셀) 마스크. True=이상치. (median/MAD는 이상치에 강건)"""
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(v, dtype=bool)
    z = np.abs(v - med) / (1.4826 * mad)
    return z > k


def tray_grid(g, value_col, mask_outliers=True):
    """트레이 12x12 격자 값. mask_outliers=True면 튀는 셀은 NaN(흰색)으로 제거."""
    grid = np.full((GRID, GRID), np.nan)
    for _, r in g.iterrows():
        if not (np.isnan(r["row"]) or np.isnan(r["col"])):
            grid[int(r["row"]), int(r["col"])] = r[value_col]
    if mask_outliers:
        flat = grid.flatten()
        flat[robust_outlier_mask(flat)] = np.nan
        grid = flat.reshape(GRID, GRID)
    return grid


def robust_clim(grid):
    """튀는 셀에 눈금이 먹히지 않도록 2~98 백분위로 색 범위 설정."""
    finite = grid[np.isfinite(grid)]
    if finite.size < 3:
        return None, None
    lo, hi = np.nanpercentile(finite, [2, 98])
    if lo == hi:
        return None, None
    return lo, hi


def per_tray_patterns(df, value_col, keys):
    """트레이별 구배계수·패턴 표 반환. 튀는 셀(이상치)은 구배 피팅 전에 제거."""
    df = df.copy()
    df["_tray"] = keys.values
    recs = []
    for tray, g in df.groupby("_tray"):
        g = g.dropna(subset=[value_col, "row", "col"])
        if len(g) < MIN_VALID_CELLS:
            recs.append({"랏-트레이": tray, "유효셀": len(g), "이상셀수": 0,
                         "패턴": "데이터 부족", "b_좌우": np.nan,
                         "c_상하": np.nan, "d_중앙외곽": np.nan,
                         "설명력R2": np.nan, "구배크기": np.nan})
            continue
        # 튀는 셀 제거 후 구배 피팅 (한 셀 때문에 패턴이 안 보이는 문제 방지)
        vals = g[value_col].values.astype(float)
        out = robust_outlier_mask(vals)
        gg = g[~out]
        b, c, d, r2s, rng = fit_trend_surface(
            gg["row"].values.astype(float),
            gg["col"].values.astype(float),
            gg[value_col].values.astype(float),
        )
        recs.append({
            "랏-트레이": tray, "유효셀": len(g), "이상셀수": int(out.sum()),
            "패턴": classify_pattern(b, c, d, r2s),
            "b_좌우": b, "c_상하": c, "d_중앙외곽": d,
            "설명력R2": r2s, "구배크기": rng,
        })
    return pd.DataFrame(recs)


def pattern_counts(pat_df):
    """패턴별 트레이 개수/비율 표."""
    analyzable = pat_df[pat_df["패턴"] != "데이터 부족"]
    n = len(analyzable)
    vc = analyzable["패턴"].value_counts()
    out = pd.DataFrame({"트레이 수": vc, "비율(%)": (100 * vc / n).round(1)})
    return out.reset_index().rename(columns={"index": "패턴"})


def strong_summary(pat_df, thr=PATTERN_R2_STRONG):
    """강한 패턴(R²≥thr) 트레이 수 / 분석가능 트레이 수 반환.
    (신호가 약한지 = 과대분류 여부를 판단하는 지표)"""
    a = pat_df[pat_df["패턴"] != "데이터 부족"]
    n = len(a)
    strong = int((a["설명력R2"] >= thr).sum())
    return strong, n


def plot_pattern_menu(count_df, title, outpath, unit=""):
    order = count_df.sort_values("트레이 수", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * len(order) + 2)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(order)))
    ax.barh(order["패턴"], order["트레이 수"], color=colors)
    for i, (_, r) in enumerate(order.iterrows()):
        ax.text(r["트레이 수"], i, "  %d개 (%.1f%%)" % (r["트레이 수"], r["비율(%)"]),
                va="center", fontsize=10)
    ax.set_xlabel("트레이 수")
    ax.set_title(title, fontsize=12)
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_pattern_cards(df, value_col, pat_df, keys, outpath, title, unit=""):
    """패턴별 대표 트레이 1개씩 히트맵 카드."""
    df = df.copy()
    df["_tray"] = keys.values
    cats = [c for c in pat_df["패턴"].unique() if c != "데이터 부족"]
    cats = sorted(cats)
    if not cats:
        return
    ncol = min(3, len(cats))
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow),
                             squeeze=False)
    for k, cat in enumerate(cats):
        ax = axes[k // ncol][k % ncol]
        sub = pat_df[pat_df["패턴"] == cat].sort_values("설명력R2", ascending=False)
        if cat.startswith("뚜렷한 패턴 없음"):
            rep_row = sub.iloc[len(sub) // 2]      # 균일: 중앙값 트레이(진짜 평평한 예)
        else:
            rep_row = sub.iloc[0]                  # 그 외: 가장 뚜렷한(R² 최고) 예
        rep = rep_row["랏-트레이"]
        r2 = rep_row.get("설명력R2", np.nan)
        mag = rep_row.get("구배크기", np.nan)
        g = df[df["_tray"] == rep]
        grid = tray_grid(g, value_col, mask_outliers=True)  # 튀는 셀 제거
        vmin, vmax = robust_clim(grid)
        im = ax.imshow(grid, cmap="coolwarm", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title("%s\nR²=%.2f · 세기=%.2f\n(%s)" % (cat, r2, mag, str(rep)[:18]),
                     fontsize=8)
        ax.set_xticks([0, 11]); ax.set_xticklabels(["왼쪽", "오른쪽"], fontsize=7)
        ax.set_yticks([0, 11]); ax.set_yticklabels(["위", "아래"], fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)
    for k in range(len(cats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_exclusion(report, outpath):
    """제외 리포트: 라벨/값을 두 위치에 배치해 한글이어도 정렬 유지(monospace 미사용)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.03, 0.94, "◆ 데이터 정리 결과 (분석에서 뺀 셀)", fontsize=13,
            fontweight="bold", va="top")
    items = list(report.items())
    y = 0.80
    for k, v in items:
        ax.text(0.05, y, str(k), fontsize=12, va="top", ha="left")
        ax.text(0.95, y, "{:,} 셀".format(v), fontsize=12, va="top", ha="right")
        y -= 0.095
    keep = report.get("분석 대상", 0)
    tot = report.get("전체 셀", 1)
    ax.text(0.05, y - 0.02, "→ 전체의 %.1f%% 를 분석에 사용" % (100 * keep / tot),
            fontsize=12, va="top", color="#1f77b4", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  시간 변화 : 온도가 시점에 따라 어떻게 변하나 (OCV1→2→3)
# ══════════════════════════════════════════════════════════════════════
def variance_gate_multi(series_list, outpath):
    """series_list: [(라벨, vc_dict), ...] 를 누적막대로. 시점별 위치효과 비교."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [s[0] for s in series_list]
    parts = ["위치(고정)%", "트레이(랏차이)%", "잔차(개별셀)%"]
    colors = ["#d1495b", "#edae49", "#66a182"]
    bottom = np.zeros(len(labels))
    for pi, p in enumerate(parts):
        vals = [(s[1][p] if s[1] else 0) for s in series_list]
        ax.bar(labels, vals, bottom=bottom, label=p, color=colors[pi])
        for gi, v in enumerate(vals):
            if v > 4:
                ax.text(gi, bottom[gi] + v / 2, "%.0f%%" % v,
                        ha="center", va="center", color="white", fontsize=10)
        bottom += vals
    ax.set_ylabel("전체 변동에서 차지하는 비율(%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("② 트레이 안 '위치'가 변동을 얼마나 만드나 (시점별)\n"
                 "빨강(위치)이 작으면 → 위치 보정으로 얻을 게 적다는 뜻", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def temperature_evolution(df, outpath):
    """시간에 따른 온도 변화: 냉각커브 + 시점별 분포."""
    means = [df[k].mean() for _, k, _ in TIMEPOINTS]
    stds = [df[k].std() for _, k, _ in TIMEPOINTS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.errorbar([0, 1, 2], means, yerr=stds, marker="o", capsize=6, lw=2,
                color="#1f77b4")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([lbl for _, _, lbl in TIMEPOINTS])
    ax.set_ylabel("셀 온도(℃)")
    for i, m in enumerate(means):
        ax.text(i, m, "  %.2f℃" % m, va="bottom", fontsize=10)
    ax.set_title("평균 온도 변화 (막대=산포)")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, k, _), c in zip(TIMEPOINTS, colors):
        v = df[k].dropna()
        ax2.hist(v, bins=60, alpha=0.5, color=c,
                 label="%s (평균 %.2f℃)" % (name, v.mean()))
    ax2.set_xlabel("셀 온도(℃)")
    ax2.set_ylabel("셀 수")
    ax2.legend()
    ax2.set_title("시점별 온도 분포")

    fig.suptitle("④ 시간이 지나며 온도가 어떻게 변하나 (냉각 %.2f℃)"
                 % (means[0] - means[2]), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return means, stds


def pattern_compare_over_time(cnt_dict, outpath):
    """시점별 패턴 분포를 그룹막대로 (패턴이 시간에 따라 바뀌나)."""
    cats = sorted(set().union(*[set(c["패턴"]) for c in cnt_dict.values()]))
    x = np.arange(len(cats))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (name, cdf) in enumerate(cnt_dict.items()):
        m = dict(zip(cdf["패턴"], cdf["트레이 수"]))
        vals = [m.get(cat, 0) for cat in cats]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[i % 3])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("트레이 수")
    ax.legend(title="측정 시점")
    ax.set_title("⑤ 시점별 온도 패턴 분포 (패턴이 시간이 지나며 바뀌나?)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def pattern_persistence(pat1, pat2, pat3, outpath):
    """트레이 온도패턴이 3시점 내내 유지되나. (전이표, 유지율%, 1==3율%) 반환."""
    m = pat1[["랏-트레이", "패턴"]].rename(columns={"패턴": "OCV1"})
    m = m.merge(pat2[["랏-트레이", "패턴"]].rename(columns={"패턴": "OCV2"}), on="랏-트레이")
    m = m.merge(pat3[["랏-트레이", "패턴"]].rename(columns={"패턴": "OCV3"}), on="랏-트레이")
    ok = m[(m["OCV1"] != "데이터 부족") & (m["OCV2"] != "데이터 부족") &
           (m["OCV3"] != "데이터 부족")]
    n = max(len(ok), 1)
    same_all = (ok["OCV1"] == ok["OCV2"]) & (ok["OCV2"] == ok["OCV3"])
    same_13 = (ok["OCV1"] == ok["OCV3"])

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [int(same_all.sum()), int((~same_all).sum())]
    ax.bar(["3시점 내내\n같은 패턴", "도중에\n바뀜"], vals,
           color=["#66a182", "#d1495b"])
    for i, v in enumerate(vals):
        ax.text(i, v, "%d개 (%.1f%%)" % (v, 100 * v / n), ha="center",
                va="bottom", fontsize=11)
    ax.set_ylabel("트레이 수")
    ax.set_title("⑥ 온도패턴이 시간이 지나도 유지되나?\n"
                 "유지=설비/구조적 원인,  바뀜=냉각과정에서 변형", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return m, 100 * same_all.mean(), 100 * same_13.mean()


def plot_tray_evolution(df, keys, pat3, outpath, n_examples=3):
    """대표 트레이 몇 개의 T1/T2/T3/냉각량 히트맵을 나란히 (패턴 변화 눈으로)."""
    df = df.copy()
    df["_tray"] = keys.values
    df["_cool"] = df["t1"] - df["t3"]
    sub = pat3[pat3["패턴"] != "데이터 부족"].sort_values("설명력R2", ascending=False)
    picked, seen = [], set()
    for _, r in sub.iterrows():
        if r["패턴"] not in seen:
            picked.append(r["랏-트레이"])
            seen.add(r["패턴"])
        if len(picked) >= n_examples:
            break
    if not picked:
        return
    fields = [("OCV1 온도", "t1"), ("OCV2 온도", "t2"),
              ("OCV3 온도", "t3"), ("냉각량(T1-T3)", "_cool")]
    fig, axes = plt.subplots(len(picked), 4, figsize=(15, 3.6 * len(picked)),
                             squeeze=False)
    for ri, tray in enumerate(picked):
        g = df[df["_tray"] == tray]
        for ci, (name, col) in enumerate(fields):
            ax = axes[ri][ci]
            grid = tray_grid(g, col, mask_outliers=True)  # 튀는 셀 제거
            vmin, vmax = robust_clim(grid)
            im = ax.imshow(grid, cmap="coolwarm", origin="upper",
                           vmin=vmin, vmax=vmax)
            if ri == 0:
                ax.set_title(name, fontsize=10)
            if ci == 0:
                ax.set_ylabel(str(tray)[:18], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("⑨ 대표 트레이의 시간별 온도 변화 (왼→오: 시간경과, 맨오른쪽=냉각량)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  ΔOCV 패턴 vs 온도 패턴 : 트렌드가 일치하나?
# ══════════════════════════════════════════════════════════════════════
def plot_ocv_temp_alignment(pat_ocv, series, outpath, title=None):
    """ΔOCV 구배 방향과 각 시리즈 구배 방향의 정렬도(코사인).
    series: [(라벨, pat_df), ...]. 강한 ΔOCV 트레이만.
    +1=같은방향, 0=무관, -1=반대."""
    strong = pat_ocv[(pat_ocv["패턴"] != "데이터 부족") &
                     (pat_ocv["설명력R2"] >= PATTERN_R2_STRONG)]
    strong = strong[["랏-트레이", "b_좌우", "c_상하"]].rename(
        columns={"b_좌우": "bo", "c_상하": "co"})
    results = {}
    for label, pdf in series:
        t = pdf[["랏-트레이", "b_좌우", "c_상하"]].rename(
            columns={"b_좌우": "bt", "c_상하": "ct"})
        m = strong.merge(t, on="랏-트레이")
        dot = m["bo"] * m["bt"] + m["co"] * m["ct"]
        denom = np.sqrt(m["bo"]**2 + m["co"]**2) * np.sqrt(m["bt"]**2 + m["ct"]**2)
        cos = (dot / denom).replace([np.inf, -np.inf], np.nan).dropna()
        results[label] = cos

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(results.keys())
    means = [results[k].mean() if len(results[k]) else 0 for k in labels]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    ax.bar(labels, means, color=[palette[i % len(palette)] for i in range(len(labels))])
    for i, mv in enumerate(means):
        ax.text(i, mv, "%.2f" % mv, ha="center",
                va="bottom" if mv >= 0 else "top", fontsize=11)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("구배 방향 정렬도 (코사인)")
    ax.set_title(title or ("⑭ ΔOCV 구배와 온도/냉각량 구배가 같은 방향인가?\n"
                 "+1=같은방향 · 0=무관 · -1=반대  |  0 근처면 온도가 ΔOCV패턴을 설명 못함"),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return {k: (float(v.mean()) if len(v) else float("nan"), int(len(v)))
            for k, v in results.items()}


def plot_ocv_vs_temp_cards(df, keys, pat_ocv, outpath, max_rows=8):
    """강한 ΔOCV 패턴 대표트레이의 ΔOCV / 온도1/2/3 / 온도변화량 을 나란히 (모양 비교)."""
    df = df.copy()
    df["_tray"] = keys.values
    df["_cool"] = df["t1"] - df["t3"]
    cats = pat_ocv[pat_ocv["패턴"] != "데이터 부족"]
    reps = []
    for cat in cats["패턴"].unique():
        sub = cats[cats["패턴"] == cat].sort_values("설명력R2", ascending=False)
        if sub.iloc[0]["설명력R2"] >= PATTERN_R2_STRONG:
            reps.append((cat, sub.iloc[0]["랏-트레이"], sub.iloc[0]["설명력R2"]))
    reps.sort(key=lambda r: -r[2])
    reps = reps[:max_rows]
    if not reps:
        return
    cols = [("ΔOCV", "docv"), ("온도 OCV1", "t1"), ("온도 OCV2", "t2"),
            ("온도 OCV3", "t3"), ("온도변화량 ΔT(T1-T3)", "_cool")]
    fig, axes = plt.subplots(len(reps), len(cols),
                             figsize=(3.7 * len(cols), 3.4 * len(reps)),
                             squeeze=False)
    for ri, (cat, tray, r2) in enumerate(reps):
        g = df[df["_tray"] == tray]
        for ci, (cname, col) in enumerate(cols):
            ax = axes[ri][ci]
            grid = tray_grid(g, col, mask_outliers=True)
            vmin, vmax = robust_clim(grid)
            im = ax.imshow(grid, cmap="coolwarm", origin="upper",
                           vmin=vmin, vmax=vmax)
            if ri == 0:
                ax.set_title(cname, fontsize=10)
            if ci == 0:
                ax.set_ylabel("%s\nR²=%.2f" % (cat, r2), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("⑮ ΔOCV 패턴별 대표트레이: ΔOCV vs 온도1/2/3 + 온도변화량 (왼쪽 ΔOCV와 오른쪽들이 닮았나?)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def within_tray_corr(df, keys, ocv_specs, temp_specs, title, outpath):
    """행(OCV류) × 열(온도류) 트레이 내 상관 히트맵 (트레이 평균 제거).
    ocv_specs/temp_specs: [(표시이름, 컬럼명), ...]. 트레이 offset 오염 제거용."""
    df = df.copy()
    df["_tray"] = keys.values
    cols = list({c for _, c in ocv_specs} | {c for _, c in temp_specs})
    for c in cols:
        df[c + "_w"] = df[c] - df.groupby("_tray")[c].transform("mean")

    onames = [n for n, _ in ocv_specs]
    tnames = [n for n, _ in temp_specs]
    mat = np.full((len(ocv_specs), len(temp_specs)), np.nan)
    for i, (_, oc) in enumerate(ocv_specs):
        for j, (_, tc) in enumerate(temp_specs):
            x = df[oc + "_w"].values
            y = df[tc + "_w"].values
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() > 10 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                mat[i, j] = np.corrcoef(x[m], y[m])[0, 1]

    fig, ax = plt.subplots(figsize=(1.6 * len(tnames) + 3, 1.1 * len(onames) + 3))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tnames))); ax.set_xticklabels(tnames, fontsize=10,
                                                          rotation=20, ha="right")
    ax.set_yticks(range(len(onames))); ax.set_yticklabels(onames, fontsize=10)
    for i in range(len(onames)):
        for j in range(len(tnames)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center",
                        fontsize=11,
                        color="white" if abs(mat[i, j]) > 0.5 else "black")
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="상관계수")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return pd.DataFrame(mat, index=onames, columns=tnames)


def ocv_temp_correlation(df, keys, outpath):
    """[16] 각 OCV(1/2/3/ΔOCV) × 각 온도(T1/2/3/ΔT) 트레이 내 상관."""
    df = df.copy()
    df["_cool"] = df["t1"] - df["t3"]
    ocv = [("OCV1", "ocv1"), ("OCV2", "ocv2"), ("OCV3", "ocv3"), ("ΔOCV", "docv")]
    tmp = [("온도T1", "t1"), ("온도T2", "t2"), ("온도T3", "t3"), ("냉각량ΔT", "_cool")]
    return within_tray_corr(df, keys, ocv, tmp,
                            "[16] 각 OCV × 각 온도 상관 (트레이 내, 평균제거)\n"
                            "0 근처면 그 셀 온도가 OCV를 거의 안 흔든다는 뜻", outpath)


# ══════════════════════════════════════════════════════════════════════
#  검증 & 구배 제거(detrend)
# ══════════════════════════════════════════════════════════════════════
def verify_docv(df):
    """ΔOCV 컬럼이 정말 OCV1-OCV3 인지 검증 (정의 불일치 조기 발견)."""
    calc = df["ocv1"] - df["ocv3"]
    diff = (df["docv"] - calc).abs()
    n_bad = int((diff > 0.01).sum())
    print("[검증] ΔOCV 컬럼 vs (OCV1-OCV3): 중앙값차=%.4f, 최대차=%.4f, "
          "불일치(>0.01) %d셀" % (diff.median(), diff.max(), n_bad))
    if n_bad > len(df) * 0.01:
        print("   ⚠ ΔOCV 정의가 OCV1-OCV3가 아닐 수 있습니다. 컬럼 정의 확인 필요!")


def time_order_diagnostic(df, keys):
    """트레이 내 측정시각 산포 진단: 셀별 순차측정이면 '시각=위치' 교란 가능."""
    if not df["st1_dt"].notna().any():
        print("[시각진단] 측정시각 없음 → 생략")
        return
    span = df.groupby(keys.values)["st1_dt"].agg(lambda s: (s.max() - s.min()).total_seconds())
    med = float(span.median())
    print("[시각진단] 트레이 내 OCV1 시각 산포(중앙값): %.1f초" % med)
    if med < 1:
        print("   → 트레이 단위 일괄측정: 측정순서로 인한 위치 교란 없음")
    else:
        print("   ⚠ 셀별 순차측정 흔적: '위치 구배'가 측정순서(완화 진행차)일 수 있음")


def detrend_docv(df, keys, outdir):
    """트레이별 트렌드면(구배)을 빼서 보정 ΔOCV 산출.
    보정값 = (ΔOCV - 트렌드면) + 트레이중앙값  → 구배만 제거, 스케일 유지.
    셀별 결과 CSV 저장 + 전/후 비교 그림([17])."""
    df = df.copy()
    df["_tray"] = keys.values
    corrected = np.full(len(df), np.nan)
    trend_all = np.full(len(df), np.nan)
    rec_before, rec_after = [], []

    for tray, g in df.groupby("_tray"):
        idx = g.index
        v = g["docv"].values.astype(float)
        r = g["row"].values.astype(float)
        c = g["col"].values.astype(float)
        med = np.nanmedian(v)
        if len(g) < MIN_VALID_CELLS:
            corrected[df.index.get_indexer(idx)] = v - med + med
            continue
        out = robust_outlier_mask(v)
        # 이상치 제외하고 트렌드면 피팅 → 전체 셀에 트렌드 적용
        b, cc, d, r2b, rngb = fit_trend_surface(r[~out], c[~out], v[~out])
        x = c - 5.5; y = r - 5.5
        r2t = x * x + y * y
        trend = b * x + cc * y + d * (r2t - r2t.mean())
        resid = v - trend
        pos = df.index.get_indexer(idx)
        corrected[pos] = resid            # 트렌드 제거(중앙값은 resid에 포함됨)
        trend_all[pos] = trend
        # 전/후 구배 세기 (검증: 후는 0 근처여야 함)
        _, _, _, r2a, rnga = fit_trend_surface(r[~out], c[~out], resid[~out])
        rec_before.append((r2b, rngb)); rec_after.append((r2a, rnga))

    df["ΔOCV_트렌드"] = trend_all
    df["ΔOCV_보정"] = corrected

    # 전/후 비교 그림
    rb = np.array(rec_before); ra = np.array(rec_after)
    sb = int((rb[:, 0] >= PATTERN_R2_STRONG).sum())
    sa = int((ra[:, 0] >= PATTERN_R2_STRONG).sum())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.bar(["보정 전", "보정 후"], [sb, sa], color=["#d1495b", "#66a182"])
    for i, v_ in enumerate([sb, sa]):
        ax.text(i, v_, "%d개" % v_, ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("강한 구배(R²≥%.2f) 트레이 수" % PATTERN_R2_STRONG)
    ax.set_title("구배 있는 트레이가 사라졌나")
    ax2 = axes[1]
    ax2.hist(rb[:, 1], bins=40, alpha=0.6, label="보정 전", color="#d1495b")
    ax2.hist(ra[:, 1], bins=40, alpha=0.6, label="보정 후", color="#66a182")
    ax2.set_xlabel("트레이 구배 세기 (ΔOCV 단위)")
    ax2.set_ylabel("트레이 수")
    ax2.legend()
    ax2.set_title("구배 세기 분포 (0으로 몰리면 성공)")
    fig.suptitle("[17] 구배 제거(detrend) 효과: ΔOCV → ΔOCV_보정", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, "17_구배제거_효과.png"), dpi=130)
    plt.close(fig)

    # 셀별 결과 저장 (CSV, 엑셀에서 바로 열림)
    out_cols = ["_tray", COL["cellno"], "row", "col", "docv",
                "ΔOCV_트렌드", "ΔOCV_보정"]
    if COL["label"] in df.columns:
        out_cols.insert(2, COL["label"])
    csv_path = os.path.join(outdir, "셀별_ΔOCV_보정.csv")
    df[out_cols].rename(columns={"_tray": "랏-트레이", "docv": "ΔOCV_원본"}) \
        .to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("[detrend] 강한구배 트레이 %d → %d개, 셀별 보정값: %s"
          % (sb, sa, csv_path))
    return df


# ══════════════════════════════════════════════════════════════════════
#  전공정(방전7) + OCV7 : 열이력 / 완화 분석
# ══════════════════════════════════════════════════════════════════════
DISCHARGE_KEYS = ["dis_tmin", "dis_tavg", "dis_tmax", "ocv7"]


def has_discharge(df):
    return all(COL[k] in df.columns for k in DISCHARGE_KEYS)


def build_discharge_features(df):
    """방전7 온도(min/avg/max)·스윙, OCV7, OCV7 시각 파생."""
    for k in ["dis_tmin", "dis_tavg", "dis_tmax", "ocv7"]:
        if COL[k] in df.columns:
            df[k] = pd.to_numeric(df[COL[k]], errors="coerce")
    if "dis_tmax" in df and "dis_tmin" in df:
        df["dis_swing"] = df["dis_tmax"] - df["dis_tmin"]
    df["st7_dt"] = (pd.to_datetime(df[COL["st7"]], errors="coerce")
                    if COL["st7"] in df.columns else pd.NaT)
    return df


def check_percell_independence(df, keys, colmap):
    """방전 온도가 셀별 독립 측정인지: 트레이 내 고유값 개수로 판별."""
    d = df.copy()
    d["_tray"] = keys.values
    for label, col in colmap:
        if col not in d:
            continue
        nun = d.groupby("_tray")[col].nunique()
        med = float(nun.median()) if len(nun) else 0
        flag = "OK(셀별 독립)" if med > 10 else "⚠ 공용센서 의심(트레이 내 값 거의 동일→공간분석 무의미)"
        print("   %-10s 트레이 내 고유값(중앙값) %.0f → %s" % (label, med, flag))


def plot_gradient_birth(pat_map, outpath):
    """구배 '탄생 시점' 추적: 구간별 구배 세기(중앙값) 막대.
    pat_map: [(라벨, pat_df), ...] 순서(OCV7 → 구간증분 → ΔOCV)."""
    labels, meds = [], []
    for label, p in pat_map:
        a = p[p["패턴"] != "데이터 부족"]["구배크기"].dropna()
        labels.append(label)
        meds.append(float(a.median()) if len(a) else np.nan)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, meds, color="#4c72b0")
    for i, mv in enumerate(meds):
        if not np.isnan(mv):
            ax.text(i, mv, "%.3f" % mv, ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("트레이 구배 세기 중앙값 (전압 단위)")
    ax.set_title("[21] 구배는 언제 생기나 (구간별 공간구배 세기)\n"
                 "OCV7에서 이미 크면=충방전공정, 특정 에이징 구간에서 커지면=완화",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def relaxation_fit(df, keys, outdir):
    """4점(OCV7,1,2,3) 완화모델: V(t)=V∞ + A·exp(-t/τ) + k·t.
    τ는 전역 그리드서치로 고정 → 셀별 (V∞, A, k) 선형 최소제곱.
    V∞=참 OCV, k=순수 자가방전 기울기. 셀별 CSV + [22] 분포 그림."""
    times = np.array([0.0, 1.0, 3.0, 4.0])   # OCV7=0, OCV1=1일, OCV2=3일, OCV3=4일
    V = df[["ocv7", "ocv1", "ocv2", "ocv3"]].values.astype(float)
    ok = ~np.isnan(V).any(axis=1)
    if ok.sum() < 50:
        print("[완화모델] 4점 유효셀 부족 → 생략")
        return df, None

    best = None
    for tau in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        basis = np.column_stack([np.ones(4), np.exp(-times / tau), times])
        coef, _, _, _ = np.linalg.lstsq(basis, V[ok].T, rcond=None)
        sse = np.nansum((V[ok].T - basis.dot(coef)) ** 2)
        if best is None or sse < best[0]:
            best = (sse, tau, basis, coef)
    _, tau, basis, coef = best

    Vinf = np.full(len(df), np.nan)
    k = np.full(len(df), np.nan)
    Vinf[ok] = coef[0]
    k[ok] = coef[2]
    df["OCV_Vinf"] = Vinf
    df["자가방전_k"] = k

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    kk = k[ok]
    axes[0].hist(kk, bins=60, color="#c44e52")
    axes[0].set_xlabel("자가방전 기울기 k (전압/일)")
    axes[0].set_ylabel("셀 수")
    axes[0].set_title("k 분포 (완화 제거된 순수 자가방전)\n오른쪽/왼쪽 꼬리 = 불량 후보")
    axes[1].scatter(Vinf[ok], kk, s=3, alpha=0.3)
    axes[1].set_xlabel("V∞ (참 OCV)")
    axes[1].set_ylabel("자가방전 기울기 k")
    axes[1].set_title("V∞ vs k")
    fig.suptitle("[22] 완화모델 분리 (τ=%.1f일 고정): V∞·k" % tau, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "22_완화모델_Vinf_k.png"), dpi=130)
    plt.close(fig)

    out = df.copy()
    out["_tray"] = keys.values
    cols = ["_tray", COL["cellno"], "ocv7", "ocv1", "ocv2", "ocv3",
            "OCV_Vinf", "자가방전_k"]
    csv = os.path.join(outdir, "셀별_완화모델.csv")
    out[cols].rename(columns={"_tray": "랏-트레이"}).to_csv(
        csv, index=False, encoding="utf-8-sig")
    print("[완화모델] τ=%.1f일, k중앙값=%.4f, 셀별: %s" % (tau, np.nanmedian(kk), csv))
    return df, tau


def pick_rep_trays(pat_ocv, n=4):
    """ΔOCV 패턴이 가장 뚜렷한(R² 최고) 대표 트레이 n개."""
    sub = pat_ocv[pat_ocv["패턴"] != "데이터 부족"].sort_values(
        "설명력R2", ascending=False)
    return [(r["랏-트레이"], "ΔOCV R²=%.2f" % r["설명력R2"])
            for _, r in sub.head(n).iterrows()]


def plot_field_evolution(df, keys, reps, fields, suptitle, outpath):
    """대표 트레이들의 여러 필드(OCV7→1→2→3 또는 온도)를 나란히 히트맵.
    reps: [(랏-트레이, 주석)], fields: [(표시이름, 컬럼)]."""
    if not reps:
        return
    df = df.copy()
    df["_tray"] = keys.values
    fig, axes = plt.subplots(len(reps), len(fields),
                             figsize=(3.6 * len(fields), 3.4 * len(reps)),
                             squeeze=False)
    for ri, (tray, ann) in enumerate(reps):
        g = df[df["_tray"] == tray]
        for ci, (cname, col) in enumerate(fields):
            ax = axes[ri][ci]
            grid = tray_grid(g, col, mask_outliers=True)
            vmin, vmax = robust_clim(grid)
            im = ax.imshow(grid, cmap="coolwarm", origin="upper",
                           vmin=vmin, vmax=vmax)
            if ri == 0:
                ax.set_title(cname, fontsize=10)
            if ci == 0:
                ax.set_ylabel("%s\n%s" % (str(tray)[:16], ann), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def detect_shared_fingerprint(df, keys, value_col, title, outpath):
    """[27] 공유 지문 탐지: 트레이 공간필드의 '공통 필드'와의 상관 분포.
    공통필드 = 트레이 간 중앙값(가끔 튀는 셀에 강건).
    ★이상치 마스킹 OFF(계통 오프셋을 스스로 지우지 않도록),
      위치별 유효 표본 수를 함께 보여 코너/특정칸이 표본부족 아티팩트인지 판별."""
    df = df.copy()
    df["_tray"] = keys.values
    vecs = []
    for _, g in df.groupby("_tray"):
        grid = tray_grid(g, value_col, mask_outliers=False)  # 계통 오프셋 보존
        vec = grid.flatten()
        if np.sum(~np.isnan(vec)) < MIN_VALID_CELLS:
            continue
        vecs.append(vec - np.nanmedian(vec))
    if len(vecs) < 5:
        print("[지문] 트레이 부족 → 생략")
        return None
    M = np.array(vecs)
    count = np.sum(~np.isnan(M), axis=0)          # 위치별 유효 트레이 수
    min_count = max(5, int(0.2 * len(vecs)))       # 20% 미만인 칸은 못 믿음
    commonfield = np.nanmedian(M, axis=0)          # 트레이 간 중앙값(강건)
    commonfield[count < min_count] = np.nan        # 표본부족 칸 제외

    cors = []
    for row in M:
        m = ~(np.isnan(row) | np.isnan(commonfield))
        if m.sum() > 50 and np.nanstd(row[m]) > 0 and np.nanstd(commonfield[m]) > 0:
            cors.append(np.corrcoef(row[m], commonfield[m])[0, 1])
    cors = np.array(cors)
    if len(cors) < 5:   # 변동 없는 신호(예: 상수 온도) → 지문 정의 불가
        print("[지문] %s 유효 상관 부족 → 생략" % value_col)
        plt.close("all")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cf = commonfield.reshape(GRID, GRID)
    lo, hi = robust_clim(cf)
    im0 = axes[0].imshow(cf, cmap="coolwarm", origin="upper", vmin=lo, vmax=hi)
    axes[0].set_title("공통 필드 (트레이 간 중앙값 = 지문)")
    axes[0].set_xticks([0, 11]); axes[0].set_xticklabels(["왼쪽", "오른쪽"], fontsize=8)
    axes[0].set_yticks([0, 11]); axes[0].set_yticklabels(["위", "아래"], fontsize=8)
    fig.colorbar(im0, ax=axes[0], shrink=0.7)
    im1 = axes[1].imshow(count.reshape(GRID, GRID), cmap="viridis", origin="upper")
    axes[1].set_title("위치별 유효 트레이 수\n(코너/특정칸이 낮으면 그 칸 지문은 못 믿음)")
    axes[1].set_xticks([0, 11]); axes[1].set_xticklabels(["왼쪽", "오른쪽"], fontsize=8)
    axes[1].set_yticks([0, 11]); axes[1].set_yticklabels(["위", "아래"], fontsize=8)
    fig.colorbar(im1, ax=axes[1], shrink=0.7)
    axes[2].hist(cors, bins=40, color="#4c72b0")
    axes[2].axvline(np.median(cors), color="k", ls="--",
                    label="중앙값 %.2f" % np.median(cors))
    axes[2].set_xlabel("각 트레이 ↔ 공통필드 상관")
    axes[2].set_ylabel("트레이 수")
    axes[2].legend()
    axes[2].set_title("1 근처 몰림=공유 지문(설비/계통 의심)\n넓게 퍼짐=트레이 고유(물리)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print("[지문] %s 공통필드 상관 중앙값=%.2f (유효표본 최소 %d)"
          % (value_col, np.median(cors), int(count.min())))
    return float(np.median(cors))


def fingerprint_all(df, keys, specs, outdir):
    """[27] 여러 신호(OCV7/1/2/3/ΔOCV/온도 등) 각각에 공유지문 탐지 + 요약 비교.
    개별 그림은 results/지문/ 에, 요약 막대는 메인에 저장."""
    sub = os.path.join(outdir, "지문")
    os.makedirs(sub, exist_ok=True)
    results = {}
    for label, col in specs:
        if col not in df.columns:
            continue
        med = detect_shared_fingerprint(df, keys, col,
            "[27] %s 공유 지문 (설비/계통 vs 물리)" % label,
            os.path.join(sub, "지문_%s.png" % label))
        if med is not None and not np.isnan(med):
            results[label] = med
    if not results:
        return results

    labels = list(results.keys())
    vals = [results[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(labels) + 2), 5.5))
    ax.bar(labels, vals,
           color=["#c44e52" if v >= 0.5 else "#4c72b0" for v in vals])
    for i, v in enumerate(vals):
        ax.text(i, v, "%.2f" % v, ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="0.5 (공유지문 기준선)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(min(-0.2, min(vals) - 0.1), 1)
    ax.set_ylabel("공통필드 상관 중앙값 (높을수록 공유 지문 강함)")
    ax.set_title("[27] 신호별 공유 지문 강도 비교\n"
                 "OCV1/2/3만 높고 OCV7 낮으면=OCV측정기 핀맵 · 온도가 높으면=열/위치",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "27_지문_요약.png"), dpi=130)
    plt.close(fig)
    print("[지문요약]", {k: round(v, 2) for k, v in results.items()})
    return results


def plot_tray_gallery(df, keys, value_col, outdir, label, n=100, per_page=25, seed=42):
    """개별 트레이 공간필드(중앙값 제거, 공통스케일)를 랜덤 n개 그려 페이지 저장.
    공통필드(지문)가 개별 트레이에 실제로 있는지 육안 검증용."""
    df = df.copy()
    df["_tray"] = keys.values
    grids, trays = {}, []
    for tray, g in df.groupby("_tray"):
        grid = tray_grid(g, value_col, mask_outliers=False)
        if np.sum(~np.isnan(grid)) < MIN_VALID_CELLS:
            continue
        grids[tray] = grid - np.nanmedian(grid)
        trays.append(tray)
    if not trays:
        print("[갤러리] 트레이 없음 → 생략")
        return
    rng = np.random.RandomState(seed)
    sel = list(rng.choice(trays, min(n, len(trays)), replace=False))
    pooled = np.concatenate([grids[t].flatten() for t in sel])
    pooled = pooled[np.isfinite(pooled)]
    v = np.nanpercentile(np.abs(pooled), 95) if pooled.size else 1.0
    v = v if v > 0 else 1.0

    ncol = nrow = int(np.sqrt(per_page))
    pages = int(np.ceil(len(sel) / per_page))
    sub = os.path.join(outdir, "갤러리")
    os.makedirs(sub, exist_ok=True)
    for p in range(pages):
        chunk = sel[p * per_page:(p + 1) * per_page]
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 15.5))
        im = None
        for i in range(nrow * ncol):
            ax = axes[i // ncol][i % ncol]
            if i < len(chunk):
                im = ax.imshow(grids[chunk[i]], cmap="coolwarm", origin="upper",
                               vmin=-v, vmax=v)
                ax.set_title(str(chunk[i])[:18], fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.axis("off")
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
        fig.suptitle("[27b] 트레이별 %s 필드 (중앙값제거, 공통스케일 ±%.3f) — %d/%d페이지"
                     % (label, v, p + 1, pages), fontsize=13)
        fig.savefig(os.path.join(sub, "갤러리_%s_p%d.png" % (label, p + 1)), dpi=110)
        plt.close(fig)
    print("[갤러리] %s: 랜덤 %d트레이 %d페이지 → %s" % (label, len(sel), pages, sub))


def rest_time_diagnostic(df, keys, outpath):
    """[28] 실제 휴지시간(OCV7→OCV1) 분포 + 위치 구조 + ΔOCV 트레이내 상관."""
    if not (df["st7_dt"].notna().any() and df["st1_dt"].notna().any()):
        print("[휴지시간] OCV7/OCV1 시각 없음 → 생략")
        return
    d = df.copy()
    d["_tray"] = keys.values
    d["_rest"] = (d["st1_dt"] - d["st7_dt"]).dt.total_seconds() / 3600.0
    for c in ["_rest", "docv"]:
        d[c + "_w"] = d[c] - d.groupby("_tray")[c].transform("mean")
    x = d["_rest_w"].values; y = d["docv_w"].values
    m = ~(np.isnan(x) | np.isnan(y))
    corr = (np.corrcoef(x[m], y[m])[0, 1]
            if m.sum() > 10 and np.std(x[m]) > 0 else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(d["_rest"].dropna(), bins=60, color="#55a868")
    axes[0].set_xlabel("실제 휴지시간 OCV7→OCV1 (시간)")
    axes[0].set_ylabel("셀 수")
    axes[0].set_title("휴지시간 분포 (중앙값 %.1fh)" % np.nanmedian(d["_rest"]))
    dd = d.dropna(subset=["_rest", "row", "col"])
    grid = np.full((GRID, GRID), np.nan)
    for (r, c), gg in dd.groupby(["row", "col"]):
        grid[int(r), int(c)] = gg["_rest"].mean()
    im = axes[1].imshow(grid, cmap="viridis", origin="upper")
    axes[1].set_title("위치별 평균 휴지시간\n(위치 구조 있으면 측정순서 교란)")
    axes[1].set_xticks([0, 11]); axes[1].set_xticklabels(["왼쪽", "오른쪽"], fontsize=8)
    axes[1].set_yticks([0, 11]); axes[1].set_yticklabels(["위", "아래"], fontsize=8)
    fig.colorbar(im, ax=axes[1], shrink=0.7)
    fig.suptitle("[28] 실제 휴지시간 & ΔOCV 교란 (트레이내 상관 r=%.2f)" % corr,
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print("[휴지시간] 중앙값=%.1fh, ΔOCV와 트레이내 상관=%.2f"
          % (np.nanmedian(d["_rest"]), corr))


# ══════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="트레이 온도/OCV 공간 패턴 분석기")
    ap.add_argument("--input", "-i", required=True, help="입력 엑셀(.xlsx)")
    ap.add_argument("--outdir", "-o", default="results", help="출력 폴더")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 60)
    print(" 트레이 공간 패턴 분석 시작")
    print("=" * 60)

    # Phase 0 ---------------------------------------------------------
    df = load_data(args.input)
    df = add_positions(df)
    df, report = quality_filter(df)
    print("\n[Phase 0] 품질필터 결과")
    for k, v in report.items():
        print("   %-28s : %d" % (k, v))
    keys = tray_key(df)
    n_trays = keys.nunique()
    print("   분석 대상 랏-트레이 수 : %d" % n_trays)
    plot_exclusion(report, os.path.join(args.outdir, "01_제외리포트.png"))

    df = build_time_features(df)
    df = build_discharge_features(df)
    verify_docv(df)
    time_order_diagnostic(df, keys)

    # Phase 1 ---------------------------------------------------------
    print("\n[Phase 1] 관계 매트릭스")
    corr = relationship_matrix(df, os.path.join(args.outdir, "02_관계표.png"))

    # Phase 2 : 위치효과 분산 게이트 (시점별) -------------------------
    print("\n[Phase 2] 위치효과 분산 게이트 (시점별)")
    vc = {}
    for name, k, _ in TIMEPOINTS:
        vc[name] = variance_components(df, k, keys)
    vc["ΔOCV"] = variance_components(df, "docv", keys)
    for name in ["OCV1", "OCV2", "OCV3", "ΔOCV"]:
        if vc[name]:
            print("   %-6s:" % name, {kk: round(v, 1) for kk, v in vc[name].items()})
    variance_gate_multi([(n, vc[n]) for n in ["OCV1", "OCV2", "OCV3", "ΔOCV"]],
                        os.path.join(args.outdir, "03_분산게이트.png"))

    # 시간에 따른 온도 변화 ------------------------------------------
    print("\n[Phase 2b] 시간에 따른 온도 변화")
    means, _ = temperature_evolution(df, os.path.join(args.outdir,
                                                       "04_온도_시간변화.png"))
    print("   평균온도 OCV1/2/3 = %.2f / %.2f / %.2f ℃  (총 냉각 %.2f℃)"
          % (means[0], means[1], means[2], means[0] - means[2]))

    # Phase 3 : 시점별 온도 패턴 + 시간변화 --------------------------
    print("\n[Phase 3] 시점별 온도 패턴 분류")
    pat, cnt = {}, {}
    for name, k, _ in TIMEPOINTS:
        pat[name] = per_tray_patterns(df, k, keys)
        cnt[name] = pattern_counts(pat[name])
        s, tot = strong_summary(pat[name])
        print("\n   [%s 온도 패턴]  강한패턴(R²≥%.2f): %d/%d (%.1f%%)"
              % (name, PATTERN_R2_STRONG, s, tot, 100 * s / max(tot, 1)))
        print(cnt[name].to_string(index=False))

    pattern_compare_over_time(cnt, os.path.join(args.outdir,
                                                "05_온도패턴_시점비교.png"))
    persist, pct_all, pct_13 = pattern_persistence(
        pat["OCV1"], pat["OCV2"], pat["OCV3"],
        os.path.join(args.outdir, "06_패턴유지.png"))
    print("\n   온도패턴 3시점 내내 유지: %.1f%%,  OCV1==OCV3: %.1f%%"
          % (pct_all, pct_13))
    plot_tray_evolution(df, keys, pat["OCV3"],
                        os.path.join(args.outdir, "09_트레이_시간진화_예시.png"))

    # 시점별 온도 패턴 대표 트레이 카드 (OCV 대표트레이의 온도 버전)
    for i, (name, k, _) in enumerate(TIMEPOINTS):
        plot_pattern_cards(
            df, k, pat[name], keys,
            os.path.join(args.outdir, "1%d_온도패턴_대표트레이_%s.png" % (i + 1, name)),
            "온도(%s) 패턴별 대표 트레이 (12x12, 튀는 셀 제거)" % name)

    # 냉각량(T1-T3) 패턴 ---------------------------------------------
    df["_cool"] = df["t1"] - df["t3"]
    pat_cool = per_tray_patterns(df, "_cool", keys)
    cnt_cool = pattern_counts(pat_cool)
    print("\n   [냉각량(T1-T3) 패턴]")
    print(cnt_cool.to_string(index=False))
    plot_pattern_menu(cnt_cool, "⑦ 냉각량(T1-T3) 패턴별 트레이 개수",
                      os.path.join(args.outdir, "07_냉각량패턴.png"))

    # ΔOCV 패턴 ------------------------------------------------------
    pat_ocv = per_tray_patterns(df, "docv", keys)
    cnt_ocv = pattern_counts(pat_ocv)
    s_ocv, tot_ocv = strong_summary(pat_ocv)
    print("\n   [ΔOCV 패턴]  강한패턴(R²≥%.2f): %d/%d (%.1f%%)"
          % (PATTERN_R2_STRONG, s_ocv, tot_ocv, 100 * s_ocv / max(tot_ocv, 1)))
    print(cnt_ocv.to_string(index=False))
    plot_pattern_menu(cnt_ocv, "⑧ ΔOCV 패턴별 트레이 개수",
                      os.path.join(args.outdir, "08_OCV패턴.png"))
    plot_pattern_cards(df, "docv", pat_ocv, keys,
                       os.path.join(args.outdir, "10_OCV패턴_대표트레이.png"),
                       "ΔOCV 패턴별 대표 트레이 (12x12, 튀는 셀 제거)")

    # ΔOCV 패턴 vs 온도/냉각량 트렌드 일치 여부 ----------------------
    align_series = [("OCV1온도", pat["OCV1"]), ("OCV2온도", pat["OCV2"]),
                    ("OCV3온도", pat["OCV3"]), ("냉각량ΔT", pat_cool)]
    align = plot_ocv_temp_alignment(pat_ocv, align_series,
                                    os.path.join(args.outdir, "14_OCV_온도_정렬도.png"))
    n_strong = next(iter(align.values()))[1]
    print("\n   ΔOCV 구배 vs 온도/냉각량 구배 정렬도(코사인, 강한ΔOCV %d트레이):" % n_strong)
    for nm, _ in align_series:
        print("      %-8s: %.2f" % (nm, align[nm][0]))
    plot_ocv_vs_temp_cards(df, keys, pat_ocv,
                           os.path.join(args.outdir, "15_OCV패턴별_온도비교.png"))

    # 각 OCV × 각 온도 상관 (트레이 내) ------------------------------
    corr_ot = ocv_temp_correlation(df, keys,
                                   os.path.join(args.outdir, "16_OCV_온도_상관.png"))
    print("\n   [각 OCV × 각 온도 상관 (트레이 내)]")
    print(corr_ot.round(2).to_string())

    # 구배 제거(detrend) → 보정 ΔOCV 산출 -----------------------------
    print("\n[Phase 5] 구배 제거(detrend)")
    detrend_docv(df, keys, args.outdir)

    # Phase 6 : 전공정(방전7)+OCV7 열이력/완화 (선택 컬럼 있을 때만) ----
    corr_dis = None
    if has_discharge(df):
        print("\n[Phase 6] 전공정(방전7)+OCV7 열이력/완화 분석")
        # 사전체크: 방전온도가 셀별 독립인가
        check_percell_independence(df, keys,
            [("방전Tmin", "dis_tmin"), ("방전Tavg", "dis_tavg"),
             ("방전Tmax", "dis_tmax")])

        # 파생: 완화 구간 증분
        df["_inc01"] = df["ocv1"] - df["ocv7"]   # 방전직후→1일 (최대완화)
        df["_inc12"] = df["ocv2"] - df["ocv1"]
        df["_inc23"] = df["ocv3"] - df["ocv2"]

        # [18] 방전온도 패턴 (Tmax 대표) + 강한패턴 요약
        pat_dtmax = per_tray_patterns(df, "dis_tmax", keys)
        pat_dtavg = per_tray_patterns(df, "dis_tavg", keys)
        pat_dswing = per_tray_patterns(df, "dis_swing", keys)
        for lb, pp in [("방전Tmax", pat_dtmax), ("방전Tavg", pat_dtavg),
                       ("방전스윙", pat_dswing)]:
            ss, tt = strong_summary(pp)
            print("   %s 강한패턴(R²≥%.2f): %d/%d" % (lb, PATTERN_R2_STRONG, ss, tt))
        plot_pattern_menu(pattern_counts(pat_dtmax), "[18] 방전 Tmax 패턴별 트레이 개수",
                          os.path.join(args.outdir, "18_방전Tmax패턴.png"))

        # [19] 정렬도 확장: ΔOCV 구배 vs 방전온도/OCV7/완화 구배 방향
        pat_ocv7 = per_tray_patterns(df, "ocv7", keys)
        pat_inc01 = per_tray_patterns(df, "_inc01", keys)
        align_dis = plot_ocv_temp_alignment(pat_ocv,
            [("방전Tmax", pat_dtmax), ("방전Tavg", pat_dtavg), ("방전스윙", pat_dswing),
             ("OCV7", pat_ocv7), ("완화OCV1-7", pat_inc01)],
            os.path.join(args.outdir, "19_방전_정렬도.png"),
            title="[19] ΔOCV 구배가 무엇과 같은 방향인가 (방전온도·OCV7·완화)\n"
                  "높으면=열이력/공정 원인 · 0이면 무관")
        print("   [19] ΔOCV vs 방전요인 정렬도:",
              {k: round(v[0], 2) for k, v in align_dis.items()})

        # [20] 상관 확장: 각 OCV × 방전온도(트레이 내)
        ocv_specs = [("OCV7", "ocv7"), ("OCV1", "ocv1"), ("ΔOCV", "docv"),
                     ("완화OCV1-7", "_inc01")]
        dis_specs = [("방전Tmin", "dis_tmin"), ("방전Tavg", "dis_tavg"),
                     ("방전Tmax", "dis_tmax"), ("방전스윙", "dis_swing")]
        corr_dis = within_tray_corr(df, keys, ocv_specs, dis_specs,
            "[20] 각 OCV × 방전온도 상관 (트레이 내, 평균제거)",
            os.path.join(args.outdir, "20_OCV_방전온도_상관.png"))
        print("\n   [각 OCV × 방전온도 상관]\n" + corr_dis.round(2).to_string())

        # [21] 구배 탄생 시점: 구간별 구배 세기
        pat_inc12 = per_tray_patterns(df, "_inc12", keys)
        pat_inc23 = per_tray_patterns(df, "_inc23", keys)
        plot_gradient_birth(
            [("OCV7(방전직후)", pat_ocv7), ("OCV1-OCV7(1일)", pat_inc01),
             ("OCV2-OCV1", pat_inc12), ("OCV3-OCV2", pat_inc23),
             ("ΔOCV(OCV1-3)", pat_ocv)],
            os.path.join(args.outdir, "21_구배탄생시점.png"))

        # [22] 4점 완화모델 → V∞, k 분리
        df, _ = relaxation_fit(df, keys, args.outdir)

        # OCV7 / 방전Tmax 대표트레이 카드 (다른 OCV처럼 패턴 히트맵)
        plot_pattern_cards(df, "ocv7", pat_ocv7, keys,
                           os.path.join(args.outdir, "23_OCV7_대표트레이.png"),
                           "OCV7(방전직후) 패턴별 대표 트레이 (12x12, 튀는 셀 제거)")
        plot_pattern_cards(df, "dis_tmax", pat_dtmax, keys,
                           os.path.join(args.outdir, "24_방전Tmax_대표트레이.png"),
                           "방전 Tmax 패턴별 대표 트레이 (12x12, 튀는 셀 제거)")

        # 대표트레이 궤적: OCV7→1→2→3, 온도(방전Tmax→OCV1/2/3)
        reps = pick_rep_trays(pat_ocv, 4)
        plot_field_evolution(df, keys, reps,
            [("OCV7", "ocv7"), ("OCV1", "ocv1"), ("OCV2", "ocv2"), ("OCV3", "ocv3")],
            "[25] 대표트레이 OCV 궤적 (OCV7→OCV1→OCV2→OCV3, 구배가 어디서 생기나)",
            os.path.join(args.outdir, "25_대표트레이_OCV궤적.png"))
        plot_field_evolution(df, keys, reps,
            [("방전Tmax", "dis_tmax"), ("OCV1온도", "t1"),
             ("OCV2온도", "t2"), ("OCV3온도", "t3")],
            "[26] 대표트레이 온도 궤적 (방전Tmax→OCV1/2/3 온도)",
            os.path.join(args.outdir, "26_대표트레이_온도궤적.png"))

        # [27] 공유 지문: 모든 OCV·온도 각각 실행 + 요약 비교
        fingerprint_all(df, keys,
            [("OCV7", "ocv7"), ("OCV1", "ocv1"), ("OCV2", "ocv2"), ("OCV3", "ocv3"),
             ("ΔOCV", "docv"), ("완화OCV1-7", "_inc01"),
             ("온도T1", "t1"), ("온도T2", "t2"), ("온도T3", "t3"), ("냉각량", "_cool"),
             ("방전Tmax", "dis_tmax"), ("방전Tavg", "dis_tavg"), ("방전스윙", "dis_swing")],
            args.outdir)
        # [27b] 트레이별 ΔOCV 필드 갤러리 (지문이 개별 트레이에 실재하는지 육안검증)
        plot_tray_gallery(df, keys, "docv", args.outdir, "ΔOCV")
        # [28] 실제 휴지시간(OCV7→OCV1) 정규화 진단
        rest_time_diagnostic(df, keys,
            os.path.join(args.outdir, "28_휴지시간.png"))
    else:
        print("\n[Phase 6] 방전7/OCV7 컬럼 없음 → 열이력/완화 분석 생략")
        df["_cool"] = df["t1"] - df["t3"]
        fingerprint_all(df, keys,
            [("OCV1", "ocv1"), ("OCV2", "ocv2"), ("OCV3", "ocv3"), ("ΔOCV", "docv"),
             ("온도T1", "t1"), ("온도T2", "t2"), ("온도T3", "t3"), ("냉각량", "_cool")],
            args.outdir)
        plot_tray_gallery(df, keys, "docv", args.outdir, "ΔOCV")

    # 냉각패턴 vs OCV패턴 겹침 (냉각이 큰 곳에서 OCV도 튀나) -----------
    merged = pd.merge(
        pat_cool[["랏-트레이", "패턴"]].rename(columns={"패턴": "냉각패턴"}),
        pat_ocv[["랏-트레이", "패턴"]].rename(columns={"패턴": "OCV패턴"}),
        on="랏-트레이", how="inner")
    both = merged[(merged["냉각패턴"] != "데이터 부족") &
                  (merged["OCV패턴"] != "데이터 부족")]
    overlap = np.mean(both["냉각패턴"] == both["OCV패턴"]) if len(both) else np.nan
    print("\n   냉각패턴과 OCV패턴이 같은 방향인 트레이 비율: %.1f%%" % (100 * overlap))

    # Excel 저장 ------------------------------------------------------
    xlsx = os.path.join(args.outdir, "패턴분석_요약.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame(list(report.items()),
                     columns=["항목", "셀 수"]).to_excel(xw, "제외리포트", index=False)
        corr.to_excel(xw, "관계표")
        pd.DataFrame({n: (vc[n] or {}) for n in ["OCV1", "OCV2", "OCV3", "ΔOCV"]}).T \
            .to_excel(xw, "분산게이트")
        for name in ["OCV1", "OCV2", "OCV3"]:
            cnt[name].to_excel(xw, "온도패턴_%s" % name, index=False)
        cnt_cool.to_excel(xw, "냉각량패턴", index=False)
        cnt_ocv.to_excel(xw, "OCV패턴_개수", index=False)
        persist.to_excel(xw, "패턴유지_전이", index=False)
        pat["OCV3"].to_excel(xw, "트레이별_온도구배_OCV3", index=False)
        pat_ocv.to_excel(xw, "트레이별_OCV구배", index=False)
        merged.to_excel(xw, "냉각vsOCV_패턴", index=False)
        pd.DataFrame({k: [v[0], v[1]] for k, v in align.items()},
                     index=["정렬도(코사인)", "트레이수"]).T.to_excel(xw, "OCV온도_정렬도")
        corr_ot.to_excel(xw, "OCV온도_상관")
        if corr_dis is not None:
            corr_dis.to_excel(xw, "OCV방전온도_상관")

    print("\n" + "=" * 60)
    print(" 완료. 결과 폴더:", os.path.abspath(args.outdir))
    print("   - PNG + 요약 Excel + 셀별 CSV(ΔOCV보정 / 완화모델)")
    print("=" * 60)


if __name__ == "__main__":
    main()
