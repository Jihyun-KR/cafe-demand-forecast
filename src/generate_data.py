"""
카페 판매 샘플 데이터 + 날씨 데이터 만들기

왜 '샘플(합성) 데이터'인가?
  실제 카페 POS 판매 데이터는 공개된 게 거의 없어서, 오피스 상권 카페 1곳을 가정하고
  '날씨와 요일이 판매에 영향을 준다'는 현실적인 규칙으로 2년치 데이터를 만들었다.
  날씨는 서울 기후(계절별 기온, 여름 장마)를 흉내 낸 값이다.

  실제 기상청 데이터로 바꾸고 싶으면 src/weather.py 의 load_kma_asos() 를 쓰면 된다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# 2024~2025 평일 공휴일
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-02-09", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-01",
    "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17",
    "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03",
    "2025-05-01", "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15",
    "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25",
])

# 월별 비 올 확률 (서울: 7~8월 장마)
RAIN_PROB = {1: .15, 2: .17, 3: .22, 4: .25, 5: .27, 6: .35,
             7: .55, 8: .45, 9: .30, 10: .20, 11: .25, 12: .18}


def make_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    doy = dates.dayofyear.values
    # 서울 연평균 약 12.5도, 1월 약 -2도 / 8월 약 27도
    temp = 12.5 + 14.5 * np.sin(2 * np.pi * (doy - 105) / 365) + rng.normal(0, 2.8, len(dates))
    rain_p = np.array([RAIN_PROB[m] for m in dates.month])
    is_rain = rng.random(len(dates)) < rain_p
    scale = np.where(dates.month.isin([7, 8]), 25, 8)
    rain_mm = np.where(is_rain, rng.exponential(scale), 0.0)
    return pd.DataFrame({"date": dates, "temp_c": temp.round(1), "rain_mm": rain_mm.round(1)})


def make_sales(w: pd.DataFrame) -> pd.DataFrame:
    t = w["temp_c"].values
    rain = (w["rain_mm"].values > 1).astype(float)
    heavy = (w["rain_mm"].values > 20).astype(float)
    dow = w["date"].dt.dayofweek.values
    off = ((dow >= 5) | w["date"].isin(HOLIDAYS)).astype(float)   # 쉬는 날
    monday, friday = (dow == 0).astype(float), (dow == 4).astype(float)

    # 오피스 상권 카페 가정: 평일엔 커피가 많이 팔리고, 쉬는 날엔 블렌디드가 잘 팔림
    menus = {
        "아이스 아메리카노": (175 + 5.2 * (t - 15)) * (1 - .25 * off) * (1 - .12 * rain - .10 * heavy) * (1 + .06 * monday),
        "따뜻한 아메리카노": (115 - 4.0 * (t - 15)) * (1 - .25 * off) * (1 + .10 * rain) * (1 + .06 * monday),
        "카페 라떼":        (105 - 1.2 * (t - 15)) * (1 - .15 * off) * (1 + .06 * rain) * (1 + .05 * friday),
        "블렌디드 음료":    (30 + 3.0 * np.clip(t - 16, 0, None)) * (1 + .40 * off) * (1 - .20 * rain - .15 * heavy) * (1 + .10 * friday),
        "따뜻한 차":        (34 - 1.3 * (t - 15)) * (1 + .05 * off) * (1 + .18 * rain),
    }
    rows = []
    for name, mu in menus.items():
        mu = np.clip(mu, 4, None)
        cups = rng.poisson(mu * rng.normal(1, .05, len(mu)))
        rows.append(pd.DataFrame({"date": w["date"], "menu": name, "cups": cups}))
    return pd.concat(rows).merge(w, on="date")


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    df = make_sales(make_weather(dates))
    df["is_holiday"] = df["date"].isin(HOLIDAYS).astype(int)
    out = Path(__file__).resolve().parents[1] / "data" / "cafe_sales_sample.csv"
    df.sort_values(["date", "menu"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out.name}  ({len(df):,}행, {df['date'].nunique()}일 × {df['menu'].nunique()}개 메뉴)")
