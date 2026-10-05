"""
실제 기상청 날씨 데이터로 바꿔 끼우기

1. 기상자료개방포털(data.kma.go.kr) → 데이터 → 기상관측 → 지상 → 종관기상관측(ASOS) → 일자료
2. 지점: 서울(108), 기간 선택, 항목: 평균기온 · 일강수량 → CSV 다운로드
3. 아래 함수로 읽으면 generate_data.py 와 같은 형식(date, temp_c, rain_mm)이 된다.
   (다운로드한 CSV의 열 이름이 다르면 rename 부분만 맞춰주면 된다)
"""
import pandas as pd


def load_kma_asos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    df = df.rename(columns={"일시": "date", "평균기온(°C)": "temp_c", "일강수량(mm)": "rain_mm"})
    df["date"] = pd.to_datetime(df["date"])
    df["rain_mm"] = df["rain_mm"].fillna(0)   # 기상청 CSV는 비가 안 온 날을 빈칸으로 둔다
    return df[["date", "temp_c", "rain_mm"]]
