"""
날씨·요일로 메뉴별 판매량 예측하기

흐름
  1. 피처 만들기: 기온, 강수량, 요일, 쉬는 날(주말·공휴일) 여부
  2. 시간 순서대로 나누기: 마지막 90일은 '미래'로 숨겨두고 평가에만 사용
  3. 비교 기준(베이스라인): "최근 4주 같은 요일 평균" — 현장에서 흔히 쓰는 감(感) 발주 방식
  4. 메뉴별 Gradient Boosting 모델 학습 → 베이스라인보다 얼마나 나은지 비교
  5. 대시보드용 결과(results.json)와 index.html 생성
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["temp_c", "rain_mm", "dow", "is_off"]
FEATURE_KO = {"temp_c": "기온", "rain_mm": "강수량", "dow": "요일", "is_off": "주말·공휴일"}
TEST_DAYS = 90


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "cafe_sales_sample.csv", parse_dates=["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["is_off"] = ((df["dow"] >= 5) | (df["is_holiday"] == 1)).astype(int)
    return df


def baseline(menu_df: pd.DataFrame, test_dates) -> np.ndarray:
    """최근 4주 같은 요일 평균 (예측 시점까지의 실제값만 사용)"""
    s = menu_df.set_index("date")["cups"]
    preds = []
    for d in test_dates:
        past = [d - pd.Timedelta(weeks=k) for k in range(1, 5)]
        preds.append(s.reindex(past).mean())
    return np.array(preds)


def mape(y, p):
    return float(np.mean(np.abs(y - p) / np.clip(y, 1, None)) * 100)


def main():
    df = load()
    cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    menus = list(df["menu"].unique())
    out = {"menus": menus, "metrics": {}, "importance": {}, "test": {}, "grid": {}}

    temps = list(range(-10, 36))
    rains = [0, 5, 25]

    for m in menus:
        d = df[df["menu"] == m].sort_values("date")
        tr, te = d[d["date"] < cutoff], d[d["date"] >= cutoff]

        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              max_depth=4, categorical_features=[2],
                                              random_state=0)
        model.fit(tr[FEATURES], tr["cups"])
        p_model = model.predict(te[FEATURES])
        p_base = baseline(d, te["date"])
        y = te["cups"].values

        out["metrics"][m] = {
            "model_mae": round(mean_absolute_error(y, p_model), 1),
            "base_mae": round(mean_absolute_error(y, p_base), 1),
            "model_mape": round(mape(y, p_model), 1),
            "base_mape": round(mape(y, p_base), 1),
        }
        imp = permutation_importance(model, te[FEATURES], y, n_repeats=10, random_state=0)
        total = max(imp.importances_mean.clip(0).sum(), 1e-9)
        out["importance"][m] = {FEATURE_KO[f]: round(float(v / total * 100), 1)
                                for f, v in zip(FEATURES, imp.importances_mean.clip(0))}
        out["test"][m] = {"actual": y.tolist(),
                          "model": np.round(p_model).astype(int).tolist(),
                          "base": np.round(p_base).astype(int).tolist()}

        # 시뮬레이터용 예측표: 기온(46) × 강수(3) × 요일(7) × 쉬는날(2)
        grid = pd.DataFrame([(t, r, w, o) for t in temps for r in rains for w in range(7) for o in (0, 1)],
                            columns=FEATURES)
        grid = grid[~((grid["dow"] >= 5) & (grid["is_off"] == 0))]   # 주말은 항상 쉬는 날
        grid["pred"] = np.clip(model.predict(grid[FEATURES]), 0, None).round().astype(int)
        out["grid"][m] = {f"{t}|{r}|{w}|{o}": int(p) for t, r, w, o, p in grid.itertuples(index=False)}

    out["test_dates"] = df[df["date"] >= cutoff]["date"].drop_duplicates().dt.strftime("%m.%d").tolist()
    out["temps"], out["rains"] = temps, rains

    # 아이스가 따뜻한 아메리카노를 앞지르는 기온 (평일, 비 없음 기준)
    g_ice, g_hot = out["grid"]["아이스 아메리카노"], out["grid"]["따뜻한 아메리카노"]
    out["crossover"] = next(t for t in temps if g_ice[f"{t}|0|2|0"] > g_hot[f"{t}|0|2|0"])

    tot_m = sum(v["model_mae"] for v in out["metrics"].values())
    tot_b = sum(v["base_mae"] for v in out["metrics"].values())
    out["summary"] = {"model_mae": round(tot_m, 1), "base_mae": round(tot_b, 1),
                      "improve_pct": round((1 - tot_m / tot_b) * 100, 1),
                      "train_days": int(df[df["date"] < cutoff]["date"].nunique()),
                      "test_days": TEST_DAYS}

    (ROOT / "dashboard" / "results.json").write_text(json.dumps(out, ensure_ascii=False))
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(out, ensure_ascii=False))
    (ROOT / "dashboard" / "index.html").write_text(html, encoding="utf-8")

    print(f"베이스라인 대비 오차 {out['summary']['improve_pct']}% 감소  |  교차 기온 {out['crossover']}°C")
    for m, v in out["metrics"].items():
        print(f"  {m:<10} 모델 MAPE {v['model_mape']:>5}%  vs  베이스라인 {v['base_mape']:>5}%")


if __name__ == "__main__":
    main()
