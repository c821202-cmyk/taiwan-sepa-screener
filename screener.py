import pandas as pd
import numpy as np
import requests
import warnings
import os
import json
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ====== 設定 ======
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")

CONFIG = {
    "from_52w_low_min": 25,
    "from_52w_high_max": 25,
    "eps_growth_min": 20,
    "min_volume": 300,
    "lookback_days": 400,
    "top_n": 50,
    "trend_min": 4,
}

# ====== API 函式 ======
def get_stock_price(stock_id, start_date, token=""):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "token": token
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == 200 and data.get("data"):
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df
    except:
        pass
    return None


def get_taiwan_stock_list(token=""):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {"dataset": "TaiwanStockInfo", "token": token}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("status") == 200:
            df = pd.DataFrame(data["data"])
            df = df[df["type"] == "twse"]
            df = df[df["stock_id"].str.match(r"^\d{4}$")]
            df = df.drop_duplicates(subset=["stock_id"], keep="first")
            return df[["stock_id", "stock_name", "industry_category"]].reset_index(drop=True)
    except:
        pass
    return None


def get_eps_growth(stock_id, token=""):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockFinancialStatements",
        "data_id": stock_id,
        "start_date": (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d"),
        "token": token
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == 200 and data.get("data"):
            df = pd.DataFrame(data["data"])
            eps_df = df[df["type"] == "EPS"][["date", "value"]].copy()
            eps_df["date"] = pd.to_datetime(eps_df["date"])
            eps_df = eps_df.sort_values("date").tail(6)
            if len(eps_df) >= 4:
                vals = eps_df["value"].astype(float).values
                yoy = ((vals[-1] - vals[-5]) / abs(vals[-5]) * 100) if len(vals) >= 5 and vals[-5] != 0 else 0
                qoq = ((vals[-1] - vals[-2]) / abs(vals[-2]) * 100) if vals[-2] != 0 else 0
                return round(yoy, 1), round(qoq, 1), round(float(vals[-1]), 2)
    except:
        pass
    return None, None, None


# ====== SEPA 核心函式 ======
def check_trend_template(df):
    if df is None or len(df) < 210:
        return 0, False, {}

    close = df["close"].values
    curr = close[-1]
    n = len(close)

    ma50  = np.mean(close[-50:])
    ma150 = np.mean(close[-150:])
    ma200 = np.mean(close[-200:])
    ma200_1m = np.mean(close[-220:-20]) if n >= 220 else np.mean(close[:-20])

    w52_high = np.max(close[-252:]) if n >= 252 else np.max(close)
    w52_low  = np.min(close[-252:]) if n >= 252 else np.min(close)

    c1 = curr > ma150 and curr > ma200
    c2 = ma150 > ma200
    c3 = ma200 > ma200_1m
    c4 = ma50 > ma150 and ma50 > ma200
    c5 = curr > ma50
    c6 = curr >= w52_low * (1 + CONFIG["from_52w_low_min"] / 100)
    c7 = curr >= w52_high * (1 - CONFIG["from_52w_high_max"] / 100)
    c8 = (curr / ma200 - 1) > 0.1

    passed = sum([c1, c2, c3, c4, c5, c6, c7, c8])

    details = {
        "price": round(curr, 2),
        "ma50": round(ma50, 2),
        "ma150": round(ma150, 2),
        "ma200": round(ma200, 2),
        "52w_high": round(w52_high, 2),
        "52w_low": round(w52_low, 2),
        "from_52w_low_pct": round((curr / w52_low - 1) * 100, 1),
        "from_52w_high_pct": round((curr / w52_high - 1) * 100, 1),
        "trend_conditions": passed,
        "trend_pass": passed == 8,
    }
    return passed, passed == 8, details


def check_vcp(df, lookback=50):
    if df is None or len(df) < lookback + 10:
        return False, 0
    recent = df.tail(lookback).copy()
    recent["range_pct"] = (recent["max"] - recent["min"]) / recent["close"] * 100
    n = len(recent)
    seg1 = recent.iloc[:n//3]["range_pct"].mean()
    seg2 = recent.iloc[n//3:2*n//3]["range_pct"].mean()
    seg3 = recent.iloc[2*n//3:]["range_pct"].mean()
    is_contracting = seg1 > seg2 > seg3
    avg_vol = df["Trading_Volume"].iloc[-30:].mean() if "Trading_Volume" in df.columns else 0
    rec_vol = df["Trading_Volume"].iloc[-5:].mean() if "Trading_Volume" in df.columns else 0
    vol_ok = rec_vol < avg_vol * 0.85 if avg_vol > 0 else False
    score = 0
    if is_contracting: score += 50
    if vol_ok: score += 50
    if seg3 < 3.0: score += 20
    return score >= 50, min(score, 100)


def calc_sepa_score(trend_cond, vcp_score, eps_yoy, from_52w_low):
    score = 0
    score += trend_cond / 8 * 40
    score += vcp_score * 0.20
    if eps_yoy is not None:
        if eps_yoy >= 50:   score += 25
        elif eps_yoy >= 30: score += 20
        elif eps_yoy >= 20: score += 15
        elif eps_yoy >= 10: score += 8
    if from_52w_low >= 80:   score += 15
    elif from_52w_low >= 50: score += 10
    elif from_52w_low >= 25: score += 5
    return round(min(score, 100), 1)


# ====== Google Sheets 更新 ======
def update_google_sheets(result_df, sheet_id):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not sa_json:
            print("未設定 GOOGLE_SERVICE_ACCOUNT_JSON，跳過 Sheets 更新")
            return False

        sa_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()

        today = datetime.now().strftime("%Y/%m/%d")
        headers = ["日期","排名","代號","股票名稱","產業","SEPA評分",
                   "趨勢條件","趨勢通過","VCP型態","EPS年成長",
                   "現價","距高點%","距低點%","訊號"]

        values = [headers]
        for rank, row in result_df.iterrows():
            values.append([
                today, int(rank),
                row["代號"], row["股票名稱"], row["產業"],
                row["SEPA評分"], row["趨勢條件"],
                row["趨勢通過"], row["VCP型態"], row["EPS年成長"],
                str(row["現價"]), row["距高點%"], row["距低點%"], row["訊號"],
            ])

        sheet.values().clear(spreadsheetId=sheet_id, range="工作表1").execute()
        sheet.values().update(
            spreadsheetId=sheet_id,
            range="工作表1!A1",
            valueInputOption="RAW",
            body={"values": values}
        ).execute()

        print(f"Google Sheets 更新成功！共 {len(result_df)} 筆")
        return True

    except Exception as e:
        print(f"Google Sheets 更新失敗：{e}")
        return False


# ====== 主程式 ======
def main():
    print(f"台股 SEPA 選股系統啟動")
    print(f"執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 取得股票清單
    print("取得台股清單...")
    stock_list = get_taiwan_stock_list(FINMIND_TOKEN)
    if stock_list is None or len(stock_list) < 100:
        print("API 失敗，使用備用清單")
        backup = [
            ("2330","台積電","半導體"),("2317","鴻海","電子零組件"),("2454","聯發科","半導體"),
            ("2308","台達電","電子零組件"),("2382","廣達","電腦周邊"),("2357","華碩","電腦周邊"),
            ("3711","日月光投控","半導體"),("2379","瑞昱","半導體"),("6488","環球晶","半導體"),
            ("2303","聯電","半導體"),("2412","中華電","電信"),("2882","國泰金","金融"),
            ("2881","富邦金","金融"),("2886","兆豐金","金融"),("1301","台塑","化工"),
            ("1303","南亞","化工"),("2002","中鋼","鋼鐵"),("2912","統一超","零售"),
            ("2207","和泰車","汽車"),("3008","大立光","光學"),
            ("6669","緯穎","伺服器"),("3034","聯詠","IC設計"),("2337","旺宏","半導體"),
            ("4967","十銓","記憶體"),("6278","台表科","半導體"),("3533","嘉澤","連接器"),
            ("2408","南亞科","記憶體"),("3231","緯創","電子零組件"),("2474","可成","機殼"),
            ("6505","台塑化","石化"),("2395","研華","工業電腦"),("2385","群光","電源"),
        ]
        stock_list = pd.DataFrame(backup, columns=["stock_id","stock_name","industry_category"])

    print(f"股票清單：{len(stock_list)} 檔")

    # 開始掃描
    start_date = (datetime.now() - timedelta(days=CONFIG["lookback_days"])).strftime("%Y-%m-%d")
    results = []
    total = len(stock_list)
    skipped = 0

    for i, (_, row) in enumerate(stock_list.iterrows(), 1):
        sid  = row["stock_id"]
        name = row["stock_name"]
        ind  = row["industry_category"]

        if i % 50 == 0:
            print(f"進度：{i}/{total}  找到：{len(results)} 檔  略過：{skipped} 檔")

        df = get_stock_price(sid, start_date, FINMIND_TOKEN)
        if df is None or len(df) < 210:
            skipped += 1
            time.sleep(0.3)
            continue

        avg_vol = df["Trading_Volume"].tail(20).mean() if "Trading_Volume" in df.columns else 0
        if avg_vol < CONFIG["min_volume"]:
            time.sleep(0.2)
            continue

        trend_cond, trend_pass, detail = check_trend_template(df)
        if trend_cond < CONFIG["trend_min"]:
            time.sleep(0.2)
            continue

        vcp_pass, vcp_score = check_vcp(df)

        eps_yoy, eps_qoq, latest_eps = None, None, None
        if trend_cond >= 6:
            eps_yoy, eps_qoq, latest_eps = get_eps_growth(sid, FINMIND_TOKEN)
            time.sleep(0.5)

        sepa = calc_sepa_score(trend_cond, vcp_score, eps_yoy, detail.get("from_52w_low_pct", 0))

        if sepa >= 75 and trend_pass:
            signal = "可進場"
        elif sepa >= 55 and trend_cond >= 6:
            signal = "觀察"
        else:
            signal = "待確認"

        eps_str = f'+{eps_yoy:.1f}%' if eps_yoy and eps_yoy > 0 else (f'{eps_yoy:.1f}%' if eps_yoy else '—')

        results.append({
            "代號": sid,
            "股票名稱": name,
            "產業": ind,
            "SEPA評分": sepa,
            "趨勢條件": f"{trend_cond}/8",
            "趨勢通過": "是" if trend_pass else "否",
            "VCP型態": "是" if vcp_pass else "否",
            "EPS年成長": eps_str,
            "現價": detail.get("price", "—"),
            "距高點%": f'{detail.get("from_52w_high_pct", 0):.1f}%',
            "距低點%": f'+{detail.get("from_52w_low_pct", 0):.1f}%',
            "訊號": signal,
        })
        time.sleep(0.3)

    print(f"\n掃描完成！找到 {len(results)} 檔  略過 {skipped} 檔")

    if not results:
        print("今日無符合條件股票")
        return

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("SEPA評分", ascending=False).reset_index(drop=True)
    result_df.index += 1

    # 儲存 JSON（給網站用）
    result_df.to_json("results.json", orient="records", force_ascii=False)
    print("results.json 已儲存")

    # 更新 Google Sheets
    if GOOGLE_SHEET_ID:
        update_google_sheets(result_df.head(CONFIG["top_n"]), GOOGLE_SHEET_ID)

    # 印出前10名
    print("\n今日 SEPA 前10名：")
    print(result_df.head(10)[["代號","股票名稱","SEPA評分","趨勢條件","VCP型態","訊號"]].to_string())


if __name__ == "__main__":
    main()
