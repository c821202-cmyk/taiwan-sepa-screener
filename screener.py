import pandas as pd
import numpy as np
import requests
import warnings
import os
import json
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")

CONFIG = {
    "from_52w_low_min": 25,
    "from_52w_high_max": 25,
    "min_volume": 300,
    "lookback_days": 400,
    "top_n": 50,
    "trend_min": 4,
    "rs_min": 70,
    "rs_entry_min": 85,
}


def get_stock_price(stock_id, start_date, token=""):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {"dataset": "TaiwanStockPrice", "data_id": stock_id, "start_date": start_date, "token": token}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == 200 and data.get("data"):
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
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
            df = df[df["type"].isin(["twse", "otc"])]
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


def calc_rs_table(stock_list, token=""):
    print("計算全市場 RS 評分...")
    rs_start = (datetime.now() - timedelta(days=420)).strftime("%Y-%m-%d")
    rs_data = {}
    total = len(stock_list)

    for i, (_, row) in enumerate(stock_list.iterrows(), 1):
        sid = row["stock_id"]
        if i % 100 == 0:
            print(f"RS 進度：{i}/{total}  完成：{len(rs_data)} 檔")
        df = get_stock_price(sid, rs_start, token)
        if df is None or len(df) < 252:
            time.sleep(0.2)
            continue
        close = df["close"].values
        try:
            r3  = (close[-1] / close[-63]  - 1) * 100 if len(close) >= 63  else 0
            r6  = (close[-1] / close[-126] - 1) * 100 if len(close) >= 126 else 0
            r9  = (close[-1] / close[-189] - 1) * 100 if len(close) >= 189 else 0
            r12 = (close[-1] / close[-252] - 1) * 100 if len(close) >= 252 else 0
            rs_data[sid] = r3 * 0.4 + r6 * 0.2 + r9 * 0.2 + r12 * 0.2
        except:
            pass
        time.sleep(0.2)

    if not rs_data:
        return {}

    values = list(rs_data.values())
    rs_table = {}
    for sid, val in rs_data.items():
        rs_table[sid] = round(sum(1 for v in values if v <= val) / len(values) * 100)

    print(f"RS 評分完成：{len(rs_table)} 檔  RS>=85：{sum(1 for v in rs_table.values() if v >= 85)} 檔")
    return rs_table


def calc_pivot_and_stops(df):
    if df is None or len(df) < 20:
        return None, None
    recent_high = df["max"].tail(20).max()
    pivot = round(recent_high * 1.005, 2)
    stop_loss = round(pivot * 0.92, 2)
    return pivot, stop_loss


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

    passed = sum([
        curr > ma150 and curr > ma200,
        ma150 > ma200,
        ma200 > ma200_1m,
        ma50 > ma150 and ma50 > ma200,
        curr > ma50,
        curr >= w52_low * (1 + CONFIG["from_52w_low_min"] / 100),
        curr >= w52_high * (1 - CONFIG["from_52w_high_max"] / 100),
        (curr / ma200 - 1) > 0.1,
    ])

    return passed, passed == 8, {
        "price": round(curr, 2),
        "ma50": round(ma50, 2),
        "ma150": round(ma150, 2),
        "ma200": round(ma200, 2),
        "52w_high": round(w52_high, 2),
        "52w_low": round(w52_low, 2),
        "from_52w_low_pct": round((curr / w52_low - 1) * 100, 1),
        "from_52w_high_pct": round((curr / w52_high - 1) * 100, 1),
        "trend_pass": passed == 8,
    }


def check_vcp(df, lookback=50):
    if df is None or len(df) < lookback + 10:
        return False, 0
    recent = df.tail(lookback).copy()
    recent["range_pct"] = (recent["max"] - recent["min"]) / recent["close"] * 100
    n = len(recent)
    seg1 = recent.iloc[:n//3]["range_pct"].mean()
    seg2 = recent.iloc[n//3:2*n//3]["range_pct"].mean()
    seg3 = recent.iloc[2*n//3:]["range_pct"].mean()
    avg_vol = df["Trading_Volume"].iloc[-30:].mean() if "Trading_Volume" in df.columns else 0
    rec_vol = df["Trading_Volume"].iloc[-5:].mean() if "Trading_Volume" in df.columns else 0
    score = 0
    if seg1 > seg2 > seg3: score += 50
    if avg_vol > 0 and rec_vol < avg_vol * 0.85: score += 50
    if seg3 < 3.0: score += 20
    return score >= 50, min(score, 100)


def calc_sepa_score(trend_cond, vcp_score, eps_yoy, from_52w_low, rs_score=None):
    score = 0
    score += trend_cond / 8 * 35
    score += vcp_score * 0.15
    if eps_yoy is not None:
        if eps_yoy >= 50:   score += 25
        elif eps_yoy >= 30: score += 20
        elif eps_yoy >= 20: score += 15
        elif eps_yoy >= 10: score += 8
    if from_52w_low >= 80:   score += 10
    elif from_52w_low >= 50: score += 7
    elif from_52w_low >= 25: score += 3
    if rs_score is not None:
        if rs_score >= 90:   score += 15
        elif rs_score >= 80: score += 12
        elif rs_score >= 70: score += 8
        elif rs_score >= 60: score += 4
    return round(min(score, 100), 1)


def update_google_sheets(result_df, sheet_id):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not sa_json:
            print("未設定 GOOGLE_SERVICE_ACCOUNT_JSON")
            return False

        sa_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()

        today = datetime.now().strftime("%Y/%m/%d")
        headers = ["日期","排名","代號","股票名稱","產業","SEPA評分","RS評分",
                   "趨勢條件","趨勢通過","VCP型態","EPS年成長",
                   "現價","進場參考價","停損價","距高點%","距低點%","訊號"]

        values = [headers]
        for rank, row in result_df.iterrows():
            values.append([
                today, int(rank),
                row["代號"], row["股票名稱"], row["產業"],
                row["SEPA評分"], str(row["RS評分"]),
                row["趨勢條件"], row["趨勢通過"], row["VCP型態"],
                row["EPS年成長"], str(row["現價"]),
                str(row["進場參考價"]), str(row["停損價"]),
                row["距高點%"], row["距低點%"], row["訊號"],
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


def main():
    print(f"台股 SEPA 選股系統啟動")
    print(f"執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    print("取得台股清單...")
    stock_list = get_taiwan_stock_list(FINMIND_TOKEN)
    if stock_list is None or len(stock_list) < 100:
        print("使用備用清單")
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

    rs_table = calc_rs_table(stock_list, FINMIND_TOKEN)

    start_date = (datetime.now() - timedelta(days=CONFIG["lookback_days"])).strftime("%Y-%m-%d")
    results = []
    skipped = 0

    for i, (_, row) in enumerate(stock_list.iterrows(), 1):
        sid  = row["stock_id"]
        name = row["stock_name"]
        ind  = row["industry_category"]

        if i % 50 == 0:
            print(f"進度：{i}/{len(stock_list)}  找到：{len(results)} 檔  略過：{skipped} 檔")

        rs_score = rs_table.get(sid, None)
        if rs_table and rs_score is not None and rs_score < CONFIG["rs_min"]:
            time.sleep(0.1)
            continue

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

        sepa = calc_sepa_score(trend_cond, vcp_score, eps_yoy,
                               detail.get("from_52w_low_pct", 0), rs_score)

        pivot, stop_loss = calc_pivot_and_stops(df)

        rs_ok = rs_score is not None and rs_score >= CONFIG["rs_entry_min"]
        if sepa >= 75 and trend_pass and vcp_pass and rs_ok:
            signal = "可進場"
        elif sepa >= 55 and trend_cond >= 6:
            signal = "觀察"
        else:
            signal = "待確認"

        eps_str = f'+{eps_yoy:.1f}%' if eps_yoy and eps_yoy > 0 else (f'{eps_yoy:.1f}%' if eps_yoy else '—')

        results.append({
            "代號": sid, "股票名稱": name, "產業": ind,
            "SEPA評分": sepa,
            "RS評分": rs_score if rs_score is not None else '—',
            "趨勢條件": f"{trend_cond}/8",
            "趨勢通過": "是" if trend_pass else "否",
            "VCP型態": "是" if vcp_pass else "否",
            "EPS年成長": eps_str,
            "現價": detail.get("price", "—"),
            "進場參考價": pivot if pivot else "—",
            "停損價": stop_loss if stop_loss else "—",
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

    result_df.to_json("results.json", orient="records", force_ascii=False)
    print("results.json 已儲存")

    if GOOGLE_SHEET_ID:
        update_google_sheets(result_df.head(CONFIG["top_n"]), GOOGLE_SHEET_ID)

    print("\n今日 SEPA 前10名：")
    print(result_df.head(10)[["代號","股票名稱","SEPA評分","RS評分","趨勢條件","VCP型態","訊號"]].to_string())


if __name__ == "__main__":
    main()
