import pandas as pd
import json

excel_path = "data/kc/科创指数成分.xlsx"
latest_kc_50 = pd.read_excel(excel_path, sheet_name="科创50_20240110")
history_kc_50 = pd.read_excel(excel_path, sheet_name="科创50历史调整")
latest_kc_100 = pd.read_excel(excel_path, sheet_name="科创100_20240110")
history_kc_100 = pd.read_excel(excel_path, sheet_name="科创100历史调整")

def get_kc_for_date(latest_kc : pd.DataFrame, history_kc : pd.DataFrame):
    data = {}
    latest_kc_set = set(latest_kc["Wind代码"].unique().tolist())
    common_kc_set = latest_kc_set.copy()
    all_kc_set = latest_kc_set.copy()
    changed_date_list = history_kc["交易日期"].unique().tolist()
    changed_date_list = [x.strftime('%Y-%m-%d') for x in changed_date_list]
    
    for changed_date in changed_date_list:
        data[changed_date] = list(latest_kc_set)
        spec_date_history_kc = history_kc[history_kc["交易日期"] == changed_date]
        add_ticker_list = spec_date_history_kc[spec_date_history_kc["状态"] == "纳入"]["代码"].unique().tolist()
        remove_ticker_list = spec_date_history_kc[spec_date_history_kc["状态"] == "剔除"]["代码"].unique().tolist()
        latest_kc_set.difference_update(add_ticker_list)
        latest_kc_set.update(remove_ticker_list)
        all_kc_set.update(remove_ticker_list)
        common_kc_set = common_kc_set.intersection(latest_kc_set)
    data["2019-07-22"] = list(latest_kc_set)
    data["all"] = list(all_kc_set)
    data["common"] = list(common_kc_set)
    return data

kc_50_hist = get_kc_for_date(latest_kc_50, history_kc_50)
kc_100_hist = get_kc_for_date(latest_kc_100, history_kc_100)

with open("data/kc/hc_50_hist.json", "w") as json_file:
    json.dump(kc_50_hist, json_file, indent=4)

with open("data/kc/hc_100_hist.json", "w") as json_file:
    json.dump(kc_100_hist, json_file, indent=4)