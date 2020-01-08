import  tushare as ts
import csv

ts.set_token("d1af48f518c17415b1b98b2ce84ab7b1a0025adfdde78e22513b31ec")
pro = ts.pro_api()

d={"ts_codeT":"S代码",
"symbol":"股票代码",
"name":"股票名称",
"area":"所在地域",
"industry":"所属行业",
"fullname":"股票全称",
"enname":"英文全称",
"market":"市场类型(主板/中小板/创业板/科创板)",
"exchange":"交易所代码",
"curr_type":"交易货币",
"list_status":"上市状态:L上市 D退市 P暂停上市",
"list_date":"上市日期",
"delist_date":"退市日期",
"is_hs":"是否沪深港通标的，N否 H沪股通 S深股通",
}

print(d.keys())
print(d.values())
data = pro.query('stock_basic', exchange='', list_status='L', fields=d.keys())
print(data)



