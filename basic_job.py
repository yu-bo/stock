

import lib.common as common
import sys
import time
import pandas as pd
import tushare as ts
from sqlalchemy.types import NVARCHAR
from sqlalchemy import inspect
import datetime

import pymysql as MySQLdb
# pymysql.install_as_MySQLdb()
# import MySQLdb

def stat_all(tmp_datetime):
     # 存款利率
    data = ts.get_deposit_rate()
    common.insert_db(data, "ts_deposit_rate", False, "`date`,`deposit_type`")

    
    # 贷款利率
    data = ts.get_loan_rate()
    common.insert_db(data, "ts_loan_rate", False, "`date`,`loan_type`")

    # 存款准备金率
    data = ts.get_rrr()
    common.insert_db(data, "ts_rrr", False, "`date`")

    # 货币供应量
    data = ts.get_money_supply()
    common.insert_db(data, "ts_money_supply", False, "`month`")

    # 货币供应量(年底余额)
    data = ts.get_money_supply_bal()
    common.insert_db(data, "ts_money_supply_bal", False, "`year`")

    # 国内生产总值(年度)
    data = ts.get_gdp_year()
    common.insert_db(data, "ts_gdp_year", False, "`year`")

    # 国内生产总值(季度)
    data = ts.get_gdp_quarter()
    common.insert_db(data, "ts_get_gdp_quarter", False, "`quarter`")

    # 三大需求对GDP贡献
    data = ts.get_gdp_for()
    common.insert_db(data, "ts_gdp_for", False, "`year`")

    # 三大产业对GDP拉动
    data = ts.get_gdp_pull()
    common.insert_db(data, "ts_gdp_pull", False, "`year`")

    # 三大产业贡献率
    data = ts.get_gdp_contrib()
    common.insert_db(data, "ts_gdp_contrib", False, "`year`")

    # 居民消费价格指数
    data = ts.get_cpi()
    common.insert_db(data, "ts_cpi", False, "`month`")

    # 工业品出厂价格指数
    data = ts.get_ppi()
    common.insert_db(data, "ts_ppi", False, "`month`")

    #############################基本面数据 http://tushare.org/fundamental.html
    # 股票列表
    data = ts.get_stock_basics()
    print(data.index)
    common.insert_db(data, "ts_stock_basics", True, "`code`")

def create_new_database():
    db = MySQLdb.connect(common.MYSQL_HOST,common.MYSQL_USER,common.MYSQL_PWD,"mysql",charset='utf8')
    with MySQLdb.connect(common.MYSQL_HOST, common.MYSQL_USER, common.MYSQL_PWD, "mysql", charset="utf8") as db:
        try:
            create_sql= "CREATE DATABASE IF NOT EXISTS %s CHARACTER SET utf8 COLLATE utf8_general_ci" % common.MYSQL_DB
            print(create_sql)
            db.execute(create_sql)
        except Exception as e:
            print("error CREATE DATABASE :" ,e)
    

if __name__ == "__main__":
    try:
        with MySQLdb.connect(common.MYSQL_HOST, common.MYSQL_USER, common.MYSQL_PWD, common.MYSQL_DB,
                             charset="utf8") as db:
            db.execute("select 1 ")
        pass
    except Exception as e:
        print("check  MYSQL_DB error and create new one :", e)
        create_new_database()
        pass
    
    # 执行数据初始化。
    # 使用方法传递。
    tmp_datetime = common.run_with_args(stat_all)
    
