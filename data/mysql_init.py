import MySQLdb 


# 使用环境变量获得数据库。兼容开发模式可docker模式。
MYSQL_HOST =  "192.168.1.9"
MYSQL_USER =  "root"
MYSQL_PWD =  "123456"
MYSQL_DB =  "stock_data"

db= MySQLdb.connect(MYSQL_HOST,MYSQL_USER,MYSQL_PWD,MYSQL_DB,charset='utf8')
cursor = db.cursor()

sqlstr= "show databases"

sql = "CREATE TABLE 'ts_deposit_rate' ( \
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY, \
date NVARCHAR(50) NOT NULL,\
deposit_type NVARCHAR(50) NOT NULL,\
rate NVARCHAR(50),\
) IF NOT EXISTS 'ts_deposit_rate' ;"



try:
  cursor.execute(sql)
  results = cursor.fetchall()
  for row in results:
    print(row)
except Exception as e:
  print(e)

