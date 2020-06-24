# stock

业余时间学习下深度学习，个人理解就是对已有的数据进行非线性映射，然后再映射的空间内找出数据的特征（个人理解可能不准，请指正）。 
QQ：840050001希望给出优化意见

目前是预测收盘价，后面会继续更新，预测更多的数据。   
  
处理过程大体如下：  
1、使用tushare获取stock信息  
2、对数据进行处理，做好train_x和train_y的对应关系  
3、训练和预测 ，网络是学习x也y之间的映射关系，网络要与数据匹配  
  
前端显示地址：https://github.com/yu-bo/my_project  
tushare地址：https://tushare.pro/register?reg=341048   
  
工程介绍  
1、get_info.py 用于从获取stock信息，目前只有日线数据，后面会增加（tushare积分不够，有些数据获取不到，大家注册下给点积分） 
2、stock_sql.py 用于将部分信息记录到数据库，方便查询检索。数据库使用的sqlite3  
3、prepare.py 用于对数据进行处理，生成train_x，train_y的对应关系，满足网络训练需要。NN可以学习映射规律，进行预测。  
4、trainning.py 网络模型，并进行训练。  
5、evaluate.py 对模型进行简单的评估。  
6、server.py flask做的后台用于数据展示。



![](https://github.com/yu-bo/stock/blob/master/20200314.png)
