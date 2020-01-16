
import pandas as pd





def load_info():
    name ="阿科力-603722.SH"
    fileName = "".join(["data/info/",name,".csv"])
    data = pd.read_csv(fileName)
    
    #print(data)
    return data.values

def trainning():
    pass

if __name__ == "__main__":
    #load_info()
    trainning()