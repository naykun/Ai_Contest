import numpy as np


file_name = [	"result_logitsDenseNet201_A.csv",
				"result_logitsInceptionResNetV2_A.csv",
				"result_logits_NASNet_A.csv"
			]

Class_Index = {
    0:'0',
    1:'1',
    2:'10',
    3:'11',
    4:'12',
    5:'13',
    6:'14',
    7:'15',
    8:'16',
    9:'17',
    10:'18',
    11:'19',
    12:'2',
    13:'3',
    14:'4',
    15:'5',
    16:'6',
    17:'7',
    18:'8',
    19:'9'
}


def fil(x):
	return list(filter(lambda x:x!="",x))

def get_Matrix(name):
	x = open(name,"r").read().split("\n")
	x = fil(x)
	x = x[1:]
	print(name+"\t\tlen:",len(x))
	ret = []
	for i in x:
		d = i.split(",")
		ret.append(d[1:])
	ret = np.array(ret, dtype = "float64")
	return ret

def get_ID(x):
	x = open(x,"r").read().split("\n")
	x = fil(x)
	x = x[0:1]
	ret = []
	for i in x:
		d = i.split(",")
		ret.append(d[1:])
	ret = np.array(ret, dtype = "float64")
	return ret

ID = [] 
ID = open("list_A.csv","r").read().split("\n")
ID = fil(ID)[1:]

tensor = []
for i in file_name:
	tensor.append(get_Matrix(i))

tensor = np.array(tensor)

if(tensor.shape[1] != len(ID)):
	print("ERROR: shape not same")

#get median 
median = np.median(tensor, axis = 0)

#get top3 and change category


def map_category(x):
	ret = []
	for i in x:
		ret.append(Class_Index[i])
	return ret



ensemble = []

for i in range(len(ID)):
	top3 = median[i].argsort()[-3:][::-1]
	top3 = map_category(top3)
	ensemble.append([ID[i]]+top3)


ensemble_file = open("ensemble.csv","w")

ensemble_file.write("%s,%s,%s,%s\n"%("FILE_ID","CATEGORY_ID0","CATEGORY_ID1","CATEGORY_ID2"))

for i in ensemble:
	ensemble_file.write("%s,%s,%s,%s\n"%tuple(i))

ensemble_file.close()


'''
median(a, 
       axis=None, 
       out=None,
       overwrite_input=False, 
       keepdims=False)

a = np,array([[10,  7,  4],
       [ 3,  2,  1]])

a = np.array([
		[ [1,2,3],
		  [2,3,4], ],
		[ [3,4,5],
		  [4,5,6], ],
		[ [5,6,7],
		  [7,8,9], ],
		[ [9,10,11],
		  [10,11,12], ]
	])
'''
