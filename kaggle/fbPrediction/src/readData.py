import zipfile
import pandas as pd

def loadData(typ):
	if(typ == 'training'):
		zf = zipfile.ZipFile('../data/train.csv.zip')
		data = pd.read_csv(zf.open(zf.namelist()[0]))
		return data
	if(typ == 'test'):
                zf = zipfile.ZipFile('../data/test.csv.zip')
                data = pd.read_csv(zf.open(zf.namelist()[0]))
                return data