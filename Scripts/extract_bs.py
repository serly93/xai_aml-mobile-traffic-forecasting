#this script takes the per service trace and split them per BSs

from pandas import read_csv
import numpy as numpy
import csv
import glob
import os

for filename in sorted(glob.glob('*.txt')): #Base Station trace from which active RNTI will be extracted and parsed
	print((filename))
	dataset = read_csv(filename, header=None, sep='\t',keep_default_na=False) 
	dataset.columns = ['squareId','Time','country_code','sms_in','sms_out','call_in','call_out','internet_traffic']

	print('dataset loaded')
	
	path = "../Datasets/Milan/PerBS"

	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)
	
	print ("Extracting BSs...")
	dict = {}
	userList=[]
	for n in range (0,len(dataset)):
		val=dataset['squareId'].iloc[n]

		userList.append(val)
		userList.append(dataset['Time'].iloc[n])
		userList.append(dataset['country_code'].iloc[n])
		userList.append(dataset['sms_in'].iloc[n])
		userList.append(dataset['sms_out'].iloc[n])
		userList.append(dataset['call_in'].iloc[n])
		userList.append(dataset['call_out'].iloc[n])
		userList.append(dataset['internet_traffic'].iloc[n])

		if val not in dict:
			dict[val] = []
			dict[val].append(userList)
			userList=[]
		else:

			dict[val].append(userList)
			userList=[]
	for key in dict:
		f=open('../Datasets/Milan/PerBS/%s.txt' %key,'a')
		for y in dict[key]:
			str_y = str(y)[1 : -1]		
			
			print(str_y, end="\n", file=f)

			
		f.close()

print('BSs extracted !')

