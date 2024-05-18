import os
import pandas as pd



file_path = '../src/data_spilited/'
files = os.listdir(file_path)
files = [x for x in files if x != '.DS_Store']
print('*' * 100)
print('\n       Variables are loading ... \n')

# print(files, len(files))

'''

Values 

'''
DRDrd = pd.read_excel(file_path + files[0], header=0)
DIRir_R2I = pd.read_excel(file_path + files[1], header=0)
WRr = pd.read_excel(file_path + files[2], header=0)
DRPrp = pd.read_excel(file_path + files[3], header=0)
sets_rpidc = pd.read_excel(file_path + files[4], header=0)
Tcd_ri = pd.read_excel(file_path + files[5], header=0)
Tcs_rp = pd.read_excel(file_path + files[6], header=0)
CEc = pd.read_excel(file_path + files[7], header=0)
COt = pd.read_excel(file_path + files[8], header=0)
MCc = pd.read_excel(file_path + files[9], header=0)
DIRir_I2R = pd.read_excel(file_path + files[10], header=0)
WCc = pd.read_excel(file_path + files[11], header=0)
FJPp = pd.read_excel(file_path + files[12], header=0)
MSp = pd.read_excel(file_path + files[13], header=0)
CFr = pd.read_excel(file_path + files[14], header=0)
MFr = pd.read_excel(file_path + files[15], header=0)
DCIci = pd.read_excel(file_path + files[16], header=0)
Tcc_cr = pd.read_excel(file_path + files[17], header=0)
FJRr = pd.read_excel(file_path + files[18], header=0)
dw_dr_wi = pd.read_excel(file_path + files[19], header=0)
FJCc = pd.read_excel(file_path + files[20], header=0)
Tct_rd = pd.read_excel(file_path + files[21], header=0)
MFt = pd.read_excel(file_path + files[22], header=0)
CPp = pd.read_excel(file_path + files[23], header=0)
TCFir = pd.read_excel(file_path + files[24], header=0)
WPp = pd.read_excel(file_path + files[25], header=0)

''' Obj 1: Cost Min Variables '''
# print('Obj1 ', '*' * 100)

# min z1

cpp = CPp[['Toronto', 'Hamilton', 'Scarborough', 'Mississauga']].values
CPp = CPp.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(CPp)

cfr = CFr[['Ottawa', 'Brampton', 'Etobicoke', 'Markham', 'Oshawa']].values
CFr = CFr.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(CFr)

cec = CEc.values
CEc = CEc.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(cec[0][0])
# print(CEc)
#
# print(TCFir)
tcfir = TCFir[['Kitchener', 'Etobicoke', 'Windsor', 'Markham', 'Vaughan']].values
TCFir = TCFir.rename(columns={'TCFir': 'index'}).set_index('index')
# print(TCFir)

dr = dw_dr_wi['dr'].values[0]
dw = dw_dr_wi['dw'].values[0]
wi = dw_dr_wi['WI'].values[0]

#
tcsrp = Tcs_rp[['Ottawa', 'Brampton', 'Etobicoke', 'Markham', 'Oshawa']].values
Tcs_rp = Tcs_rp.rename(columns={'Unnamed: 0': 'index'}).set_index('index')

#
tct_rd = Tct_rd[['Ottawa', 'Brampton', 'Etobicoke', 'Markham', 'Oshawa']].values
Tct_rd = Tct_rd.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(Tct_rd)
#
tcccr = Tcc_cr[['Scarborough', 'Markham', 'Vaughan']].values
Tcc_cr = Tcc_cr.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(Tcc_cr)

#

tcdri = Tcd_ri[['Ottawa', 'Brampton', 'Etobicoke', 'Markham', 'Oshawa']].values
Tcd_ri = Tcd_ri.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(Tcd_ri)
#

''' Obj 2: CO2 Min Variables'''
# print('Obj2 ', '*' * 100)
#
cot = COt[['Truck Petrol1', 'Truck Petrol2', 'Truck Diesel 1', 'Truck Diesel 2']].values
COt = COt.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(COt)
#

dcici = DCIci[['Kitchener', 'Etobicoke', 'Windsor', 'Markham', 'Vaughan']].values
DCIci = DCIci.rename(columns={'From (C) / To (I)': 'index'}).set_index('index')
# print(DCIci)
#
diriri2r = DIRir_I2R[['Ottawa', 'Brampton', 'Etobicoke', 'Markham', 'Oshawa']].values
DIRir_I2R = DIRir_I2R.rename(columns={'From (I) / To (R)': 'index'}).set_index('index')
# print(DIRir_I2R)
#
drprp = DRPrp[['Toronto', 'Hamilton', 'Scarborough', 'Mississauga']].values
DRPrp = DRPrp.rename(columns={'From (R) / To (P)': 'index'}).set_index('index')
# print(DRPrp)
# print(DRPrp.loc['Ottawa'])
# print(drprp)
#
drdrd = DRDrd[['Oshawa', 'Richmond Hill', 'Oakville']].values
DRDrd = DRDrd.rename(columns={'From (R) / To (D)': 'index'}).set_index('index')
# print(DRDrd)

#
dirirr2i = DIRir_R2I[['Kitchener', 'Etobicoke', 'Windsor', 'Markham', 'Vaughan']].values
DIRir_R2I = DIRir_R2I.rename(columns={'From (R) / To (I)': 'index'}).set_index('index')
# print(DIRir_R2I)

#
''' Obj 3: Social factor Max Variables'''
# print('Obj3 ', '*' * 100)

#
fjrr = FJRr[['Ottawa', 'Brampton', 'Etobicoke', 'Markham', 'Oshawa']].values
FJRr = FJRr.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(FJRr)
#

fjpp = FJPp[['Toronto', 'Hamilton', 'Scarborough', 'Mississauga']].values
FJPp = FJPp.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(FJPp)
# print(fjpp)
#
fjcc = FJCc[['Scarborough', 'Markham', 'Vaughan']].values
FJCc = FJCc.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(FJCc)

# MSp = pd.read_excel(file_path + files[13], header=0)
MSp = MSp.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# print(MSp.values)
# print(MSp.values.sum())

MFt = MFt.rename(columns={'Unnamed: 0': 'index'}).set_index('index')

MFr = MFr.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
# import numpy as np
# print(type(np.array(MFt.values)))
MCc = MCc.rename(columns={'Unnamed: 0': 'index'}).set_index('index')

WPp = WPp.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
WRr = WRr.rename(columns={'Unnamed: 0': 'index'}).set_index('index')
WCc = WCc.rename(columns={'Unnamed: 0': 'index'}).set_index('index')

