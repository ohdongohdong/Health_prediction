import csv
import numpy as np

csvlist = []
with open('../data/raw_data.csv') as csvfile:
    values = csv.reader(csvfile)
    tag = values.next() # row0(tag)
    # row1 = values.next()
    # row2 = values.next()
    """or"""
    for row in values:
        # print row # ro something else
        # row_nan = ['nan' if x=='' else x for x in row]
        # row_np = np.array(row_nan).astype(np.float)
        # csvarray = np.append(csvarray,row_np)
        csvlist.extend(row) # all values of data

csvlist_nan = ['nan' if x == '' else x for x in csvlist]
csvarray = np.array(csvlist_nan).astype(np.float)
csvmatrix = csvarray.reshape((-1, len(tag)))  # (2210067, 47)

# idx = label(2002y = 0, 2003y = 1 ...)
years, years_idx = np.unique(csvmatrix[:, 0], return_inverse=True)  # total years : 12
patient_id, ptid_idx = np.unique(csvmatrix[:, 1], return_inverse=True)  # total patient : 596284

data_raw = np.empty([patient_id.size, years.size, len(tag)]) # (596284,12,47)
data_raw[:] = np.nan
data_raw[ptid_idx, years_idx, :] = csvmatrix  # (596284,12,47)

# data_02y_08y = data_raw[:,0:7]
# data_09y_13y = data_raw[:,7:]

# find patient that hasn't nan values of years
gooddata_candidate_flag = np.sum(np.isnan(data_raw[:, :, 0]), axis=1)
number_of_gooddata_candidate = np.sum(gooddata_candidate_flag == 0)

# except the patients that has nan values of years
gooddata_candidate = data_raw[gooddata_candidate_flag == 0, :, :]

gooddata = gooddata_candidate   # (11876, 12, 47)


# data preprocessing
# includes:
# 1. some part of 'nan' to '0' (for 'whether or not' questions)
# 2. exclude some data that only exist in part of the "time index"
(gooddata[:, :, 17:45])[np.isnan(gooddata[:, :, 17:45])] = 0

# delete ['WAIST', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE','GLY_CD', 'OLIG_OCCU_CD', 'OLIG_PH']
gooddata = np.delete(gooddata, np.s_[5, 10, 11, 12, 14, 15, 16], axis=2)
goodtag = np.delete(tag, np.s_[5, 10, 11, 12, 14, 15, 16]) # tag 47 -> 40

# except the patients that has nan values
gooddata = gooddata[np.sum(np.isnan(gooddata), axis=(1, 2)) == 0, :, :] # (11692, 12, 40)

# raw_data: original data stored in .csv file
# gooddata: exclude person with 'nan' data and some incomplete healthcare components
# goodtag: tags (exclude incomplete healthcare components)
np.save('raw_data_set', data_raw)
np.save('gooddata', gooddata)
np.save('goodtag', goodtag)
# nancount_02_08 = np.sum(np.isnan(data_02_08[:,:,0]),axis=1)
# data_good = data[nancount_02_08==0,:,:]
