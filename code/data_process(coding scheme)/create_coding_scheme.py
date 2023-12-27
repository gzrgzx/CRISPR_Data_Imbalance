# import sys
# import pickle as pkl
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from sklearn.utils import Bunch
#
#
# cd33_data = pd.read_pickle("cd33 (dataset II-1).pkl")
#
# cd33_mut = cd33_data[0]
#
#
#
# print(cd33_mut.columns)
#
# print(cd33_mut)
#
# print(sum(cd33_mut['Day21-ETP-binarized']))
#
# target_data = {'sgrna':cd33_mut['30mer'],'otdna':cd33_mut['30mer_mut'],'Day21-ETP':cd33_mut['Day21-ETP'],'label':cd33_mut['Day21-ETP-binarized']}
#
# target_data = pd.DataFrame(target_data)
#
# target_data.to_csv('cd33_offTarget.csv')

# -*- coding: utf-8 -*-
"""Python file offtargetCreateGuideSeqDataset.py for datasets generation."""

import sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import Bunch

# we import the off-targets from guideseq
fpath = 'cd33_offTarget.csv'
dfGuideSeq = pd.read_csv(fpath, sep=',')

# # 处理数据
#
# target_dna_list = []
# target_rna_list = []
# target_label = []
#
# for n in range(len(dfGuideSeq)):
#     target_dna = list(dfGuideSeq.loc[n, 'off_seq'])
#     target_rna = list(dfGuideSeq.loc[n, 'on_seq'])
#     if target_rna[-3] == 'N':
#         if target_dna[-3] >= 'A' and target_dna[-3] <= 'Z':
#             target_rna[-3] = target_dna[-3]
#
#     for i in range(len(target_dna)):
#         if target_dna[i] >= 'a' and target_dna[i] <= 'z':
#             target_dna[i] = chr(ord(target_dna[i]) - ord('a') + ord('A'))
#         if target_dna[i] == 'N':
#             target_dna[i] = target_rna[i]
#
#     target_dna = ''.join(target_dna)
#     target_rna = ''.join(target_rna)
#     target_rna_list.append(target_rna)
#     target_dna_list.append(target_dna)
#     if dfGuideSeq.loc[n, 'reads'] == 0:
#         target_label.append(0)
#     else:
#         target_label.append(1)
#
#
# target_data = {'sgrna':target_rna_list,'otdna':target_dna_list,'label':target_label}
#
# target_data = pd.DataFrame(target_data)
#
# target_data.to_csv('SITE-Seq_offTarget.csv')


# dfGuideSeq = dfGuideSeq.drop_duplicates(
#     subset=['otSeq'], keep=False, ignore_index=True)


# we encode the new validated off-targets
# as described in the references
def one_hot_encode_seq2(data):
    """One-hot encoding of the sequences."""
    # define universe of possible input values
    alphabet = 'ATGC'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    # print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    # print(onehot_encoded)
    # invert encoding
    inverted = int_to_char[np.argmax(onehot_encoded[0])]
    # print(inverted)
    return onehot_encoded


def one_hot_encode_seq(data):
    """One-hot encoding of the sequences."""
    # define universe of possible input values
    alphabet = 'AGCT'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    # print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    # print(onehot_encoded)
    # invert encoding
    inverted = int_to_char[np.argmax(onehot_encoded[0])]
    # print(inverted)
    return onehot_encoded


def flatten_one_hot_encode_seq(seq):
    """Flatten one hot encoded sequences."""
    return np.asarray(seq).flatten(order='C')

#6*23编码方案
# channel_first = {'A':1,'T':2,'G':3,'C':4}
encoded_list = []
for n in range(len(dfGuideSeq)):
    target_dna = dfGuideSeq.loc[n, 'otdna']
    target_rna = dfGuideSeq.loc[n, 'sgrna']
    arr1 = one_hot_encode_seq2(target_dna)
    arr2 = one_hot_encode_seq2(target_rna)
    arr = np.zeros((23, 6))
    for m in range(len(arr1)):
        if arr1[m] == arr2[m]:
            arr[m][0:4] = arr1[m]
            arr[m][4:6] = [0, 0]
        else:
            temp1 = arr1[m].index(1)
            temp2 = arr2[m].index(1)
            if temp1 < temp2:
                arr[m][0:4] = np.add(arr1[m], arr2[m])
                arr[m][4:6] = [1, 0]
            else:
                arr[m][0:4] = np.add(arr1[m], arr2[m])
                arr[m][4:6] = [0, 1]
    arr = flatten_one_hot_encode_seq(arr)
    encoded_list.append(arr)

dfGuideSeq['encoded'] = pd.Series(encoded_list)

# we consider the encoded column for the 4x23 encoding
dfGuideSeq4x23 = pd.DataFrame(dfGuideSeq['encoded'].values.tolist())

# print(dfGuideSeq4x23[0:2])

guideseq4x23 = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=(dfGuideSeq4x23.values).reshape(-1, 23, 6, order='C'))
    # images=dfGuideSeq4x23.values)
plt.imshow(guideseq4x23.images[0], cmap='Greys')

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encoded6x23cd33withoutTsai.pkl", "wb")
    pkl.dump(guideseq4x23, pickle_out)
    pickle_out.close()


#7*23编码方案
# channel_first = {'A':1,'T':2,'G':3,'C':4}
encoded_list = []
for n in range(len(dfGuideSeq)):
    target_dna = dfGuideSeq.loc[n, 'otdna']
    target_rna = dfGuideSeq.loc[n, 'sgrna']
    arr1 = one_hot_encode_seq2(target_dna)
    arr2 = one_hot_encode_seq2(target_rna)
    arr = np.zeros((23, 8))
    for m in range(len(arr1)):
        if arr1[m] == arr2[m]:
            arr[m][0:4] = arr1[m]
            arr[m][4:7] = [0, 0, 0]
        else:
            temp = np.add(arr1[m], arr2[m]).tolist()
            temp1 = arr1[m].index(1)
            temp2 = arr2[m].index(1)
            if temp1 < temp2:
                arr[m][0:4] = temp
                arr[m][4:6] = [1, 0]
            else:
                arr[m][0:4] = temp
                arr[m][4:6] = [0, 1]
            if temp == [1,0,1,0] or temp == [0,1,0,1]:
                arr[m][6] = 1
            else:
                arr[m][6] = -1
        if m >= 20:
            arr[m][7] = 1
        else:
            arr[m][7] = 0
    arr = flatten_one_hot_encode_seq(arr)
    encoded_list.append(arr)

dfGuideSeq['encoded'] = pd.Series(encoded_list)

# we consider the encoded column for the 4x23 encoding
dfGuideSeq4x23 = pd.DataFrame(dfGuideSeq['encoded'].values.tolist())

# print(dfGuideSeq4x23[0:2])

guideseq4x23 = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=(dfGuideSeq4x23.values).reshape(-1, 23, 8, order='C'))
    # images=dfGuideSeq4x23.values)
plt.imshow(guideseq4x23.images[0], cmap='Greys')

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encoded7x23cd33withoutTsai.pkl", "wb")
    pkl.dump(guideseq4x23, pickle_out)
    pickle_out.close()


#4*23编码
enc_dna_list = []
enc_rna_list = []
encoded_list = []
for n in range(len(dfGuideSeq)):
    target_dna = dfGuideSeq.loc[n, 'otdna']
    target_rna = dfGuideSeq.loc[n, 'sgrna']
    arr1 = one_hot_encode_seq(target_dna)
    arr2 = one_hot_encode_seq(target_rna)
    arr = np.zeros((23, 4))
    for m in range(len(arr1)):
        if arr1[m] == arr2[m]:
            arr[m] = arr1[m]
        else:
            arr[m] = np.add(arr1[m], arr2[m])
    arr = flatten_one_hot_encode_seq(arr)
    enc_dna_list.append(np.asarray(arr1).flatten('C'))
    enc_rna_list.append(np.asarray(arr2).flatten('C'))
    encoded_list.append(np.asarray(arr).flatten('C'))

dfGuideSeq['enc_dna'] = pd.Series(enc_dna_list)
dfGuideSeq['enc_rna'] = pd.Series(enc_rna_list)
dfGuideSeq['encoded'] = pd.Series(encoded_list)

# we consider the encoded column for the 4x23 encoding
dfGuideSeq4x23 = pd.DataFrame(dfGuideSeq['encoded'].values.tolist())

# save the encoded results as 4x23 images
# we put the results in bunch
guideseq4x23 = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=(dfGuideSeq4x23.values).reshape(-1, 4, 23, order='F'))
plt.imshow(guideseq4x23.images[0], cmap='Greys')

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encoded4x23cd33withoutTsai.pkl", "wb")
    pkl.dump(guideseq4x23, pickle_out)
    pickle_out.close()


# we have to transform the RNA and DNA sequences to
# a 8x23 image
# we create a new column on the crispor data df
# we structure the images as the mnist dataset
# digits.target_names is the name
# digits.target is the binary classification
# digits.images is the 8x23 pixels of the image
# a. we do it for the put. off-target

# we store the image in im
im = np.zeros((len(dfGuideSeq), 8, 23))

cnt = 0
for n in range(len(dfGuideSeq)):
    arr1 = one_hot_encode_seq(dfGuideSeq.loc[n, 'sgrna'])
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(dfGuideSeq.loc[n, 'otdna'])
    arr2 = np.asarray(arr2).T
    arr = np.concatenate((arr1, arr2))
    im[n] = arr
    cnt += 1

# we put the results in bunch
guideseq8x23 = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=im)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encoded8x23cd33withoutTsai.pkl", "wb")
    pkl.dump(guideseq8x23, pickle_out)
    pickle_out.close()

plt.imshow(guideseq8x23.images[0], cmap='Greys')
plt.savefig('guideseq8x23.pdf')


#加区域分开编码
new9x23 = np.zeros((im.shape[0],9,23))
position = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
for n in range(im.shape[0]):
    for m in range(im.shape[1]):
        for k in range(im.shape[2]):
            if im[n,m,k] != 0:
                new9x23[n,m,k] = 1
    new9x23[n,8] = position

print("加区域分开编码")
print(new9x23[0])

new9x23 = new9x23.transpose((0,2,1))
print(new9x23[0])

new_coding = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=new9x23
)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encodedposition9x23cd33withoutTsai.pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()



# print(new9x23[0])


#加区分碱基错配类型和区域划分

encoded_list = np.zeros((im.shape[0],5,23))
for n in range(im.shape[0]):
    for m in range(im.shape[2]):
        arr1 = im[n,0:4,m].tolist()
        # print(arr1)
        arr2 = im[n,4:8,m].tolist()
        # print(arr2)
        arr = []
        if arr1 == arr2:
            arr = [0,0,0,0,0]
        else:
            arr = np.add(arr1,arr2).tolist()
            if (arr == [1,1,0,0]) or (arr == [0,0,1,1]):
                arr.append(1)
            else:
                arr.append(-1)
        encoded_list[n,:,m] = arr


new9x23 = np.zeros((im.shape[0],14,23))
position = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
for n in range(im.shape[0]):
    for m in range(im.shape[1]):
        for k in range(im.shape[2]):
            if im[n,m,k] != 0:
                new9x23[n,m,k] = 1
    new9x23[n,8:13] = encoded_list[n]
    new9x23[n,13] = position


# new9x23 = np.concatenate((new9x23,encoded_list),axis=1)

print("加区分碱基错配类型和区域划分")
print(new9x23[0])

new9x23 = new9x23.transpose((0,2,1))
print(new9x23[0])

new_coding = Bunch(
    # target_names=df_enc['name'].values,
    target=dfGuideSeq['label'].values,
    images=new9x23
)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encodedmismatchtype14x23cd33withoutTsai.pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()


