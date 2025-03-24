import os

pat = './Alergico'
files = os.listdir(pat)
re={}
cnt=0
# Iterate over the files
for filename in files:
    id=filename.split('_')[0]
    re[id]=0
    cnt+=1

print(str(len(re))+ ' and ' + str(cnt))