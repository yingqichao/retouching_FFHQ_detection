import os
import collections

file_names = ['00_34_closingeye.txt',
              'eye_close.txt',
              'sunglasses.txt',
              '00_34_baby.txt',
              'child_35.txt'
              ]
ban_set = set()

for file_name in file_names:
    with open(os.path.join('/hotdata/FFHQ', file_name), 'r') as f:
        # ban_list += [line.strip() for line in f.readlines()]
        for line in f.readlines():

            ban_set.add(line.strip())

print(len(ban_set))
