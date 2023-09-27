import os

f = open('../weak.txt', 'r')
fw = open('../weak_new.txt', 'w')

lines = f.readlines()

for line in lines:
    line = line.strip()
    if line != '':
        temp = line.split('\t')
        char = temp[0]
        if temp[1] == 'B-Disease':
            label = 'B-disease'
        elif temp[1] == 'I-Disease':
            label == 'I-disease'
        else:
            label = temp[1]
        new_line = char + ' ' + label + '\n'
        fw.write(new_line)
    else:
        fw.write('\n')

f.close()
fw.close()