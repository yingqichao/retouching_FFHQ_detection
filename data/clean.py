import os

with open("E:\\FFHQ_megvii_three_process\\three_process_new.txt", "a") as fw:
    with open("E:\\FFHQ_megvii_three_process\\three_process.txt", "r") as fr:
        for line in fr:
            # cells = line.strip('\n').split('\t')
            # assert len(cells) == 2, "strange! len of label exceed 2, check!"
            if not "groupshare" in line:
                fw.write(line)
                print(line)



# testdata = respond['result']
# fw = open("E:/testfile", 'w')
# fw.write(testdata)
# fw.close()
# fw = open("E:/testfile", 'r')
# decoded = base64.b64decode(fw.readline())
# fw = open("E:/testfile_1.png", 'wb')
# fw.write(decoded)
# fw.close()