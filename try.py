import random
if __name__=="__main__":
    a="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/wiki1m.txt"
    save_p="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/wiki15w.txt"
    with open(a,"r") as f:
        data=f.readlines()
    random.shuffle(data)
    new_data=[]
    for i in data[0:200000]:
        if len(i.strip().split(" "))>3:
            new_data.append(i)
        if len(new_data)==150000:
            break
    with open(save_p,"w") as f:
        f.writelines(new_data)