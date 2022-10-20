if __name__=="__main__":
    a="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/wiki1m.txt"
    save_p="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/new2_wiki1m.txt"
    with open(a,"r") as f:
        data=f.readlines()
    new_data=[]
    for i in data:
        if i.strip()=="" or i.strip()==None:
            len(i.strip()<=1)
            continue
        new_data.append(i)
    with open(save_p,"w") as f:
        f.writelines(new_data)