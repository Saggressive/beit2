import random
import json
if __name__=="__main__":
    a="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/wiki1m_for_simcse.txt"
    j="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/flickr_random_captions.json"
    save_p="/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/wiki1m_flickr_for_simcse.txt"
    with open(a,"r") as f:
        data=f.readlines()
    with open(j,"r") as f:
        j_data=json.load(f)
    j_text=[i['caption'].strip() for i in j_data]
    j_text=[i+"\n" for i in j_text]
    print(len(j_text))
    data.extend(j_text)
    with open(save_p,"w") as f:
        f.writelines(data)
