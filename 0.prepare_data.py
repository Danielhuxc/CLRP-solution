import pandas as pd
def main():
    with open('./extra_data/simple_english_wiki.txt') as f:
        contents = f.read()
    contents = contents.split('\n\n')
    for i in range(len(contents)):
        contents[i] = '\n'.join(contents[i].split('\n')[1:])
    length = 1000
    all_data_split = []
    for txt in contents:
        [all_data_split.append(txt[0+i:length+i]) for i in range(0, len(txt), length)]
    with open('./extra_data/cbt_valid.txt') as f:
        cbt_v = f.read()
    with open('./extra_data/cbt_test.txt') as f:
        cbt_te = f.read()
    with open('./extra_data/cbt_train.txt') as f:
        cbt_tr = f.read()
    cbt = cbt_v+cbt_te+cbt_tr
    cbt = cbt.replace('`',"'")
    cbt = cbt.replace("''",'"')
    _=[all_data_split.append(cbt[0+i:length+i]) for i in range(0, len(cbt), length)]
    df = pd.DataFrame()
    df['excerpt'] = all_data_split
    df.to_csv('./extra_data/extra_excerpt.csv',index=False)

if __name__ == "__main__":
    main()
