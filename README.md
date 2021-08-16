# CLRP-solution
### requirements:
numpy==1.20.2 \
pandas==1.2.4 \
transformers==4.5.1 \
torch==1.9.0+cu111 \
sklearn==0.0 \
tqdm==4.60.0

24GB VRAM

### prep:
download pretrained roberta-large and deberta-large from: \
https://huggingface.co/roberta-large \
https://huggingface.co/microsoft/deberta-large \
and save it in \
./pretrained/roberta-large\
./pretrained/deberta-large
<br/><br/>
download \
Children's Book Test from: \
https://research.fb.com/downloads/babi/ \
Simple Wiki Dump from: \
https://github.com/LGDoor/Dump-of-Simple-English-Wiki \
and save it as follows \
./extra_data/cbt_test.txt \
./extra_data/cbt_train.txt \
./extra_data/cbt_valid.txt \
./extra_data/simple_english_wiki.txt

CLRP training data goes to \
./data/train.csv \
./data/test.csv

### train from scratch:
./run_train.sh \
takes about 30 hours

### predict:
python 4.predict.py ./{path_to_source_file}.csv ./{path_to_save}.csv 3 0 ./models/roberta_2/ ./models/deberta_1/ ./models/deberta_2/ \
make sure the column name is 'excerpt' in source csv file

### solution writeup:
https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258095
