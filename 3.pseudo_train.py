#args:
#1. type of model: 'ro' or 'de'
#2. label path
#3. save path
#4. mode: 0=mix 1=5fold

from components.train import train_pseudo, train_pseudo_5fold
from components.util import generate_config
import sys
import numpy as np

def main():
    ###
    # training using extra training data
    ###
    config = generate_config(sys.argv[1],'None',sys.argv[3],'custom','2')
    if sys.argv[4]=='0':
        min_valid_loss = train_pseudo(config,sys.argv[2])
        print(min_valid_loss)
    else:
        min_valid_loss = train_pseudo_5fold(config,sys.argv[2])
        print(min_valid_loss)

if __name__ == "__main__":
    main()
