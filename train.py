import torch
import torch.nn as nn
from model import *
from dataset import *

def main():
    trainloader, validloader = get_dataloaders()
    print('Data Preprocessed and got DataLoaders...')
    
    model = Face_Emotion_CNN()
    if torch.cuda.is_available():
        model.cuda()
        print('GPU Found!!!, Moving Model to CUDA.')
    else:
        print('GPU not found!!, using model with CPU.')

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


if __name__ == '__main__':
    main()