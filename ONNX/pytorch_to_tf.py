import sys
sys.path.append('../')

from PyTorch.model import *

import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare


def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model

def torch2tf(model):
    dummy_input = torch.randn(1,1,48,48)
    torch.onnx.export(model,dummy_input, './models/onnx_model.onnx')

    onnx_model = onnx.load('./models/onnx_model.onnx')

    tf_model = prepare(onnx_model)
    tf_model.export_graph('./models/model_simple.pb')
    


if __name__ == '__main__':
    model_path = '../PyTorch/models/best_model.pt'
    model = load_trained_model(model_path)
    torch2tf(model)
