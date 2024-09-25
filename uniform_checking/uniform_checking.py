import onnx
import onnxruntime
import numpy as np
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_uniform(ort_session, x):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    _, predicted = torch.max(torch.from_numpy(ort_outs[0]), 1)
    label = True if predicted.item() == 1 else False
    return label