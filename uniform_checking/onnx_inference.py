import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Initialize model with the pretrained weights
model = SimpleCNN(num_classes=2)
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
model.load_state_dict(torch.load('model.pth', map_location=map_location))

# set the model to inference mode
model.eval()
# img = Image.open("./data/positive/20240728_172006_id2.jpg").convert('RGB')
img = Image.open("./data/negative/53394_f.webp").convert('RGB')
# x = torch.randn(1, 3, 224, 224, requires_grad=True)

x = transform(img)
x = x.unsqueeze(0)
onnx_model = onnx.load("uniform_check.onnx")
onnx.checker.check_model(onnx_model)
torch_out = model(x)
ort_session = onnxruntime.InferenceSession("uniform_check.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
_, predicted = torch.max(torch.from_numpy(ort_outs[0]), 1)
label = " This person is wearing Swinburne Uniform" if predicted.item() == 1 else "This person is not wearing Swinburne Uniform"
print("fuck: ", label)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")