# ONNX Tutorial for ML Engineers in Embedded ADAS Systems

## Objective
Convert a PyTorch model to ONNX format and run it with ONNX Runtime for embedded ADAS applications.

---

## 1. Install Required Packages

```bash
pip install torch torchvision onnx onnxruntime onnxsim
```

---

## 2. Define and Train a Simple PyTorch Model

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x

model = SimpleCNN()
dummy_input = torch.randn(1, 3, 32, 32)
```

---

## 3. Export to ONNX

```python
torch.onnx.export(model, dummy_input, "simple_cnn.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                  opset_version=11)
```

---

## 4. Simplify ONNX Model (Optional)

```bash
python -m onnxsim simple_cnn.onnx simple_cnn_sim.onnx
```

---

## 5. Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("simple_cnn_sim.onnx")
input_name = session.get_inputs()[0].name

input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)
result = session.run(None, {input_name: input_data})

print("Model output:", result[0])
```

---

## 6. Debug & Optimization

- Use `Netron` to visualize model: https://netron.app/
- Quantization (ONNX Runtime Toolkit): https://onnxruntime.ai/docs/performance/quantization.html

---

## 7. Deployment to Embedded System

1. Convert ONNX to target runtime format (e.g., Qualcomm QNN or TensorRT)
2. Cross-compile ONNX Runtime if needed
3. Integrate into C++ inference pipeline on embedded OS (Linux/QNX/AAOS)

---

## References

- https://onnx.ai/
- https://github.com/onnx/tutorials
- https://onnxruntime.ai/