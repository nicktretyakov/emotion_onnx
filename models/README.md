# Models directory

Place your exported ONNX models here:

- `face_emotion.onnx` — facial expression model (expected input: `[1, 3, 224, 224]` or `[1, 3, 48, 48]`)
- `audio_emotion.onnx` — speech emotion model
- `text_emotion.onnx` — (optional) text emotion model

## How to export a model from PyTorch

```python
import torch
import torch.nn as nn

# Example: simple model
class DummyEmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 7)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = DummyEmotionNet()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "face_emotion.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

print("Exported face_emotion.onnx")
```

You can replace the dummy model with a real trained network (ResNet, EfficientNet, etc.).
