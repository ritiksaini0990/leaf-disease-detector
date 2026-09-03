import torch

from models.attention_model import CornDiseaseAttentionNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CornDiseaseAttentionNet(
    num_classes=4
).to(device)


# Create one fake image batch
x = torch.randn(
    4, 3, 224, 224
).to(device)


output = model(x)


print("Device:", device)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Model created successfully!")