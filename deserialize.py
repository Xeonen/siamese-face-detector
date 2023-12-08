import torch
from SiameseModels import SiameseNetworkTripletLoss

model = SiameseNetworkTripletLoss()
model.load_state_dict(torch.load("model_e_4_l_0.1467277131297434.pth"))
model = model.to("cpu")
torch.save(model.state_dict(), "model_cpu.pth")