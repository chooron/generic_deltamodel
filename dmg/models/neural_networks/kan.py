import torch
import torch.nn as nn
from timekan.models.tkan_lstm import tKANLSTM
from timekan.utils.datasets import mackey_glass

class RecurrentKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.tkan = tKANLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            return_sequences=False,
            bidirectional=True,
            kan_type='fourier',
            sub_kan_configs={'gridsize': 50, 'addbias': True}
        )
        self.regressor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        features = self.tkan(x)
        return self.regressor(features).squeeze(-1)

x_train, y_train, x_test, y_test = mackey_glass()

model = RecurrentKAN(input_dim=1, hidden_dim=16)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/10, Training MSE: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_mse = criterion(test_outputs, y_test).item()
    print(f"Test MSE: {test_mse:.4f}")