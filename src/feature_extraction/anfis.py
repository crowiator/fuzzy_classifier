import torch
from anfis import AnfisNet, MSELoss

inputs  = torch.tensor(X_df.values, dtype=torch.float32)
targets = torch.tensor(pd.get_dummies(y).values, dtype=torch.float32)

model = AnfisNet(n_inputs=inputs.shape[1], n_rules=8, n_outputs=5)  # 5 AAMI tried
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    output = model(inputs)
    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()