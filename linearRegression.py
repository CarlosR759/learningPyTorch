import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

x = np.array([x for x in range(100)])
x = x.reshape(-1,1)
y = 46 + 2 * x.flatten()

x
y

plt.scatter(x,y, label='Initial Data')
plt.title('Pre Pytorch')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


#Normalizing data
x_mean, x_std = x.mean(), x.std()
x_normalized = (x - x_mean) / x_std
x_tensor = torch.tensor(x_normalized, dtype=torch.float32)

print(x_tensor.shape)


y_mean, y_std = y.mean(), y.std()
y_normalized = (y - y_mean) / y_std
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)


class LinearRegressionModel(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.linear = nn.Linear(in_features, out_features)

	def forward(self, x):
		return self.linear(x).squeeze(1)
		
in_features = 1
out_features = 1
model = LinearRegressionModel(in_features, out_features)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

num_epoch = 10

for epoch in range(num_epoch):
	#Foward pass
	output = model(x_tensor)
	# Loss calculation
	loss = criterion(output, y_tensor)
	# Backpropagation
	optimizer.zero_grad() #Cleaning the gradients
	loss.backward()
	optimizer.step()
	print(f'Epoch [{epoch +1}/{num_epoch}], Loss: {loss: {loss.item():.2f}}')
	

#Testing model
new_x = 121
new_x_normalized = (new_x - x_mean) / x_std
new_x_tensor = torch.tensor(new_x_normalized, dtype = torch.float32).view(1,-1)

model.eval()
with torch.no_grad():
	prediction_normalized = model(new_x_tensor)
	
prediction_denormalized = prediction_normalized.item() * y_std + y_mean
print(f"Predicted value for x = {new_x}: {prediction_denormalized}")


plt.scatter(x, y, label='Initial_data')

fit_line = model(x_tensor).detach().numpy() * y_std + y_mean
plt.plot(x, fit_line, 'r', label='PyTorch Line')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PyTorch with Prefictions')
plt.show()

