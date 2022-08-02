import torch
import math
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-5, 5, 1000, device=device, dtype=dtype)
y = 3*x**2 + 2*x + 1
 

a1 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b1 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c1 = torch.randn((), device=device, dtype=dtype, requires_grad=True)

ada_coef_a = 0
ada_coef_b = 0
ada_coef_c = 0

learning_rate = 2
for t in range(100):
   
    pred_y = a1 * (x**2) + b1 * x + c1
    loss = (0.5*((pred_y-y).pow(2))).sum()
    loss.backward()
    ada_coef_a = ada_coef_a + a1.grad**2
    ada_coef_b = ada_coef_a + b1.grad**2
    ada_coef_c = ada_coef_a + c1.grad**2

   
    with torch.no_grad():
        a1 -= (learning_rate/np.sqrt(ada_coef_a)) * a1.grad
        b1 -= (learning_rate/np.sqrt(ada_coef_b)) * b1.grad
        c1 -= (learning_rate/np.sqrt(ada_coef_c)) * c1.grad

        # Manually zero the gradients after updating weights
        a1.grad = None
        b1.grad = None
        c1.grad = None

plt.plot(x.detach().numpy(), y.detach().numpy())
plt.plot(x.detach().numpy(), pred_y.detach().numpy())
plt.show()