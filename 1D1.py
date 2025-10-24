#complete 1D code resnet
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

from scipy.io import savemat

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import seaborn as sns
import random
from torch.utils.data import IterableDataset

# Set the random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define the neural network with ResNet blocks and skip connections
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.activation = nn.GELU()    #ReLU()  LeakyReLU()     GELU()   Tanh()   Sigmoid()
        self.fc2 = nn.Linear(out_channels, out_channels)

        # Skip connection (identity mapping if input and output have the same shape)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += self.shortcut(x)  # Adding the skip connection
        out = self.activation(out)
        return out

class PINN(nn.Module):
    def __init__(self, hidden_size=25):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(2, hidden_size)

        # Define the ResNet blocks with skip connections
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(hidden_size, hidden_size),
            ResNetBlock(hidden_size, hidden_size),
            ResNetBlock(hidden_size, hidden_size),
            ResNetBlock(hidden_size, hidden_size)
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.nn.functional.gelu(x)   #relu(x)   leaky_relu(x)  gelu(x)   tanh(x)  sigmoid(x)  # Initial input processing

        # Pass through the ResNet blocks
        x = self.resnet_blocks(x)

        # Output layer (no activation here)
        x = self.output_layer(x)
        return x

# Updated Loss Function to include the new term
def loss_function(x, z, F):
    weightvalue = 10
    N = x.shape[0]
    M = z.shape[0]
    loss = torch.tensor(0.0, requires_grad=True)

    # Compute the derivative dF/dz
    dF_dz = torch.autograd.grad(F.sum(), z, grad_outputs=torch.ones_like(F.sum().to('cuda')), create_graph=True)[0]

    # Now compute the terms you want in your loss function
    term1Complete = 0
    term2Complete = 0
    term3Complete = 0

    for i in range(N):
      term1 = 0
      term1_vector = (dF_dz[i,:]**2 - 1)**2 * torch.exp(-(z[0,:]**2) / 2)
      term1 = term1_vector.sum()
      term1Complete += (term1 / M)

      term2 = 0
      for j in range(i):
        term2_vector = dF_dz[j,:] * torch.exp(-(z[0,:]**2) / 2)
        term2 += term2_vector.sum()
      term2Complete += (term2 / (N * M))**2

      term3_vector = dF_dz[i,:] * torch.exp(-(z[0,:]**2) / 2)
      term3Complete += term3_vector.sum()

    term1Complete = term1Complete / N
    term2Complete = term2Complete / N

    term3Complete = (term3Complete / (N * M))**2
    loss = term1Complete + term2Complete + weightvalue*term3Complete
   
    return loss, term1Complete.item(), term2Complete.item(), term3Complete.item()

# Updated Training Function
def train_pinn_updated(load_pretrained=False, pretrained_path="model_parameters.pth"):
    model = PINN().to('cuda')

    # Initialize weights and biases using Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Load pretrained model parameters if specified
    if load_pretrained:
        model.load_state_dict(torch.load(pretrained_path))
        print("Loaded pretrained model parameters.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    epochs = 2000
    loss_history = []
    term1_history = []
    term2_history = []
    term3_history = []

    # Sample points
    x = torch.linspace(0, 1, 201).view(-1, 1).to('cuda')
    M = 2
    z = torch.linspace(-M, M, 201).view(-1, 1).to('cuda')
    z.requires_grad = True

    # Create a grid of x and z values with indexing argument
    X, Z = torch.meshgrid(x.squeeze(), z.squeeze(), indexing='ij')

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        inputs = torch.cat([X.unsqueeze(-1), Z.unsqueeze(-1)], dim=-1).to('cuda')
        F = model(inputs.view(-1, 2)).view(X.shape)

        # Compute the loss
        loss, term1, term2, term3 = loss_function(X, Z, F)
        loss_history.append(loss.item())
        term1_history.append(term1)
        term2_history.append(term2)
        term3_history.append(term3)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step(loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

    return model, loss_history, term1_history, term2_history, term3_history, X, Z

# Updated Visualization function with adjusted vertical axis labels and further shifted plots
def visualize_model_updated(model, loss_history, term1_history, term2_history, term3_history, x, z):

    # Plotting the loss over epochs
    plt.figure(figsize=(3.5, 2))
    bc_history = np.array(term2_history) + np.array(term3_history)
    plt.plot(loss_history, label=r'$\text{Total}$', color='blue')
    plt.plot(term1_history, label=r'$|\text{Energy}|^2$', color='red')
    plt.plot(bc_history, label=r'$|\text{BC}|^2$', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    # plt.grid()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)
    plt.savefig('1D1_loss.png')  # Save the figure
    plt.close()  # Close the figure
    
    savemat('1D1_loss_data.mat', {
    'loss_history': loss_history,
    'term1_history': term1_history,
    'term2_history': term2_history,
    'term3_history': term3_history,
    'bc_history': bc_history,
    })

    # Generate points for visualization with requires_grad=True
    n_x = len(torch.unique(x))  # Number of unique values along x-axis
    n_z = len(torch.unique(z))  # Number of unique values along z-axis
    grid_shape = (n_x, n_z)

    # Concatenate after enabling gradient tracking
    xz_input = torch.cat([x.unsqueeze(-1), z.unsqueeze(-1)], dim=-1).to('cuda')

    # Forward pass for the generated points
    F_pred = model(xz_input).view(grid_shape)

    # Calculate gradients
    dF_dz = torch.autograd.grad(F_pred.sum(), z, grad_outputs=torch.ones_like(F_pred.sum()), create_graph=True)[0]

    # Reshape flattened inputs to 2D grids for plotting
    x_np = x.view(grid_shape).detach().cpu().numpy()
    z_np = z.view(grid_shape).detach().cpu().numpy()

    # Plotting the 3D surface of F(x, z)
    fig_3d_surface = plt.figure(figsize=(2.2, 2))
    ax_3d_surface = fig_3d_surface.add_subplot(111, projection='3d')
    ax_3d_surface.plot_surface(x.detach().cpu().numpy(), z.detach().cpu().numpy(), F_pred.detach().cpu().numpy(), cmap='viridis')
    ax_3d_surface.set_xlabel('x', labelpad=1)
    ax_3d_surface.set_ylabel('z', labelpad=1)
    ax_3d_surface.set_zlabel('F', labelpad=1)
    ax_3d_surface.tick_params(axis='x', pad=0)
    ax_3d_surface.tick_params(axis='y', pad=0)
    ax_3d_surface.tick_params(axis='z', pad=0)
    fig_3d_surface.subplots_adjust(left=0.00, right=0.7, bottom=0.2, top=1)      
    fig_3d_surface.subplots_adjust(left=0.00, right=0.7, bottom=0.2, top=1)
    plt.savefig(f'1D1_3d_F.png')  # Save each subplot
    plt.close()

    savemat('1D1_3d_F_data.mat', {
        'x': x_np,
        'z': z_np,
        'F': F_pred.detach().cpu().numpy()
    })

    # Plotting dF/dz as a 3D surface
    fig_dFdz = plt.figure(figsize=(2.2, 2))
    ax_dFdz = fig_dFdz.add_subplot(111, projection='3d')
    ax_dFdz.plot_surface(x.detach().cpu().numpy(), z.detach().cpu().numpy(), dF_dz.detach().cpu().numpy(), cmap='coolwarm')
    ax_3d_surface.set_xlabel('x', labelpad=1)
    ax_3d_surface.set_ylabel('z', labelpad=1)
    ax_3d_surface.set_zlabel('F', labelpad=1)
    ax_3d_surface.tick_params(axis='x', pad=0)
    ax_3d_surface.tick_params(axis='y', pad=0)
    ax_3d_surface.tick_params(axis='z', pad=0)        
    fig_dFdz.subplots_adjust(left=0.00, right=0.7, bottom=0.2, top=1)
    plt.savefig('1D1_3d_dFdz.png')
    plt.close()

    savemat('1D1_3d_dFdz_data.mat', {
        'x': x_np,
        'z': z_np,
        'dFdz': dF_dz.detach().cpu().numpy()
    })

    # 2D plots of F vs z for three values of x
    x_values = [0.25, 0.5, 0.75]
    plt.figure(figsize=(3.5, 2))
    for x_val in x_values:
        idx = (np.abs(x_np[:, 0] - x_val)).argmin()
        plt.plot(z.detach().cpu().numpy()[idx, :], F_pred.detach().cpu().numpy()[idx, :], label=f'x = {x_val}')
        # Save F(z) at fixed x
        savemat(f'1D1_F_x{x_val}_data.mat', {
            'z': z_np[idx, :],
            'F': F_pred.detach().cpu().numpy()[idx, :]
        })

    plt.xlabel('z')
    plt.ylabel('F')
    plt.legend()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)
    plt.savefig('1d1_plot_F_vs_z.png')
    plt.close()

    # 2D plots of dF/dz vs z for three values of x
    plt.figure(figsize=(3.5, 2))
    for x_val in x_values:
        idx = (np.abs(x_np[:, 0] - x_val)).argmin()
        plt.plot(z_np[idx, :], dF_dz.detach().cpu().numpy()[idx, :], label=f'x = {x_val}')
        # Save dFdz(z) at fixed x
        savemat(f'1D1_dFdz_x{x_val}_data.mat', {
            'z': z_np[idx, :],
            'dFdz': dF_dz.detach().cpu().numpy()[idx, :]
        })
    plt.xlabel('z')
    plt.ylabel('dF/dz')
    plt.legend()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)
    plt.savefig('1d1_plot_dFdz_vs_z.png')
    plt.close()

    # Generate 10,000 z values following a Gaussian distribution for a fixed x value (e.g., 0.5)
    fixed_x = 0.5
    z_values = np.random.normal(loc=0, scale=1, size=10000)

    # Clip the z values to be within the range -2 to 2
    z_values = np.clip(z_values, -2, 2)

    x_values = np.full_like(z_values, fixed_x)

    # Convert to tensors and ensure they require grad
    x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(-1).to('cuda').requires_grad_(True)
    z_tensor = torch.tensor(z_values, dtype=torch.float32).unsqueeze(-1).to('cuda').requires_grad_(True)

    # Concatenate x and z tensors
    xz_tensor = torch.cat([x_tensor, z_tensor], dim=-1)

    # Forward pass through the model
    F_tensor = model(xz_tensor)
    F_values = F_tensor.detach().cpu().numpy()

    # Calculate gradients for the generated points
    dF_dz_values = torch.autograd.grad(F_tensor.sum(), z_tensor, grad_outputs=torch.ones_like(F_tensor.sum()).to('cuda'), create_graph=True)[0].detach().cpu().numpy()

    # Plot histogram of dF/dz values
    plt.figure(figsize=(2.3,1.6))
    plt.hist(dF_dz_values, bins=20, density=True, alpha=0.6, color='r')
    plt.xlabel('dF/dz values')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('1D1_histogram_dFdz.png')
    plt.close()
    
    savemat('1D1_histogram_dFdz_data.mat', {
        'dFdz_values': dF_dz_values
    })
    
    # Subset x and z for a smaller grid
    x = x.reshape(-1)  # Flatten x to 1D
    z = z.reshape(-1)  # Flatten z to 1D
    
    x_subset = x[::10]
    z_subset = z[::10]
    
    # Dimensions
    N, M = len(x_subset), len(z_subset)
    
    # Create a smaller meshgrid using the subsets
    X, Z = torch.meshgrid(
        x_subset.squeeze().to('cuda'),
        z_subset.squeeze().to('cuda'),
        indexing='ij'
    )
    
    # Compute F values for all combinations
    inputs = torch.cat([X.unsqueeze(-1), Z.unsqueeze(-1)], dim=-1)
    inputs = inputs.view(-1, 2)
    
    # Enable gradients for z to compute derivatives
    inputs.requires_grad_(True)
    
    # Compute F values
    with torch.enable_grad():  # Enable gradients for computations
        F_values = model(inputs)
    
    # Compute the derivative of F with respect to z
    F_grad_z = torch.autograd.grad(F_values, inputs, grad_outputs=torch.ones_like(F_values), retain_graph=True)[0][:, 1]
    
    # Reshape F_grad_z back to the shape of the meshgrid
    F_grad_z = F_grad_z.view(len(x_subset), len(z_subset))
    
    # Compute the exponential term
    exp_term = torch.exp(-(Z ** 2) / 2)
    
    # Compute contributions
    contributions = F_grad_z * exp_term
    
    # Sum over z dimension
    inner_sum = contributions.sum(dim=1) / M
    
    # Compute cumulative sum over x dimension
    U = torch.cumsum(inner_sum, dim=0) / N
    U = U.detach().cpu().numpy()
    
    # Create meshgrid for plotting
    X = x_subset.cpu().numpy()

    # Plot U vs. x
    plt.figure(figsize=(3.5, 2))
    plt.plot(X, U, label='U', color='blue')
    plt.xlabel('x')
    plt.ylabel('U')
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)
    plt.savefig('1D1_U.png')  # Save the figure
    plt.close()

    savemat('1D1_U_data.mat', {
        'X': X,
        'U': U
    })


# Train the model with the updated loss function
# trained_model, loss_history_updated, x, z = train_pinn_updated(load_pretrained=True)
trained_model, loss_history, term1_history, term2_history, term3_history, x, z = train_pinn_updated()

# Visualize the results
visualize_model_updated(trained_model, loss_history, term1_history, term2_history, term3_history, x.to('cuda'), z.to('cuda'))

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in trained_model.state_dict():
    print(param_tensor, "\t", trained_model.state_dict()[param_tensor].size())

# If you want to save the model parameters to a file
torch.save(trained_model.state_dict(), "model_parameters1D1.pth")

# To load the model parameters from a file
trained_model.load_state_dict(torch.load("model_parameters1D1.pth"))
