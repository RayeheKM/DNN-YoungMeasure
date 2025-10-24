#2D example
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

set_seed(13)

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
    def __init__(self, hidden_size=50):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(4, hidden_size)  # Update to 4 inputs

        # Define the ResNet blocks with skip connections
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(hidden_size, hidden_size),
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

class StochasticMeshgridDataset(IterableDataset):
    def __init__(self, x, y, z, w, batch_size):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.batch_size = batch_size

    def __iter__(self):
        while True:  # Infinite generator, DataLoader can control stopping
            indices_x1 = torch.randint(0, len(self.x), (self.batch_size,)).to('cuda')
            indices_y1 = torch.randint(0, len(self.y), (self.batch_size,)).to('cuda')
            indices_z1 = torch.randint(0, len(self.z), (self.batch_size,)).to('cuda')
            indices_w1 = torch.randint(0, len(self.w), (self.batch_size,)).to('cuda')
            
            indices_x2 = torch.randint(0, len(self.x), (self.batch_size,)).to('cuda')
            indices_y2 = torch.randint(0, len(self.y), (self.batch_size,)).to('cuda')
            indices_z2 = torch.randint(0, len(self.z), (self.batch_size,)).to('cuda')
            indices_w2 = torch.randint(0, len(self.w), (self.batch_size,)).to('cuda')
            
            x_batch1 = self.x[indices_x1].to('cuda')
            y_batch1 = self.y[indices_y1].to('cuda')
            z_batch1 = self.z[indices_z1].to('cuda')
            w_batch1 = self.w[indices_w1].to('cuda')
            
            x_batch2 = self.x[indices_x2].to('cuda')
            y_batch2 = self.y[indices_y2].to('cuda')
            z_batch2 = self.z[indices_z2].to('cuda')
            w_batch2 = self.w[indices_w2].to('cuda')
            
            yield (x_batch1, y_batch1, z_batch1, w_batch1), (x_batch2, y_batch2, z_batch2, w_batch2)

# Updated Loss Function to include the new term
def loss_function_minibatch(x1, y1, z1, w1, x2, y2, z2, w2, F):
    weightvalue1 = 1
    weightvalue2 = 25

    N1 = x1.shape[0]
    M1 = y1.shape[1]
    R1 = z1.shape[2]
    T1 = w1.shape[3]

    N2 = x2.shape[0]
    M2 = y2.shape[1]
    R2 = z2.shape[2]
    T2 = w2.shape[3]

    loss = torch.tensor(0.0, requires_grad=True)

    # Compute the derivative dF/dz
    dF_dz1 = torch.autograd.grad(F.sum(), z1, grad_outputs=torch.ones_like(F.sum().to('cuda')), create_graph=True)[0]
    dF_dw1 = torch.autograd.grad(F.sum(), w1, grad_outputs=torch.ones_like(F.sum().to('cuda')), create_graph=True)[0]
    dF_dz2 = torch.autograd.grad(F.sum(), z2, grad_outputs=torch.ones_like(F.sum().to('cuda')), create_graph=True)[0]
    dF_dw2 = torch.autograd.grad(F.sum(), w2, grad_outputs=torch.ones_like(F.sum().to('cuda')), create_graph=True)[0]
    d2F_dx1_dw1 = torch.autograd.grad(dF_dw1.sum(), x1, grad_outputs=torch.ones_like(dF_dw1.sum().to('cuda')), create_graph=True)[0]
    d2F_dy1_dz1 = torch.autograd.grad(dF_dz1.sum(), y1, grad_outputs=torch.ones_like(dF_dz1.sum().to('cuda')), create_graph=True)[0]
    d2F_dx2_dw2 = torch.autograd.grad(dF_dw2.sum(), x1, grad_outputs=torch.ones_like(dF_dw2.sum().to('cuda')), create_graph=True)[0]
    d2F_dy2_dz2 = torch.autograd.grad(dF_dz2.sum(), y1, grad_outputs=torch.ones_like(dF_dz2.sum().to('cuda')), create_graph=True)[0]

    # Compute term1Complete1 using vectorization
    term1 = ((dF_dz1**2 - 1)**2 + dF_dw1**2) * torch.exp(-((z1**2 + w1**2)) / 2)
    term1_2 = ((dF_dz2**2 - 1)**2 + dF_dw2**2) * torch.exp(-((z2**2 + w2**2)) / 2)
    term1Complete1_vec = (torch.sum(term1) / (R1 * T1)) + (torch.sum(term1_2) / (R2 * T2))

    # Compute term2Complete1 using vectorization
    term2 = dF_dz1 * torch.exp(-((z1**2 + w1**2)) / 2)
    term2_2 = dF_dz2 * torch.exp(-((z2**2 + w2**2)) / 2)
    term2Complete1_vec = 2 * torch.sum((torch.sum(term2, dim=(0, 2, 3)) / (R1 * T1 * N1)) * (torch.sum(term2_2, dim=(0, 2, 3)) / (R2 * T2 * N2)))

    # Compute term3Complete1 using vectorization

    x1_np = x1[:,0,0,0].detach().cpu().numpy()
    x2_np = x2[:,0,0,0].detach().cpu().numpy()

    term3 = dF_dw1 * torch.exp(-((z1**2 + w1**2)) / 2)
    term3_2 = dF_dw2 * torch.exp(-((z2**2 + w2**2)) / 2)
    term3Complete1_vec = 2 * torch.sum((torch.sum(term3, dim=(1, 2, 3)) / (R1 * T1 * M1)- torch.full_like(torch.sum(term3, dim=(1, 2, 3)), 0.5))*(torch.sum(term3_2, dim=(1, 2, 3)) / (R2 * T2 * M2)- torch.full_like(torch.sum(term3_2, dim=(1, 2, 3)), 0.5)))

    term1Complete1_vec /= (N1 * M1)
    term2Complete1_vec = weightvalue1 * term2Complete1_vec / M1
    term3Complete1_vec = weightvalue2 * term3Complete1_vec / N1

    #Curl terms
    termC = d2F_dx1_dw1 * torch.exp(-((z1**2 + w1**2)) / 2)
    termCC = d2F_dy1_dz1 * torch.exp(-((z1**2 + w1**2)) / 2)
    termCCC = (termC-termCC)
    termC2 = d2F_dx2_dw2 * torch.exp(-((z2**2 + w2**2)) / 2)
    termCC2 = d2F_dy2_dz2 * torch.exp(-((z2**2 + w2**2)) / 2)
    termCCC2 = (termC2-termCC2)
    termC_vec = (torch.sum(termCCC, dim=(2, 3)) / (R1 * T1))*(torch.sum(termCCC2, dim=(2, 3)) / (R2 * T2))
    termCComplete_vec = torch.sum(termC_vec, dim=(0, 1)) / ( M1 * N1 )

    # Convert the results to numpy arrays for further processing
    term1Complete1_vec_np = term1Complete1_vec.detach().cpu().numpy()
    term2Complete1_vec_np = term2Complete1_vec.detach().cpu().numpy()
    term3Complete1_vec_np = term3Complete1_vec.detach().cpu().numpy()
    termCComplete_vec_np = termCComplete_vec.detach().cpu().numpy()

    loss = term1Complete1_vec + term2Complete1_vec + term3Complete1_vec + termCComplete_vec

    return loss, term1Complete1_vec.item(), term2Complete1_vec.item(), term3Complete1_vec.item(), termCComplete_vec.item()

# Updated Training Function
def train_pinn_updated_minibatch(load_pretrained=False, pretrained_path="model_parameters.pth"):
    model = PINN().to('cuda')

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    if load_pretrained:
        model.load_state_dict(torch.load(pretrained_path, weights_only=True))
        print("Loaded pretrained model parameters.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    epochs = 1000
    loss_history = []
    term1_history = []
    term2_history = []
    term3_history = []
    termc_history = []

    x = torch.linspace(0, 1, 201).view(-1, 1).to('cuda')
    y = torch.linspace(0, 1, 201).view(-1, 1).to('cuda')
    M = 2
    z = torch.linspace(-M, M, 201).view(-1, 1).to('cuda')
    w = torch.linspace(-M, M, 201).view(-1, 1).to('cuda')
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    w.requires_grad = True

    initial_batch_size = 5
    max_batches = 100

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_term1 = 0.0
        epoch_term2 = 0.0
        epoch_term3 = 0.0
        epoch_termc = 0.0

        # Dynamically adjust batch size
        batch_size = initial_batch_size + epoch // 100  # Increase batch size every 10 epochs
        dataset = StochasticMeshgridDataset(x, y, z, w, batch_size)

        for i, ((x_batch1, y_batch1, z_batch1, w_batch1), (x_batch2, y_batch2, z_batch2, w_batch2)) in zip(range(max_batches), dataset):
            optimizer.zero_grad()

            X1, Y1, Z1, W1 = torch.meshgrid(x_batch1.squeeze(), y_batch1.squeeze(), z_batch1.squeeze(), w_batch1.squeeze(), indexing='ij')
            X2, Y2, Z2, W2 = torch.meshgrid(x_batch2.squeeze(), y_batch2.squeeze(), z_batch2.squeeze(), w_batch2.squeeze(), indexing='ij')

            inputs1 = torch.cat([X1.unsqueeze(-1), Y1.unsqueeze(-1), Z1.unsqueeze(-1), W1.unsqueeze(-1)], dim=-1)
            inputs2 = torch.cat([X2.unsqueeze(-1), Y2.unsqueeze(-1), Z2.unsqueeze(-1), W2.unsqueeze(-1)], dim=-1)

            inputs = torch.cat([inputs1, inputs2], dim=0).to('cuda')

            F = model(inputs.view(-1, 4))
            loss, term1, term2, term3, termc = loss_function_minibatch(X1, Y1, Z1, W1, X2, Y2, Z2, W2, F)
            epoch_loss += loss.item()
            epoch_term1 += term1
            epoch_term2 += term2
            epoch_term3 += term3
            epoch_termc += termc

            loss.backward()
            optimizer.step()

        # Average the losses over all batches
        epoch_loss = epoch_loss / max_batches
        epoch_term1 = epoch_term1 / max_batches
        epoch_term2 = epoch_term2 / max_batches
        epoch_term3 = epoch_term3 / max_batches
        epoch_termc = epoch_termc / max_batches

        # Step the scheduler
        scheduler.step(epoch_loss)

        # Append to history
        loss_history.append(epoch_loss)
        term1_history.append(epoch_term1)
        term2_history.append(epoch_term2)
        term3_history.append(epoch_term3)
        termc_history.append(epoch_termc)

        print(f'Epoch {epoch}, Loss: {epoch_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

    return model, loss_history, term1_history, term2_history, term3_history, termc_history, x, y, z, w
    
# Updated Visualization function with adjusted vertical axis labels and further shifted plots

def visualize_model_updated_minibatch(model, loss_history, term1_history, term2_history, term3_history, termc_history, x, y, z, w):

    # Plotting the loss over epochs
    plt.figure(figsize=(3.5, 2))
    energy_history = np.array(term1_history) + np.array(termc_history)
    bc_history = np.array(term2_history) + np.array(term3_history)
    plt.plot(loss_history, label=r'$\text{Total}$', color='blue')
    plt.plot(energy_history, label=r'$|\text{Energy}|^2$', color='red')
    plt.plot(bc_history, label=r'$|\text{BC}|^2$', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    # plt.grid()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)
    plt.savefig('2DNew_loss.png')  # Save the figure
    plt.close()  # Close the figure

    savemat('2DNew_loss_data.mat', {
    'loss_history': loss_history,
    'term1_history': term1_history,
    'term2_history': term2_history,
    'term3_history': term3_history,
    'termc_history': termc_history,
    'bc_history': bc_history,
    'energy_history': energy_history,
    })

    model.eval()

    # Compute F
    Z, W = torch.meshgrid(z.squeeze().to('cuda'), w.squeeze().to('cuda'), indexing='ij')

    # Enable gradients for z and w to compute derivatives
    # Z.requires_grad_(True)
    # W.requires_grad_(True)
    # Enable gradients for Z and W to compute derivatives
    Z = Z.clone().detach().requires_grad_(True)
    W = W.clone().detach().requires_grad_(True)

    fixed_points = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    
    for i, (x_fixed, y_fixed) in enumerate(fixed_points):
        # Compute F for fixed x and y
        x_fixed_tensor = torch.full_like(Z, x_fixed).to('cuda')
        y_fixed_tensor = torch.full_like(W, y_fixed).to('cuda')
    
        F = model(torch.cat([x_fixed_tensor.unsqueeze(-1), y_fixed_tensor.unsqueeze(-1), Z.unsqueeze(-1), W.unsqueeze(-1)], dim=-1).view(-1, 4))
        F = F.view(Z.shape)
    
        dF_dz, = torch.autograd.grad(F, Z, grad_outputs=torch.ones_like(F).to('cuda'), create_graph=True)
        dF_dw, = torch.autograd.grad(F, W, grad_outputs=torch.ones_like(F).to('cuda'), create_graph=True)
    
        # Plot F
        fig = plt.figure(figsize=(2.2, 2))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Z.detach().cpu().numpy(), W.detach().cpu().numpy(), F.detach().cpu().numpy(), cmap='viridis')
        ax.set_xlabel('z', labelpad=1)
        ax.set_ylabel('w', labelpad=1)
        ax.set_zlabel('F', labelpad=1)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=0)
        ax.tick_params(axis='z', pad=0)        
        # ax.set_title(f'F at x={x_fixed}, y={y_fixed}')
        # plt.tight_layout()
        fig.subplots_adjust(left=0.00, right=0.7, bottom=0.2, top=1)
        plt.savefig(f'2DNew_3d_F_x{x_fixed}_y{y_fixed}.png')  # Save each subplot
        plt.close(fig)  # Close the figure
    
        # Plot dF/dz
        fig = plt.figure(figsize=(2.2, 2))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Z.detach().cpu().numpy(), W.detach().cpu().numpy(), dF_dz.detach().cpu().numpy(), cmap='plasma')
        ax.set_xlabel('z', labelpad=1)
        ax.set_ylabel('w', labelpad=1)
        ax.set_zlabel('∂F/∂z', labelpad=1)
        ax.set_zlim(-1.5, 1.5)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=0)
        ax.tick_params(axis='z', pad=0)
        # ax.set_title(f'∂F/∂z at x={x_fixed}, y={y_fixed}')
        # plt.tight_layout()
        fig.subplots_adjust(left=0.00, right=0.7, bottom=0.2, top=1)
        plt.savefig(f'2DNew_3d_dFdz_x{x_fixed}_y{y_fixed}.png')  # Save each subplot
        plt.close(fig)  # Close the figure
    
        # Plot dF/dw
        fig = plt.figure(figsize=(2.2, 2))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Z.detach().cpu().numpy(), W.detach().cpu().numpy(), dF_dw.detach().cpu().numpy(), cmap='magma')
        ax.set_xlabel('z', labelpad=0)
        ax.set_ylabel('w', labelpad=0)
        ax.set_zlabel('∂F/∂w', labelpad=0)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=0)
        ax.tick_params(axis='z', pad=0)
        ax.set_zlim(-1.5, 1.5)
        # ax.set_title(f'∂F/∂w at x={x_fixed}, y={y_fixed}')
        # plt.tight_layout()
        fig.subplots_adjust(left=0.00, right=0.7, bottom=0.2, top=1)
        plt.savefig(f'2DNew_3d_dFdw_x{x_fixed}_y{y_fixed}.png')  # Save each subplot
        plt.close(fig)  # Close the figure

        # Save F surface data
        savemat(f'2DNew_3d_F_x{x_fixed}_y{y_fixed}_data.mat', {
            'Z': Z.detach().cpu().numpy(),
            'W': W.detach().cpu().numpy(),
            'F': F.detach().cpu().numpy()
        })

        # Save dF/dz
        savemat(f'2DNew_3d_dFdz_x{x_fixed}_y{y_fixed}_data.mat', {
            'Z': Z.detach().cpu().numpy(),
            'W': W.detach().cpu().numpy(),
            'dFdz': dF_dz.detach().cpu().numpy()
        })

        # Save dF/dw
        savemat(f'2DNew_3d_dFdw_x{x_fixed}_y{y_fixed}_data.mat', {
            'Z': Z.detach().cpu().numpy(),
            'W': W.detach().cpu().numpy(),
            'dFdw': dF_dw.detach().cpu().numpy()
        })


    fixed_points = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    
    for col, (x_value, y_value) in enumerate(fixed_points):
        # Generate 10k values for z and w from a normal distribution
        z_values = np.random.normal(0, 1, 10000)
        w_values = np.random.normal(0, 1, 10000)
    
        # Convert to tensors
        z_tensor = torch.tensor(z_values, dtype=torch.float32).view(-1, 1).to('cuda')
        w_tensor = torch.tensor(w_values, dtype=torch.float32).view(-1, 1).to('cuda')
        x_tensor = torch.full_like(z_tensor, x_value).to('cuda')
        y_tensor = torch.full_like(z_tensor, y_value).to('cuda')
    
        # Enable gradient tracking for z and w
        z_tensor.requires_grad_(True)
        w_tensor.requires_grad_(True)
    
        # Combine inputs and compute F
        inputs = torch.cat([x_tensor, y_tensor, z_tensor, w_tensor], dim=-1)
        F_pred = model(inputs)
    
        # Compute derivatives
        dF_dz = torch.autograd.grad(F_pred.sum(), z_tensor, create_graph=True)[0]
        dF_dw = torch.autograd.grad(F_pred.sum(), w_tensor, create_graph=True)[0]
    
        # Plot dF/dz histograms (second row)
        fig = plt.figure(figsize=(2.3, 1.6))
        plt.hist(dF_dz.detach().cpu().numpy(), bins=20, alpha=0.5, color='red', density=True)
        plt.xlim(-1.5, 1.5)
        plt.xlabel('dF/dz')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'2DNew_histogram_dFdz_x{x_value}_y{y_value}.png')
        plt.close(fig)
    
        # Plot dF/dw histograms (third row)
        fig = plt.figure(figsize=(2.3, 1.6))
        plt.hist(dF_dw.detach().cpu().numpy(), bins=20, alpha=0.5, color='green', density=True)
        plt.xlim(-1.5, 1.5)
        plt.xlabel('dF/dw')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'2DNew_histogram_dFdw_x{x_value}_y{y_value}.png')
        plt.close(fig)

        savemat(f'2DNew_histogram_dFdz_x{x_value}_y{y_value}_data.mat', {
            'dFdz': dF_dz.detach().cpu().numpy()
        })

        savemat(f'2DNew_histogram_dFdw_x{x_value}_y{y_value}_data.mat', {
            'dFdw': dF_dw.detach().cpu().numpy()
        })
    
    # Create 4D grids for x, y, z, w
    x_subset = x[::5]
    y_subset = y[::5]
    z_subset = z[::5]
    w_subset = w[::5]

    # Compute U(x_i, y_m) using the vectorized implementation
    N, M, R, T = len(x_subset), len(y_subset), len(z_subset), len(w_subset)

    # Create a smaller meshgrid using the subsets
    X, Y, Z, W = torch.meshgrid(
        x_subset.squeeze().to('cuda'),
        y_subset.squeeze().to('cuda'),
        z_subset.squeeze().to('cuda'),
        w_subset.squeeze().to('cuda'),
        indexing='ij'
    )

    # Compute F values for all combinations
    inputs = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1), W.unsqueeze(-1)], dim=-1)
    inputs = inputs.view(-1, 4)
    
    # Enable gradients for w to compute derivatives
    inputs.requires_grad_(True)
    
    # Compute F values
    with torch.enable_grad():  # Enable gradients for computations
        F_values = model(inputs)
    
    # Compute the derivative of F with respect to w
    F_grad_w = torch.autograd.grad(F_values, inputs, grad_outputs=torch.ones_like(F_values), retain_graph=True)[0][:, 3]
    
    # Reshape F_grad_w back to the shape of the meshgrid
    F_grad_w = F_grad_w.view(len(x_subset), len(y_subset), len(z_subset), len(w_subset))
    
    # Compute the derivative of F with respect to z
    F_grad_z = torch.autograd.grad(F_values, inputs, grad_outputs=torch.ones_like(F_values), retain_graph=True)[0][:, 2]
    
    # Reshape F_grad_z back to the shape of the meshgrid
    F_grad_z = F_grad_z.view(len(x_subset), len(y_subset), len(z_subset), len(w_subset))


    # Compute the exponential term
    exp_term = torch.exp(-((Z**2 + W**2)) / 2)
    
    # Compute contributions
    contributions = F_grad_w * exp_term
    
    contributions2 = F_grad_z * exp_term
    
    # Sum over z and w dimensions
    inner_sum = contributions.sum(dim=(2, 3)) / (R * T)
    inner_sum2 = contributions2.sum(dim=(2, 3)) / (R * T)
    
    # Compute cumulative sum over y dimension
    U = torch.cumsum(inner_sum, dim=1) / M  # Use cumulative sum along y-dimension
    U = U.detach().cpu().numpy()
    
    U2 = torch.cumsum(inner_sum2, dim=1) / N  # Use cumulative sum along y-dimension
    U2 = U2.detach().cpu().numpy()

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_subset.detach().cpu().numpy(), y_subset.detach().cpu().numpy(), indexing='ij')

    # Plot U vs. x and y
    fig = plt.figure(figsize=(3, 2.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ax.elev, azim=ax.azim + 30)
    ax.plot_surface(X, Y, U, cmap='viridis')
    ax.set_xlabel('x', labelpad=1)
    ax.set_ylabel('y', labelpad=1)
    ax.set_zlabel('U', labelpad=1)
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.tick_params(axis='z', pad=0)
    fig.subplots_adjust(left=0.01, right=0.8, bottom=0.1, top=1)
    plt.savefig('2DNew_U.png')  # Save the figure
    plt.close()

    savemat('2DNew_U_data.mat', {
        'X': X,
        'Y': Y,
        'U': U
    })

    # # Plot U vs. x and y
    # fig = plt.figure(figsize=(3, 2.6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, U2, cmap='viridis')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('U')
    # fig.subplots_adjust(left=0.00, right=0.85, bottom=0.15, top=1)
    # plt.savefig('2DNew_U2.png')  # Save the figure
    # plt.close()
    
    
# Train the model with the updated loss function
trained_model, loss_history, term1_history, term2_history, term3_history, termc_history, x, y, z, w = train_pinn_updated_minibatch()

# Visualize the results
visualize_model_updated_minibatch(trained_model, loss_history, term1_history, term2_history, term3_history, termc_history, x.to('cuda'), y.to('cuda'), z.to('cuda'), w.to('cuda'))

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in trained_model.state_dict():
    print(param_tensor, "\t", trained_model.state_dict()[param_tensor].size())

# If you want to save the model parameters to a file
torch.save(trained_model.state_dict(), "model_parameters2D23.pth")

# To load the model parameters from a file
trained_model.load_state_dict(torch.load("model_parameters2D23.pth"))
