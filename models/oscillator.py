import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .base import PINN1D


## DAMPED OSCILLATOR PINN 
class DampedOscillatorPINN(nn.Module):
    """
    A class to solve the damped oscillator problem using a PINN.
    """
    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation=nn.Tanh):
        super(DampedOscillatorPINN, self).__init__()
        
        # use GPU (if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # create model and move to device
        self.net = PINN1D(n_hidden_layers, n_neurons_per_layer, activation)
        self.net.to(self.device)
        
        # setup oscillator parameters
        self.omega_0 = 2.0  # natural frequency
        self.zeta = 0.1     # damping ratio
        self.A = 1.0        # amplitude
        self.phi = 0.0      # phase
        
        omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
        period = 2 * np.pi / omega_d
        t_data_max = 2 * period  
        t_max = 3 * t_data_max 
        
        # time domain limits
        self.t_lims = (0, t_max)
        
        # setup domain points
        self.setup_domain()

        # figure size
        self.figsize = (10, 6)
    
    def setup_domain(self, n_collocation=10000, n_data=10):
        """
        Setup the domain points for training the PINN
        """
        # unpack domain limits
        t0 = self.t_lims[0]
        t1 = self.t_lims[1]
        
        # 1. Collocation points (random in the domain)
        t_collocation = torch.rand(n_collocation, 1) * (t1 - t0) + t0
        
        # 2. Generate data points from the analytic solution
        # damped frequency
        omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
        
        # Compute period and set t_max to cover 2 oscillations
        period = 2 * np.pi / omega_d
        t_max = 2 * period
        
        # Generate time points (concentrated in early oscillations)
        t_data_np = np.linspace(0, t_max, n_data)
        
        # Compute analytic solution
        x_data_np = self.analytic_solution(t_data_np)
        
        # Convert to tensors and move to device
        t_data = torch.tensor(t_data_np, dtype=torch.float32).reshape(-1, 1).to(self.device)
        x_data = torch.tensor(x_data_np, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # Move collocation points to device
        self.t_collocation = t_collocation.to(self.device)
        
        # Store data points
        self.t_data = t_data
        self.x_data = x_data
        
        print(f"Domain setup: {n_collocation} collocation points, {n_data} data points")
    
    def analytic_solution(self, t):
        """
        Compute the analytic solution for a damped oscillator
        """
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
            
        # damped frequency
        omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
        
        # solution
        x = self.A * np.exp(-self.zeta * self.omega_0 * t) * np.cos(omega_d * t - self.phi)
        
        return x
    
    def compute_pde_residual(self, t):
        """
        Compute the PDE residual for the damped oscillator equation
        """
        # create copy of t that requires gradients
        t = t.clone().detach().requires_grad_(True)
        
        # forward pass to get x(t)
        x = self.net(t)
        
        # first derivative (x')
        x_t = torch.autograd.grad(
            outputs=x,
            inputs=t,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # second derivative (x'')
        x_tt = torch.autograd.grad(
            outputs=x_t,
            inputs=t,
            grad_outputs=torch.ones_like(x_t),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # damped oscillator equation: x'' + 2*zeta*omega_0*x' + omega_0^2*x = 0
        residual = x_tt + 2 * self.zeta * self.omega_0 * x_t + (self.omega_0**2) * x
        
        return residual
    
    def loss_fn(self):
        """
        Compute the total loss combining data, PDE, and initial condition losses.
        """
        # compute data loss - MSE between predictions and data points
        x_pred = self.net(self.t_data)
        data_loss = torch.mean((x_pred - self.x_data)**2)
        
        # compute PDE residual and MSE residual loss
        residual = self.compute_pde_residual(self.t_collocation)
        pde_loss = torch.mean(residual**2)
        
        # Compute initial condition loss
        # 1. Position at t=0
        t_init = torch.zeros(1, 1, requires_grad=True).to(self.device)
        x_init = self.net(t_init)
        x_init_target = torch.tensor([[self.A]]).to(self.device) 
        ic_pos_loss = ((x_init - x_init_target)**2).mean()
        
        # 2. Velocity at t=0
        x_t_init = torch.autograd.grad(
            outputs=x_init,
            inputs=t_init,
            grad_outputs=torch.ones_like(x_init),
            create_graph=True,
            retain_graph=True
        )[0]
        x_t_init_target = torch.tensor([[0.0]]).to(self.device)
        ic_vel_loss = ((x_t_init - x_t_init_target)**2).mean()
        
        # combined initial condition loss
        ic_loss = ic_pos_loss + ic_vel_loss
        
        # loss weights
        lambda_data = 0.5
        lambda_physics = 15.0
        lambda_ic = 5.0
        
        # combined loss
        total_loss = lambda_data * data_loss + lambda_physics * pde_loss + lambda_ic * ic_loss
        
        return total_loss
    
    def train(self, adam_epochs=1000, lbfgs_epochs=50000, verbose=True, verbose_step=100):
        """
        Train the PINN using Adam followed by L-BFGS optimization
        """
        # initialize losses list
        losses = []
        
        # Phase 1: Adam optimization
        if verbose:
            print("Starting Adam optimization...")
        
        optimizer = torch.optim.Adam(self.net.parameters())
        
        for epoch in range(adam_epochs):
            # zero gradients
            optimizer.zero_grad()
            
            # compute loss
            loss = self.loss_fn()
            
            # backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # store loss
            losses.append(loss.item())
            
            # print progress
            if verbose and (epoch % verbose_step == 0 or epoch == adam_epochs - 1):
                print(f"Adam - Epoch {epoch}/{adam_epochs}, Loss: {loss.item():.6e}")
        
        # Phase 2: L-BFGS optimization
        if verbose:
            print("\nStarting L-BFGS optimization...")
        
        # L-BFGS requires closure that reevaluates the model and returns loss
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            
            # store loss
            losses.append(loss.item())

            # print progress if verbose
            if verbose and len(losses) % verbose_step == 0:
                print(f"L-BFGS - Iteration {len(losses) - adam_epochs}, Loss: {loss.item():.6e}")
                
            return loss
        
        # initialize L-BFGS optimizer
        optimizer = torch.optim.LBFGS(self.net.parameters(),
                                    lr=1.0,
                                    max_iter=lbfgs_epochs,
                                    max_eval=lbfgs_epochs*2,
                                    tolerance_grad=1e-9,
                                    tolerance_change=1e-16,
                                    history_size=50,
                                    line_search_fn="strong_wolfe")
        
        # run the optimizer
        optimizer.step(closure)
        
        if verbose:
            print(f"Training complete. Final loss: {losses[-1]:.6e}")
        
        # store losses for later reference
        self.losses = losses
        
        return losses
    
    def visualize(self, t_range=None, n_points=500):
        """
        Visualize the PINN solution, analytic solution, and data points
        """
        # set default time range if not provided
        if t_range is None:
            # use the same logic as in naive approach
            if hasattr(self, 't_data'):
                t_min = 0
                t_max = 3 * torch.max(self.t_data).item()
            else:
                omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
                period = 2 * np.pi / omega_d
                t_min = 0
                t_max = 3 * 2 * period  
        else:
            t_min, t_max = t_range
        
        # Create smooth time points for plotting
        t_smooth = np.linspace(t_min, t_max, n_points)
        t_smooth_tensor = torch.tensor(t_smooth, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # get PINN predictions
        with torch.no_grad():
            x_pred = self.net(t_smooth_tensor).cpu().numpy()
        
        # get analytic solution
        x_analytic = self.analytic_solution(t_smooth)
        
        # create plot
        plt.figure(figsize=self.figsize)
        
        # plot analytic solution (black)
        plt.plot(t_smooth, x_analytic, 'k-', label='Analytic Solution', linewidth=2)
        
        # plot PINN solution (green)
        plt.plot(t_smooth, x_pred, 'g--', label='PINN', linewidth=2)
        
        # Plot data points (orange)
        t_data_np = self.t_data.cpu().numpy()
        x_data_np = self.x_data.cpu().numpy()
        plt.scatter(t_data_np, x_data_np, color='orange', marker='o', label='Data Points', s=50)
        
        # determine the minimum y value for plotting collocation points
        y_min = min(np.min(x_analytic), np.min(x_pred))
        if hasattr(self, 'x_data'):
            y_min = min(y_min, np.min(x_data_np))
        y_min = y_min * 1.1  
        
        # 100 uniformly distributed collocation points for visualization
        t_coll_sample = np.linspace(t_min, t_max, 100).reshape(-1, 1)

        # plot collocation points at minimum y-value
        plt.scatter(t_coll_sample, np.ones_like(t_coll_sample) * y_min, 
                   color='#0C8CE9', marker='x', label='Collocation Points', s=30, alpha=0.7)
        
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        
        # turn off top and right borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return