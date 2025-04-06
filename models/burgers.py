import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .base import PINN

## BURGERS EQUATION
class BURGERS(nn.Module):
    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation = nn.Tanh, seed=None):

        super(BURGERS, self).__init__()

        # set random seed if provided (we want reproducibility sometimes)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # use GPU (if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # create model and move to GPU
        self.net = PINN(n_hidden_layers, n_neurons_per_layer, activation)
        self.net.to(self.device)

        # set viscosity 
        self.nu = 0.01 / np.pi

        # define domain limits
        self.x_lims = (-1, 1)
        self.t_lims = (0, 1)

        # setup domain points
        self.setup_domain()

        # setup figure size 
        self.figsize = (10, 6)

    def setup_domain(self, n_collocation=10000, n_initial=100, n_boundary=100):

        """
        Setup the domain points for training the PINN to solve the Burgers equation.
        """

        # unpack domain limits 
        x0 = self.x_lims[0]
        x1 = self.x_lims[1]
        
        t0 = self.t_lims[0]
        t1 = self.t_lims[1]
        
        # 1. Collocation points (random in the domain)
        x_collocation = torch.rand(n_collocation, 1) * (x1 - x0) + x0
        t_collocation = torch.rand(n_collocation, 1) * (t1 - t0) + t0
        
        # 2. Initial condition points (t=0) 
        x_initial = torch.linspace(x0, x1, n_initial).reshape(-1, 1)
        t_initial = torch.zeros_like(x_initial)
        u_initial = -torch.sin(np.pi * x_initial)  # u(x,0) = -sin(pi*x)
        
        # 3. Boundary condition points - uniform grid
        # Left boundary (x=-1)
        t_boundary_left = torch.linspace(t0, t1, n_boundary//2).reshape(-1, 1)
        x_boundary_left = torch.ones_like(t_boundary_left) * x0
        
        # Right boundary (x=1)
        t_boundary_right = torch.linspace(t0, t1, n_boundary//2).reshape(-1, 1)
        x_boundary_right = torch.ones_like(t_boundary_right) * x1
        
        # Combine boundary points
        x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
        t_boundary = torch.cat([t_boundary_left, t_boundary_right], dim=0)
        u_boundary = torch.zeros_like(x_boundary)  # u(-1,t) = u(1,t) = 0
        
        # Move all tensors to the device
        self.x_collocation = x_collocation.to(self.device)
        self.t_collocation = t_collocation.to(self.device)
        
        self.x_initial = x_initial.to(self.device)
        self.t_initial = t_initial.to(self.device)
        self.u_initial = u_initial.to(self.device)
        
        self.x_boundary = x_boundary.to(self.device)
        self.t_boundary = t_boundary.to(self.device)
        self.u_boundary = u_boundary.to(self.device)
        
        print(f"\nDomain setup complete with {n_collocation} collocation points, {n_initial} initial points, and {n_boundary} boundary points.")
        
    def create_evaluation_grid(self, nx=1000, nt=1000):

        """
        Create a fine grid to test the PINN.
        """

        # unpack domain limits 
        x0 = self.x_lims[0]
        x1 = self.x_lims[1]

        t0 = self.t_lims[0]
        t1 = self.t_lims[1]

        # define points in each dimension 
        x = torch.linspace(x0, x1, nx).to(self.device)
        t = torch.linspace(t0, t1, nt).to(self.device)
        
        # create meshgrid
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        # reshape for network input
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)

        ## flat versions are for network input and meshgrids are for plotting 
        
        return X, T, X_flat, T_flat

    # LOSS FUNCTIONS 
    def compute_pde_residual(self, x, t):

        """
        Compute the PDE residual for the Burgers equation. (physics loss)
        """

        # copies of x and t that require gradients
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        
        # forward pass = u(x,t)
        u = self.net(x, t)
        
        # calculate gradients
        # first-order derivatives
        u_grad = torch.autograd.grad(
            outputs=u, 
            inputs=[t, x], 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )
        u_t = u_grad[0]
        u_x = u_grad[1]
        
        # second-order derivative (u_xx)
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # compute PDE residual: u_t + u*u_x - nu*u_xx = 0
        residual = u_t + (u * u_x) - (self.nu * u_xx)
        
        return residual

    def compute_data_loss(self):

        """
        Compute the data loss for the Burgers equation. (ICs and BCs)
        """
        
        # forward pass for IC and BC points = u(x,t)
        u_pred_initial = self.net(self.x_initial, self.t_initial)
        u_pred_boundary = self.net(self.x_boundary, self.t_boundary)
        
        # compute and combine MSE losses
        initial_loss = torch.mean((u_pred_initial - self.u_initial)**2)
        boundary_loss = torch.mean((u_pred_boundary - self.u_boundary)**2)
        
        # total data loss
        data_loss = initial_loss + boundary_loss
        
        return data_loss

    def loss_fn(self):
        """
        Compute the total loss function: combining data and PDE losses.
        """
        # compute data loss
        data_loss = self.compute_data_loss()
        
        # compute PDE residual and MSE residual loss
        residual = self.compute_pde_residual(self.x_collocation, self.t_collocation)
        pde_loss = torch.mean(residual**2) # we want MSE 
        
        # total loss 
        total_loss = data_loss + pde_loss
        
        return total_loss
    
    ## TRAINING 
    def train(self, adam_epochs=1000, lbfgs_epochs=50000, verbose=True, verbose_step=100):
        
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
        
        # L-BFGS requires a closure that reevaluates the model and returns the loss
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            
            # Store current loss value directly in the main losses list
            losses.append(loss.item())
            
            # Print progress if verbose
            if verbose and len(losses) % verbose_step == 0:
                print(f"L-BFGS - Iteration {len(losses) - adam_epochs}, Loss: {loss.item():.6e}")
                
            return loss
        
        # initialize L-BFGS optimizer
        optimizer = torch.optim.LBFGS(self.net.parameters(),
                                    lr = 1.0, 
                                    max_iter=lbfgs_epochs,
                                    max_eval=lbfgs_epochs*2,
                                    tolerance_grad=1e-9,
                                    tolerance_change=1e-16,
                                    history_size=50,
                                    line_search_fn="strong_wolfe")
        
        # run the optimizer
        optimizer.step(closure)
        
        if verbose:
            print(f"L-BFGS complete, Final Loss: {losses[-1]:.6e}")
        
        return losses
    
    ## TESTING
    def test(self, nx=1000, nt=1000):
        """
        Evaluate the PINN on a finer (evaluation) grid by computing the loss terms = test loss
        """

        # print progress
        print(f"Evaluating PINN performance on a {nx}×{nt} grid...")
        
        # create a fine evaluation grid
        X, T, X_flat, T_flat = self.create_evaluation_grid(nx=nx, nt=nt)
        
        # create initial condition points
        x_initial = torch.linspace(self.x_lims[0], self.x_lims[1], nx).reshape(-1, 1).to(self.device)
        t_initial = torch.zeros_like(x_initial)
        u_initial = -torch.sin(np.pi * x_initial)
        
        # create boundary condition points
        t_boundary = torch.linspace(self.t_lims[0], self.t_lims[1], nt).reshape(-1, 1).to(self.device)
        x_boundary_left = torch.ones_like(t_boundary) * self.x_lims[0]
        x_boundary_right = torch.ones_like(t_boundary) * self.x_lims[1]
        x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
        t_boundary = torch.cat([t_boundary, t_boundary], dim=0)
        u_boundary = torch.zeros_like(x_boundary)
        
        # compute initial and boundary condition losses
        with torch.no_grad():
            # initial condition loss
            u_pred_initial = self.net(x_initial, t_initial)
            ic_loss = torch.mean((u_pred_initial - u_initial)**2)
            
            # boundary condition loss
            u_pred_boundary = self.net(x_boundary, t_boundary)
            bc_loss = torch.mean((u_pred_boundary - u_boundary)**2)
        
        # compute PDE residual loss 
        residual = self.compute_pde_residual(X_flat, T_flat)
        pde_loss = torch.mean(residual**2)
        
        # total loss
        total_loss = pde_loss + ic_loss + bc_loss
        
        # report results
        print(f"PDE Loss: {pde_loss.item():.6e}")
        print(f"Initial Condition Loss: {ic_loss.item():.6e}")
        print(f"Boundary Condition Loss: {bc_loss.item():.6e}")
        print(f"Total Loss: {total_loss.item():.6e}")
        
        return total_loss.item()

    
    ## PLOTTING 
    def compute_profile(self, t_val):
        """
        Compute the solution profile at specific time point(s).
        """
        # create evaluation grid
        X, T, X_flat, T_flat = self.create_evaluation_grid()
        
        # predict PINN solution on the grid
        with torch.no_grad():
            U_pred_flat = self.net(X_flat, T_flat)
            U_pred = U_pred_flat.reshape(X.shape)
        
        # convert to numpy for plotting
        X_np = X.cpu().numpy()
        T_np = T.cpu().numpy()
        U_pred_np = U_pred.cpu().numpy()
        
        # find closest time index
        t_idx = np.argmin(np.abs(T_np[0, :] - t_val))
        
        # extract the profile at the given time
        x_profile = X_np[:, t_idx]
        u_profile = U_pred_np[:, t_idx]
        
        return x_profile, u_profile
    
    def plot_profiles(self, t_points=[0.0, 0.2, 0.4, 0.6, 0.8]):
        """
        Plot solution profiles at multiple time points on the same plot.
        """
        # create a single figure
        plt.figure(figsize=self.figsize)
        
        # plot PINN predictions for each time sample
        for t_val in t_points:
            # compute the profile at this time
            x_profile, u_profile = self.compute_profile(t_val)
            
            # plot solution with different colors for each time step
            plt.plot(x_profile, u_profile, label=f't = {t_val}')
        
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.tight_layout()
        
        return
    
    def plot_losses(self, losses):
        """
        Plot the losses over epochs to visualize training progress.
        """
        plt.figure(figsize=self.figsize)
        plt.plot(losses)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        return
    
    def plot_spacetime(self, show_data = True):

        """
        Plot the solution as a color map over space and time.
        """

        # create evaluation grid
        X, T, X_flat, T_flat = self.create_evaluation_grid(nx=1000, nt=1000)
        
        # predict PINN solution on the grid
        with torch.no_grad():
            U_pred_flat = self.net(X_flat, T_flat)
            U_pred = U_pred_flat.reshape(X.shape)
        
        # convert to numpy for plotting
        X_np = X.cpu().numpy()
        T_np = T.cpu().numpy()
        U_pred_np = U_pred.cpu().numpy()
        
        # convert collocation points to numpy
        x_coll_np = self.x_collocation.cpu().numpy()
        t_coll_np = self.t_collocation.cpu().numpy()

        plt.figure(figsize=(10, 3))
        
        # plot the solution as a color map
        contour = plt.pcolormesh(T_np[0, :], X_np[:, 0], U_pred_np, cmap='plasma', shading='auto')
        plt.colorbar(contour, label='u(x,t)')
        
        # plot the collocation points
        if show_data:
            plt.scatter(t_coll_np, x_coll_np, marker='x', color='black', s=0.5, alpha=0.7, label='Collocation points')
        
            # plot initial and boundary points
            plt.scatter(self.t_initial.cpu().numpy(), self.x_initial.cpu().numpy(), 
                    marker='x', color='black', s=2, label='Initial condition')
            plt.scatter(self.t_boundary.cpu().numpy(), self.x_boundary.cpu().numpy(), 
                   marker='x', color='black', s=2, label='Boundary condition')
        
        plt.xlabel('Time (t)')
        plt.ylabel('Position (x)')
        plt.tight_layout()
        
        return

    def animate_solution(self, running_time=5, fps=60, save_path=None, dpi=200):
        """
        Create an animation of the solution profiles over time.
        """
        # calculate the number of frames needed for the requested duration and fps
        n_frames = int(running_time * fps)
        
        # generate equally spaced time points over the entire time domain
        t_start, t_end = self.t_lims
        time_points = np.linspace(t_start, t_end, n_frames)
        
        # set up the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=dpi)
        
        # compute the initial profile to get x values and set plot limits
        x_profile, u_profile = self.compute_profile(t_start)
        
        # create the line that will be updated with increased line width for better visibility
        line, = ax.plot(x_profile, u_profile, linewidth=2, color='black')
        
        # set up the plot 
        ax.set_xlim(self.x_lims)
        ax.set_ylim(-1.1, 1.1) 
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Text to display the current time with improved formatting
        time_text = ax.text(0.02, 0.95, f't = {t_start:.3f}', transform=ax.transAxes, 
                           fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # update function for animation
        def update(frame):
            t_val = time_points[frame]
            x_profile, u_profile = self.compute_profile(t_val)
            
            # update the line data
            line.set_data(x_profile, u_profile)
            
            # update the time text
            time_text.set_text(f't = {t_val:.3f}')
            
            return line, time_text
        
        plt.tight_layout()
        
        # create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=1000/fps, 
            blit=True, repeat=True
        )
        
        # save animation if path is provided
        if save_path:
            
            writer = animation.PillowWriter(
                fps=fps,
                metadata=dict(artist='PINN Simulation'),
                bitrate=1800
            )

            anim.save(save_path, writer=writer, dpi=dpi)
            print(f"Animation saved to {save_path}")
        
        return anim

