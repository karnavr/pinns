import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## MAIN PINN CLASS
class PINN(nn.Module):
    """
    General PINN class for two inputs (x,t) and one output (u).
    """

    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation = nn.Tanh):

        super(PINN, self).__init__()

        # setup network parameters / activation 
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation

        # input (x, t) and output (u)
        self.n_input_nodes = 2
        self.n_output_nodes = 1

        # network layers
        layer_list = [nn.Linear(self.n_input_nodes, n_neurons_per_layer)]
        layer_list.append(activation())

        for _ in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            layer_list.append(activation())

        layer_list.append(nn.Linear(n_neurons_per_layer, self.n_output_nodes))

        # assign layers and initialize weights
        self.model = nn.Sequential(*layer_list)
        self._initialize_weights()

    # forward pass method         
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1) # Combine inputs into a single tensor
        output = self.model(inputs) # Pass through network layers
        return output
    
    # initialize weights (xavier normal)
    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight) # Xavier normal initialization
                nn.init.zeros_(layer.bias) # Zero initialization for biases
        
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


## PINN1D CLASS (for 1D problems with single input/output)
class PINN1D(nn.Module):
    """
    A class for 1D PINNs with a single input and output.
    """
    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation=nn.Tanh):
        super(PINN1D, self).__init__()

        # setup network parameters / activation 
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation

        # input (t) and output (x)
        self.n_input_nodes = 1
        self.n_output_nodes = 1

        # network layers
        layer_list = [nn.Linear(self.n_input_nodes, n_neurons_per_layer)]
        layer_list.append(activation())

        for _ in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            layer_list.append(activation())

        layer_list.append(nn.Linear(n_neurons_per_layer, self.n_output_nodes))

        # assign layers and initialize weights
        self.model = nn.Sequential(*layer_list)
        self._initialize_weights()

    # forward pass function         
    def forward(self, t):
        output = self.model(t)
        return output
    
    # initialize weights function (xavier normal)
    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # Xavier normal initialization
                nn.init.zeros_(layer.bias)  # Zero initialization for biases



## DAMPED OSCILLATOR NAIVE 
class DampedOscillatorNaive(nn.Module):
    """
    A class to solve the damped oscillator problem using a naive neural network.
    """
    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation=nn.Tanh):
        super(DampedOscillatorNaive, self).__init__()
        
        # setup network parameters / activation 
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation
        
        # use GPU (if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # input (t) and output (x)
        self.n_input_nodes = 1
        self.n_output_nodes = 1
        
        # network layers
        layer_list = [nn.Linear(self.n_input_nodes, n_neurons_per_layer)]
        layer_list.append(activation())
        
        for _ in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            layer_list.append(activation())
        
        layer_list.append(nn.Linear(n_neurons_per_layer, self.n_output_nodes))
        
        # assign layers
        self.model = nn.Sequential(*layer_list)
        self.model.to(self.device)
        
        # setup oscillator parameters 
        self.omega_0 = 2.0  # natural frequency
        self.zeta = 0.1     # damping ratio
        self.A = 1.0        # amplitude
        self.phi = 0.0      # phase
        
        self.figsize = (10, 6)
    
    def forward(self, t):
        return self.model(t)
    
    def analytic_solution(self, t):
        """
        Computes the analytic solution for a damped oscillator. 
        """
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
            
        # damped frequency
        omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
        
        # compute solution
        x = self.A * np.exp(-self.zeta * self.omega_0 * t) * np.cos(omega_d * t - self.phi)
        
        return x
    
    def generate_data(self, n_points=10, t_max=None, noise_level=0.0):
        """
        Generate data points from the analytic solution.
        Sample points from the first two oscillations by default.
        """
        # damped frequency
        omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
        
        # period and set t_max to cover 2 oscillations if not specified
        period = 2 * np.pi / omega_d
        if t_max is None:
            t_max = 2 * period
        
        # generate time points (concentrated in early oscillations)
        t_data = np.linspace(0, t_max, n_points)
        
        # compute analytic solution
        x_data = self.analytic_solution(t_data)
        
        # optionally add noise
        if noise_level > 0:
            x_data = x_data + np.random.normal(0, noise_level, size=x_data.shape)
        
        # convert to tensors and move to device
        t_data = torch.tensor(t_data, dtype=torch.float32).reshape(-1, 1).to(self.device)
        x_data = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # store data
        self.t_data = t_data
        self.x_data = x_data
        
        print(f"Generated {n_points} data points covering {t_max:.2f} time units")
        return t_data, x_data
    
    def loss_fn(self, t_data, x_data):
        """
        Compute MSE loss between predictions and data points
        """
        # forward pass to get predictions
        x_pred = self.forward(t_data)
        
        # compute mean squared error
        loss = torch.mean((x_pred - x_data)**2)
        
        return loss
    
    def train(self, n_epochs=2000, verbose=True, verbose_step=100, learning_rate=0.001):
        """
        Train the neural network using Adam optimizer — simply minimizes MSE loss between predictions and data points.
        """
        
        # initialize optimizer and losses list
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        
        if verbose:
            print(f"Starting training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            # zero gradients
            optimizer.zero_grad()
            
            # compute loss
            loss = self.loss_fn(self.t_data, self.x_data)
            
            # backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # store loss
            losses.append(loss.item())
            
            # print progress
            if verbose and (epoch % verbose_step == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.6e}")
        
        if verbose:
            print(f"Training complete. Final loss: {losses[-1]:.6e}")
        
        # store losses
        self.losses = losses
        
        return losses
    
    def plot_losses(self):
        """
        Plot the loss history during training.
        """

        plt.figure(figsize=self.figsize)
        plt.plot(self.losses)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.grid(False)
        
        # turn off top and right borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return
    
    def visualize(self, t_range=None, plot_data=True, n_points=500):
        """
        Visualize the neural network solution and analytic solution.
        """
        # set default time range if not provided
        if t_range is None:
            if hasattr(self, 't_data'):
                t_min = 0
                t_max = 3 * torch.max(self.t_data).item()
            else:
                # damped frequency and set range to cover 3 periods
                omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
                period = 2 * np.pi / omega_d
                t_min = 0
                t_max = 3 * period
        else:
            t_min, t_max = t_range
        
        # create smooth time points for plotting (increased for smoother curves)
        t_smooth = np.linspace(t_min, t_max, n_points)
        t_smooth_tensor = torch.tensor(t_smooth, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # get neural network predictions
        with torch.no_grad():
            x_pred = self.forward(t_smooth_tensor).cpu().numpy()
        
        # get analytic solution
        x_analytic = self.analytic_solution(t_smooth)
        
        # create plot
        plt.figure(figsize=self.figsize)
        
        # plot analytic solution (black)
        plt.plot(t_smooth, x_analytic, 'k-', label='Analytic Solution', linewidth=2)
        
        # plot neural network solution (green)
        plt.plot(t_smooth, x_pred, 'g--', label='Neural Network', linewidth=2)
        
        # plot data points if requested (orange)
        if plot_data and hasattr(self, 't_data') and hasattr(self, 'x_data'):
            t_data_np = self.t_data.cpu().numpy()
            x_data_np = self.x_data.cpu().numpy()
            plt.scatter(t_data_np, x_data_np, color='orange', marker='o', label='Data Points', s=50)
    
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        
        # turn off top and right borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return

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