import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .base import PINN

import json
import os

## KDV EQUATION
class KDV(nn.Module):
    def __init__(self, num_solitons=1, n_hidden_layers=3, n_neurons_per_layer=32, activation=nn.Tanh, seed=None):
    
        super(KDV, self).__init__()
        
        # Store number of solitons
        self.num_solitons = num_solitons
        
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
        
        # Set KdV equation coefficients
        # Standard form: u_t + 6*u*u_x + u_xxx = 0
        self.alpha = 6.0  # coefficient for nonlinear term
        self.beta = 1.0   # coefficient for dispersive term
        
        # define domain limits (based on Nadia's thesis)
        if num_solitons == 1:
            self.x_lims = (-50, 50)
            self.t_lims = (0, 14)

        if num_solitons == 3:
            self.x_lims = (-50, 30)
            self.t_lims = (0, 18)
        
        # setup training domain points
        self.setup_training_domain()
        self.test_domain_created = False
        
        # setup figure size 
        self.figsize = (10, 6)
        
        # Track optimizer information
        self.adam_epochs = 0

    def soliton_initial(self, x, k, phi=0):
        """
        Compute a single soliton profile with the sech^2 solution to the KdV equation.
        
        Parameters:
        -----------
        x: torch.Tensor
            Spatial coordinates
        k: float
            Wavenumber (related to amplitude and speed)
        phi: float
            Phase parameter
            
        Returns:
        --------
        torch.Tensor
            The soliton profile at the specified locations
        """
        # u = 2k² sech(kx + φ)²
        return 2 * (k**2) * torch.pow(1.0 / torch.cosh(k * x + phi), 2)

    def setup_training_domain(self, n_collocation=30000, n_initial=30000, n_boundary=30000):
        """
        Setup the domain points for training the PINN to solve the KdV equation.
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
        
        # Generate initial condition based on number of solitons
        if self.num_solitons == 1:
            # One-soliton initial condition - parameters from Nadia's thesis
            k = 0.9  # wavenumber
            phi = 12  # phase parameter

            self.k_vector = np.array([k])
            self.phi_vector = np.array([phi])

            u_initial = self.soliton_initial(x_initial, k, phi)

        elif self.num_solitons == 2:
            # Two-soliton initial condition - parameters from Nadia's thesis
            # For linear combination of solitons
            k1 = np.sqrt(4.0/4)  # First wavenumber (c₁ = 3.23)
            k2 = np.sqrt(0.9/4)  # Second wavenumber (c₂ = 0.5)
            phi1 = 16  # First phase
            phi2 = -5   # Second phase

            self.k_vector = np.array([k1, k2])
            self.phi_vector = np.array([phi1, phi2])
            
            # Linear superposition of two solitons
            u_initial = self.soliton_initial(x_initial, k1, phi1) + self.soliton_initial(x_initial, k2, phi2)

        elif self.num_solitons == 3:

            k1 = np.sqrt(1)  # First wavenumber (c₁ = 3.23)
            k2 = np.sqrt(0.7)  # Second wavenumber (c₂ = 0.5)
            k3 = np.sqrt(0.2)

            phi1 = 45
            phi2 = 25
            phi3 = 5

            self.k_vector = np.array([k1, k2, k3])
            self.phi_vector = np.array([phi1, phi2, phi3])

            u_initial = self.soliton_initial(x_initial, k1, phi1) + self.soliton_initial(x_initial, k2, phi2) + self.soliton_initial(x_initial, k3, phi3)
            
        else:
            raise ValueError(f"Support for {self.num_solitons} solitons not implemented yet")
    
        # 3. Boundary condition points - uniform grid
        # Left boundary (x=x0)
        t_boundary_left = torch.linspace(t0, t1, n_boundary//2).reshape(-1, 1)
        x_boundary_left = torch.ones_like(t_boundary_left) * x0
        
        # Right boundary (x=x1)
        t_boundary_right = torch.linspace(t0, t1, n_boundary//2).reshape(-1, 1)
        x_boundary_right = torch.ones_like(t_boundary_right) * x1
        
        # Combine boundary points
        x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
        t_boundary = torch.cat([t_boundary_left, t_boundary_right], dim=0)
        
        # For KdV solitons, u should approach zero at the boundaries
        u_boundary = torch.zeros_like(x_boundary)
        
        # Move all tensors to the device
        self.x_collocation = x_collocation.to(self.device)
        self.t_collocation = t_collocation.to(self.device)
        
        self.x_initial = x_initial.to(self.device)
        self.t_initial = t_initial.to(self.device)
        self.u_initial = u_initial.to(self.device)
        
        self.x_boundary = x_boundary.to(self.device)
        self.t_boundary = t_boundary.to(self.device)
        self.u_boundary = u_boundary.to(self.device)
        
        print(f"""
                Training domain setup complete: 
                - {n_collocation} collocation points
                - {n_initial} initial points
                - {n_boundary} boundary points""")
        print(f"Using {self.num_solitons}-soliton initial condition.")
        
    def setup_testing_domain(self, nx=1000, nt=1000):
        """
        Create a regular grid for testing and visualization.
        
        This function sets up a uniform meshgrid covering the entire domain
        for evaluation and visualization of the PINN solution. The grid
        is stored as instance variables for later use.
        """
        # unpack domain limits 
        x0 = self.x_lims[0]
        x1 = self.x_lims[1]
        
        t0 = self.t_lims[0]
        t1 = self.t_lims[1]
        
        # define points in each dimension 
        x = torch.linspace(x0, x1, nx).to(self.device)
        t = torch.linspace(t0, t1, nt).to(self.device)

        self.x_test = x.cpu().numpy()
        self.t_test = t.cpu().numpy()
        
        # create meshgrid
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        # reshape for network input
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        
        # Store as instance variables
        self.X_test = X
        self.T_test = T
        self.X_flat_test = X_flat
        self.T_flat_test = T_flat
        
        # Set flag indicating test domain has been created
        self.test_domain_created = True
        
        print(f"Testing domain created with {nx}x{nt} grid points.")
    
        return 

    # LOSS FUNCTIONS 
    def compute_pde_residual(self, x, t):
        """
        Compute the PDE residual for the KdV equation. (physics loss)
        """
        # copies of x and t that require gradients
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        
        # forward pass = u(x,t)
        u = self.net(x, t)
        
        # calculate derivatives needed for KdV
        
        # first-order derivatives
        u_grad = torch.autograd.grad(
            outputs=u, 
            inputs=[t, x], 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )
        u_t = u_grad[0]  # temporal derivative
        u_x = u_grad[1]  # first spatial derivative
        
        # second-order spatial derivative (u_xx)
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # third-order spatial derivative (u_xxx)
        u_xxx = torch.autograd.grad(
            outputs=u_xx,
            inputs=x,
            grad_outputs=torch.ones_like(u_xx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # compute PDE residual for KdV equation
        # Standard form: u_t + alpha*u*u_x + beta*u_xxx = 0
        residual = u_t + self.alpha * u * u_x + self.beta * u_xxx
        
        return residual

    def compute_data_loss(self):
        """
        Compute the data loss for the KdV equation. (ICs and BCs)
        Returns both total data loss and individual components.
        """
        # forward pass for IC and BC points = u(x,t)
        u_pred_initial = self.net(self.x_initial, self.t_initial)
        u_pred_boundary = self.net(self.x_boundary, self.t_boundary)
        
        # compute and combine MSE losses
        initial_loss = torch.mean((u_pred_initial - self.u_initial)**2)
        boundary_loss = torch.mean((u_pred_boundary - self.u_boundary)**2)
        
        # total data loss
        data_loss = initial_loss + boundary_loss
        
        return data_loss, initial_loss, boundary_loss

    def loss_fn(self):
        """
        Compute the total loss function: combining data and PDE losses.
        Returns both the total loss and individual components.
        """
        # compute data loss
        data_loss, initial_loss, boundary_loss = self.compute_data_loss()
        
        # compute PDE residual and MSE residual loss
        residual = self.compute_pde_residual(self.x_collocation, self.t_collocation)
        pde_loss = torch.mean(residual**2) # we want MSE 

        # loss weights
        data_loss_weight = 0.1
        pde_loss_weight = 0.9
        
        # total loss 
        total_loss = data_loss_weight * data_loss + pde_loss_weight * pde_loss
        
        return total_loss, initial_loss, boundary_loss, pde_loss
    
    ## TRAINING 
    def train(self, adam_epochs=1000, lbfgs_epochs=50000, verbose=True, verbose_step=100):
        
        # initialize losses list 
        losses = {
            'total': [],
            'initial': [],
            'boundary': [],
            'pde': []
        }
        
        # Phase 1: Adam optimization
        if verbose:
            print("Starting Adam optimization...")
        
        optimizer = torch.optim.Adam(self.net.parameters())
        
        for epoch in range(adam_epochs):
            # zero gradients
            optimizer.zero_grad()
            
            # compute loss
            total_loss, initial_loss, boundary_loss, pde_loss = self.loss_fn()
            
            # backpropagation and optimization
            total_loss.backward()
            optimizer.step()
            
            # store losses
            losses['total'].append(total_loss.item())
            losses['initial'].append(initial_loss.item())
            losses['boundary'].append(boundary_loss.item())
            losses['pde'].append(pde_loss.item())
            
            # print progress
            if verbose and (epoch % verbose_step == 0 or epoch == adam_epochs - 1):
                print(f"Adam - Epoch {epoch}/{adam_epochs}, Total Loss: {total_loss.item():.6e}")
        
        # Store the number of Adam epochs as an instance variable
        self.adam_epochs = adam_epochs
        
        # Phase 2: L-BFGS optimization
        if verbose:
            print("\nStarting L-BFGS optimization...")
        
        # L-BFGS requires a closure that reevaluates the model and returns the loss
        def closure():
            optimizer.zero_grad()
            total_loss, initial_loss, boundary_loss, pde_loss = self.loss_fn()
            total_loss.backward()
            
            # Store current loss values
            losses['total'].append(total_loss.item())
            losses['initial'].append(initial_loss.item())
            losses['boundary'].append(boundary_loss.item())
            losses['pde'].append(pde_loss.item())
            
            # Print progress if verbose
            if verbose and len(losses['total']) % verbose_step == 0:
                print(f"L-BFGS - Iteration {len(losses['total']) - adam_epochs}, Total Loss: {total_loss.item():.6e}")
                
            return total_loss
        
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
            print(f"L-BFGS complete, Final Loss: {losses['total'][-1]:.6e}")

        # save losses as instance variable
        self.losses = losses
        
        return
    
    ## TESTING
    def analytical_solution(self, x, t):
        """
        Compute the analytical solution for KdV equation at any space-time point.
        
        Parameters:
        -----------
        x: torch.Tensor
            Spatial coordinates
        t: torch.Tensor
            Time coordinates
            
        Returns:
        --------
        torch.Tensor
            The exact solution at specified points
        """
        if self.num_solitons == 1:
            # One-soliton analytical solution
            k = 0.9
            phi = 12
            
            # For each time t, the soliton shifts by 4k²t
            argument = (k*x) - (4*(k**3) * t) + phi
            return 2 * (k**2) * torch.pow(1.0 / torch.cosh(argument), 2)
        else:
            raise ValueError(f"Analytical solution for {self.num_solitons} solitons not implemented")


    def single_soliton(self, x, t, k, phi):
        """
        Compute the analytical solution for a single soliton.
        """
        # Use the k and phi parameters directly, don't override them
        argument = (k*x) - (4*(k**3) * t) + phi
        return 2 * (k**2) * torch.pow(1.0 / torch.cosh(argument), 2)
        
    def compute_linear_combination(self, x, t, k_vector, phi_vector):
        """
        Compute the analytical solution for a linear combination of solitons.
        """
        solution = torch.zeros_like(x)
        for k, phi in zip(k_vector, phi_vector):
            solution += self.single_soliton(x, t, k, phi)

        return solution.cpu().numpy()

    def compute_test_solutions(self, force_recompute=False):
        """
        Compute the PINN prediction and analytical solution (if available) over the entire test domain.
        
        This function evaluates the trained neural network over the previously created test domain
        and stores the results as instance variables. It also computes the analytical solution when 
        available for comparison.
        
        Parameters:
        -----------
        force_recompute: bool
            If True, recompute the solutions even if they've been computed before
            
        Returns:
        --------
        dict:
            Dictionary containing 'predicted' and (if available) 'exact' solutions
        
        Raises:
        -------
        RuntimeError:
            If setup_testing_domain() hasn't been called yet
        """
        # Check if test domain has been created
        if not hasattr(self, 'test_domain_created') or not self.test_domain_created:
            raise RuntimeError("Test domain not created. Call setup_testing_domain() first.")
        
        # Skip computation if already done (unless forced)
        if hasattr(self, 'test_solution_computed') and self.test_solution_computed and not force_recompute:
            print("Test solutions already computed. Use force_recompute=True to recompute.")
            return
        
        print("Computing solutions over the test domain...")
        
        # Compute PINN predictions
        with torch.no_grad():
            U_pred_flat = self.net(self.X_flat_test, self.T_flat_test)
            U_pred = U_pred_flat.reshape(self.X_test.shape)
        
        # Store as instance variable
        self.U_pred = U_pred
        
        # Try to compute analytical solution if available
        try:
            print("Attempting to compute analytical solution...")
            U_exact = self.analytical_solution(self.X_test, self.T_test)
            self.U_exact = U_exact
            has_exact = True
            print("Analytical solution computed successfully.")
        except (ValueError, NotImplementedError, AttributeError) as e:
            print(f"Analytical solution not available: {e}")
            has_exact = False
        
        # Set flag that solutions have been computed
        self.test_solution_computed = True
        
        # Create numpy versions for easy plotting
        self.X_np = self.X_test.cpu().numpy()
        self.T_np = self.T_test.cpu().numpy()
        self.U_pred_np = self.U_pred.cpu().numpy()
        
        if has_exact:
            self.U_exact_np = self.U_exact.cpu().numpy()
        
        print("Test solutions computation complete.")
        
        # Return dictionary of solutions
        result = {'predicted': self.U_pred}
        if has_exact:
            result['exact'] = self.U_exact

        self.U_lin_comb_np = self.compute_linear_combination(self.X_test, self.T_test, self.k_vector, self.phi_vector)
        
        return result

    def test(self, plot_heatmap=False):
        """
        Compute error metrics between the predicted and analytical solutions.
        
        Parameters:
        -----------
        plot_heatmap: bool
            Whether to plot a heatmap of the absolute error
            
        Returns:
        --------
        dict:
            Dictionary containing error metrics
        
        Notes:
        ------
        This function requires an analytical solution to compare against.
        """
        # Ensure test domain exists and solutions are computed
        if not hasattr(self, 'test_domain_created') or not self.test_domain_created:
            print("Test domain not created. Creating default testing domain...")
            self.setup_testing_domain()
        
        if not hasattr(self, 'test_solution_computed') or not self.test_solution_computed:
            print("Test solutions not computed. Computing solutions...")
            self.compute_test_solutions()
        
        # Check if analytical solution exists
        if not hasattr(self, 'U_exact') or self.U_exact is None:
            raise ValueError("Analytical solution not available. Error metrics cannot be computed.")
        
        # Compute absolute error
        abs_error = torch.abs(self.U_pred - self.U_exact)
        self.abs_error_np = abs_error.cpu().numpy() # Store error as numpy array for plotting
        
        # Compute mean absolute error (MAE)
        mae = torch.mean(abs_error).item()
        
        # Compute relative L2 error
        l2_error = torch.sqrt(torch.mean((self.U_pred - self.U_exact)**2)).item()
        rel_l2_error = l2_error / torch.sqrt(torch.mean(self.U_exact**2)).item()
        
        # Compute maximum error (L-infinity norm)
        max_error = torch.max(abs_error).item()
        
        # Create dictionary of error metrics
        metrics = {
            'mae': mae,
            'l2_error': l2_error,
            'rel_l2_error': rel_l2_error,
            'max_error': max_error
        }
        
        # Print a summary of the error metrics
        print(f"Error Metrics Summary:")
        print(f"  Mean Absolute Error (MAE): {mae:.6e}")
        print(f"  L2 Error: {l2_error:.6e}")
        print(f"  Relative L2 Error: {rel_l2_error:.6e}")
        print(f"  Maximum Error: {max_error:.6e}")
        
        # Plot error heatmap if requested
        if plot_heatmap:
            plt.figure(figsize=(10, 6))
            # Use LogNorm for logarithmic scale colorbar
            from matplotlib.colors import LogNorm
            
            contour = plt.pcolormesh(self.T_np[0, :], self.X_np[:, 0], self.abs_error_np, 
                                cmap='hot', norm=LogNorm())
            plt.colorbar(contour, label='Absolute Error |u_pred - u_exact| (log scale)')
            plt.xlabel('Time (t)')
            plt.ylabel('Position (x)')
            plt.tight_layout()
        
        return 

    ## PLOTTING 
    def compute_profile(self, t_val):
        """
        Extract a solution profile at a specific time point from the computed test solutions.
        
        Parameters:
        -----------
        t_val: float
            Time value at which to extract the profile
            
        Returns:
        --------
        tuple: (x_profile, u_profile)
            Arrays containing the x-coordinates and solution values at the specified time
        """
        # Ensure test domain exists
        if not hasattr(self, 'test_domain_created') or not self.test_domain_created:
            print("Test domain not created. Creating default testing domain...")
            self.setup_testing_domain()
        
        # Ensure solutions are computed
        if not hasattr(self, 'test_solution_computed') or not self.test_solution_computed:
            print("Test solutions not computed. Computing solutions...")
            self.compute_test_solutions()
        
        # Find closest time index
        t_idx = np.argmin(np.abs(self.T_np[0, :] - t_val))
        
        # Extract the profile at the given time
        x_profile = self.X_np[:, t_idx]
        u_profile = self.U_pred_np[:, t_idx]
        
        return x_profile, u_profile

    def plot_profiles(self, t_points=[0.0, 2.0, 4.0, 6.0, 8.0]):
        """
        Plot solution profiles at multiple time points on the same plot.
        """
        # Ensure test domain exists and solutions are computed
        if not hasattr(self, 'test_domain_created') or not self.test_domain_created:
            print("Test domain not created. Creating default testing domain...")
            self.setup_testing_domain()
        
        if not hasattr(self, 'test_solution_computed') or not self.test_solution_computed:
            print("Test solutions not computed. Computing solutions...")
            self.compute_test_solutions()
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Plot PINN predictions for each time sample
        for t_val in t_points:
            # Compute the profile at this time
            x_profile, u_profile = self.compute_profile(t_val)
            
            # Plot solution with different colors for each time step
            plt.plot(x_profile, u_profile, label=f't = {t_val}')
        
        # If exact solution is available, plot for the last time point
        if hasattr(self, 'U_exact_np'):
            t_idx = np.argmin(np.abs(self.T_np[0, :] - t_points[-1]))
            plt.plot(self.X_np[:, t_idx], self.U_exact_np[:, t_idx], 'k--', 
                    label=f'Exact (t = {t_points[-1]})')
        
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.tight_layout()
        
        return

    def plot_losses(self, losses, component=None, show_optimizer_switch=True):
        """
        Plot the losses over epochs to visualize training progress.
        
        Parameters:
        -----------
        losses: dict
            Dictionary with loss components
        component: str, list, or None
            If None, plots total loss (default behavior)
            If 'all', plots all loss components
            If a list of strings, plots all specified components
        show_optimizer_switch: bool
            Whether to show a vertical line indicating the switch from Adam to LBFGS optimizer
        """
        plt.figure(figsize=self.figsize)
        
        # Convert single component to list for uniform processing
        if component is None:
            component = ['total']
        elif component == 'all':
            component = list(losses.keys())
        elif isinstance(component, str):
            component = [component]
        
        # Plot each requested component
        for comp in component:
            if comp in losses:
                plt.plot(losses[comp], label=f'{comp} loss')
            else:
                raise ValueError(f"Unknown loss component '{comp}'")
        
        # Add vertical line for optimizer switch if requested
        if show_optimizer_switch and hasattr(self, 'adam_epochs') and self.adam_epochs > 0:
            # Add a vertical line at the Adam-to-LBFGS transition
            plt.axvline(x=self.adam_epochs, color='r', linestyle='--', alpha=0.7)
            
            # Add text annotation
            plt.text(self.adam_epochs + 5, 0.2, 'Adam → L-BFGS', 
                     rotation=90, verticalalignment='center', transform=plt.gca().get_xaxis_transform())
        
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
            
        plt.tight_layout()
        return

    def plot_spacetime(self, show_data=False):
        """
        Plot the solution as a color map over space and time.
        
        Parameters:
        -----------
        show_data: bool
            Whether to show the training data points (collocation, initial, boundary)
        """
        # Ensure test domain exists and solutions are computed
        if not hasattr(self, 'test_domain_created') or not self.test_domain_created:
            self.setup_testing_domain()
        
        if not hasattr(self, 'test_solution_computed') or not self.test_solution_computed:
            self.compute_test_solutions()

        plt.figure(figsize=(10, 6))
        
        # Plot the solution as a color map
        contour = plt.pcolormesh(self.T_np[0, :], self.X_np[:, 0], self.U_pred_np, 
                                cmap='plasma', shading='auto')
        plt.colorbar(contour, label='u(x,t)')
        
        # Plot the training data points if requested
        if show_data:
            # Convert collocation points to numpy
            x_coll_np = self.x_collocation.cpu().numpy()
            t_coll_np = self.t_collocation.cpu().numpy()
            
            # Plot with different markers and sizes for clarity
            plt.scatter(t_coll_np, x_coll_np, marker='.', color='black', s=0.3, alpha=0.5, 
                    label='Collocation points')
            plt.scatter(self.t_initial.cpu().numpy(), self.x_initial.cpu().numpy(),
                    marker='x', color='white', s=3, label='Initial condition')
            plt.scatter(self.t_boundary.cpu().numpy(), self.x_boundary.cpu().numpy(),
                    marker='o', color='red', s=1, label='Boundary condition')
            plt.legend(loc='upper right', fontsize='small')
        
        plt.xlabel('Time (t)')
        plt.ylabel('Position (x)')
        plt.tight_layout()
        
        return

    def animate_solution(self, running_time=5, fps=60, save_path=None, dpi=200):
        """
        Create an animation of the solution profiles over time.
        
        Parameters:
        -----------
        running_time: float
            Duration of the animation in seconds
        fps: int
            Frames per second
        save_path: str or None
            If provided, save the animation to this file path
        dpi: int
            Resolution for saved animation
            
        Returns:
        --------
        matplotlib.animation.Animation
            The animation object
        """
        # Ensure test domain exists and solutions are computed
        if not hasattr(self, 'test_domain_created') or not self.test_domain_created:
            print("Test domain not created. Creating default testing domain...")
            self.setup_testing_domain()
        
        if not hasattr(self, 'test_solution_computed') or not self.test_solution_computed:
            print("Test solutions not computed. Computing solutions...")
            self.compute_test_solutions()
        
        # Calculate the number of frames needed for the requested duration and fps
        n_frames = int(running_time * fps)
        
        # Generate equally spaced time points over the entire time domain
        t_start, t_end = self.t_lims
        time_points = np.linspace(t_start, t_end, n_frames)
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=dpi)
        
        # Get the initial profile to set up the plot
        x_profile, u_profile = self.compute_profile(t_start)
        
        # Create the line that will be updated with increased line width for better visibility
        line, = ax.plot(x_profile, u_profile, linewidth=2, color='black')
        
        # Set up the plot limits
        ax.set_xlim(self.x_lims)
        
        # Set y-limits based on the solution range with 10% margin
        y_min = np.min(self.U_pred_np) * 1.1 if np.min(self.U_pred_np) < 0 else np.min(self.U_pred_np) * 0.9
        y_max = np.max(self.U_pred_np) * 1.1
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Text to display the current time with improved formatting
        time_text = ax.text(0.02, 0.95, f't = {t_start:.3f}', transform=ax.transAxes, 
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Update function for animation
        def update(frame):
            t_val = time_points[frame]
            x_profile, u_profile = self.compute_profile(t_val)
            
            # Update the line data
            line.set_data(x_profile, u_profile)
            
            # Update the time text
            time_text.set_text(f't = {t_val:.3f}')
            
            return line, time_text
        
        plt.tight_layout()
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=1000/fps, 
            blit=True, repeat=True
        )
        
        # Save animation if path is provided
        if save_path:
            writer = animation.PillowWriter(
                fps=fps,
                metadata=dict(artist='PINN Simulation'),
                bitrate=1800
            )

            anim.save(save_path, writer=writer, dpi=dpi)
            print(f"Animation saved to {save_path}")
        
        return anim
    
    # SAVING RESULTS
    def save_model_result(self, filename):
        """
        Save model results to a JSON file.
        
        Parameters:
        -----------
        filename : str
            Path to save the JSON file
        """

        
        # Make sure solutions are computed
        if not hasattr(self, 'test_solution_computed') or not self.test_solution_computed:
            print("Test solutions not computed. Computing solutions...")
            self.compute_test_solutions()
        
        # Prepare the results dictionary
        results = {
            "domain": {
                "x": self.X_np[:, 0].tolist(),  # First column = x coordinates
                "t": self.T_np[0, :].tolist()   # First row = t coordinates
            },
            "solution": {
                "u_pred": self.U_pred_np.tolist(),
                "u_lin_comb": self.U_lin_comb_np.tolist()
            },
            "losses": {}
        }
        
        # Add exact solution if available
        if hasattr(self, 'U_exact_np'):
            results["solution"]["u_exact"] = self.U_exact_np.tolist()
        
        # Add losses if available
        if hasattr(self, 'losses'):
            # Convert any NumPy or PyTorch values to regular Python types
            losses_dict = {}
            for key, values in self.losses.items():
                losses_dict[key] = [float(val) for val in values]
            results["losses"] = losses_dict
        
        # Check if file already exists
        if os.path.exists(filename):
            raise FileExistsError(f"File {filename} already exists. Please choose a different filename.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f)
        
        print(f"Results saved to {filename}")
        return