# Make key classes available directly from the package
from .base import PINN, PINN1D
from .burgers import BURGERS
from .oscillator import DampedOscillatorPINN

# Define what gets imported with "from pinns.models import *"
__all__ = ['PINN', 'PINN1D', 'BURGERS', 'DampedOscillatorPINN'] 