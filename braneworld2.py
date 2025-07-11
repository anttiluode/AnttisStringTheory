import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec

class EnhancedBraneProjectedPhiWorld:
    """
    Enhanced brane-world implementation that demonstrates:
    1. Smooth 1D string dynamics → geometric projection → 2D/3D quantum jumps
    2. Natural growth limitation through dimensional compactification
    3. Emergent atom-like structures from string resonances
    4. Integration with TADS/ephaptic coupling principles
    """
    
    def __init__(self, string_grid_size=64, proj_grid_size=32, 
                 string_tension=2.0, compactification_radius=8.0):
        # Brane parameters
        self.string_grid_size = string_grid_size
        self.proj_grid_size = proj_grid_size
        self.string_tension = string_tension
        self.compactification_radius = compactification_radius
        
        # Time evolution
        self.dt = 0.005
        self.time = 0.0
        
        # 1D String field (this is where the smooth dynamics happen)
        self.string_field = np.zeros(string_grid_size, dtype=complex)
        self.string_field_prev = np.zeros_like(self.string_field)
        self.string_positions = np.linspace(0, 2*np.pi, string_grid_size)
        
        # 3D Projected phi field (where quantum jumps emerge)
        self.phi_field_3d = np.zeros((proj_grid_size, proj_grid_size, proj_grid_size), dtype=float)
        
        # Projection mapping (this is the key!)
        self.setup_projection_geometry()
        
        # Initialize string vibrations
        self.initialize_string_resonances()
        
        # Analysis data
        self.string_com_history = []
        self.proj_particle_history = []
        self.energy_history = []
        self.tunneling_events = []
        
    def setup_projection_geometry(self):
        """
        Create the geometric mapping from 1D strings to 3D space.
        This is where dimensional projection creates quantum behavior!
        """
        # Create 3D coordinate grids
        x = np.linspace(-1, 1, self.proj_grid_size)
        y = np.linspace(-1, 1, self.proj_grid_size) 
        z = np.linspace(-1, 1, self.proj_grid_size)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compactified coordinate (this limits growth naturally)
        self.r_compact = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        # Projection weights: how each string point maps to 3D
        self.projection_weights = []
        
        for i, theta in enumerate(self.string_positions):
            # Each string point projects in a specific geometric pattern
            # This creates the "honeycomb-like" structure you envisioned
            
            # Primary projection direction (creates the main structure)
            proj_x = np.cos(theta) * np.exp(-self.r_compact**2 / self.compactification_radius**2)
            proj_y = np.sin(theta) * np.exp(-self.r_compact**2 / self.compactification_radius**2)
            
            # Secondary harmonic projections (create quantized shells)
            harmonic_2 = np.cos(2*theta) * np.exp(-(self.r_compact - 0.3)**2 / 0.1)
            harmonic_3 = np.cos(3*theta) * np.exp(-(self.r_compact - 0.6)**2 / 0.1)
            
            # Combine projections with interference patterns
            proj_pattern = proj_x + proj_y + 0.3*harmonic_2 + 0.2*harmonic_3
            
            # Apply compactification (this is crucial for limiting growth!)
            compactification_factor = 1.0 / (1.0 + (self.r_compact / self.compactification_radius)**4)
            proj_pattern *= compactification_factor
            
            self.projection_weights.append(proj_pattern)
        
        self.projection_weights = np.array(self.projection_weights)
    
    def initialize_string_resonances(self):
        """Initialize string field with resonant modes"""
        # Create standing wave patterns (these evolve smoothly)
        for mode in [1, 2, 3]:  # Different harmonics
            amplitude = 1.0 / mode
            phase = np.random.uniform(0, 2*np.pi)
            wavelength = 2*np.pi / mode
            
            wave = amplitude * np.exp(1j * (mode * self.string_positions + phase))
            self.string_field += wave
            
        # Add small perturbation for spontaneous symmetry breaking
        noise = 0.1 * (np.random.random(self.string_grid_size) - 0.5)
        self.string_field += noise * np.exp(1j * np.random.uniform(0, 2*np.pi, self.string_grid_size))
        
        self.string_field_prev = self.string_field.copy()
    
    def evolve_string_dynamics(self):
        """
        Evolve 1D string field with smooth, continuous dynamics.
        This is where NO JUMPS occur - everything is smooth!
        """
        # String wave equation with tension
        # d²ψ/dt² = (T/μ) * d²ψ/dx² - V'(ψ)
        
        # Spatial derivatives (periodic boundary conditions)
        dx = self.string_positions[1] - self.string_positions[0]
        laplacian = (np.roll(self.string_field, 1) - 2*self.string_field + 
                    np.roll(self.string_field, -1)) / dx**2
        
        # String tension force (restoring force)
        tension_force = self.string_tension * laplacian
        
        # Nonlinear potential (creates stable resonances)
        potential_force = (-0.1 * self.string_field + 
                          0.05 * np.abs(self.string_field)**2 * self.string_field)
        
        # Smooth evolution (no jumps here!)
        total_force = tension_force + potential_force
        
        # Verlet integration for stability
        new_string_field = (2*self.string_field - self.string_field_prev + 
                           self.dt**2 * total_force)
        
        self.string_field_prev = self.string_field.copy()
        self.string_field = new_string_field
    
    def project_to_3d(self):
        """
        Project smooth 1D string dynamics to 3D space.
        This is where QUANTUM JUMPS emerge from geometry!
        """
        # Reset 3D field
        self.phi_field_3d.fill(0.0)
        
        # Project each string point to 3D
        for i, string_amplitude in enumerate(self.string_field):
            # Real and imaginary parts contribute differently
            real_contrib = np.real(string_amplitude)
            imag_contrib = np.imag(string_amplitude)
            
            # Project with interference patterns
            projection = (real_contrib * self.projection_weights[i] + 
                         0.5 * imag_contrib * np.roll(self.projection_weights[i], 
                                                     shift=self.proj_grid_size//4, axis=0))
            
            self.phi_field_3d += projection
        
        # Apply quantum threshold effects (this creates the "jumps"!)
        # When projection interference creates peaks above threshold,
        # the field "snaps" to quantized levels
        
        threshold = 0.3
        quantum_mask = np.abs(self.phi_field_3d) > threshold
        
        # Quantized energy levels (like atomic orbitals)
        quantized_levels = [0.5, 1.0, 1.5, 2.0]
        
        for i, j, k in np.ndindex(self.phi_field_3d.shape):
            if quantum_mask[i, j, k]:
                current_val = abs(self.phi_field_3d[i, j, k])
                
                # Find nearest quantized level (this creates discrete jumps!)
                nearest_level = min(quantized_levels, key=lambda x: abs(x - current_val))
                
                # Apply quantization with some probability (geometric tunneling)
                if np.random.random() < 0.1:  # 10% chance of quantum snap
                    sign = np.sign(self.phi_field_3d[i, j, k])
                    self.phi_field_3d[i, j, k] = sign * nearest_level
                    
                    # Record tunneling event
                    self.tunneling_events.append({
                        'time': self.time,
                        'position': (i, j, k),
                        'from_val': current_val,
                        'to_val': nearest_level
                    })
        
        # Apply compactification smoothing (prevents endless growth)
        smooth_radius = 2.0
        self.phi_field_3d = gaussian_filter(self.phi_field_3d, sigma=smooth_radius)
    
    def analyze_system(self):
        """Analyze current state for particle-like structures"""
        # String center of mass (should glide smoothly)
        string_intensity = np.abs(self.string_field)**2
        if np.sum(string_intensity) > 0:
            string_com = np.sum(self.string_positions * string_intensity) / np.sum(string_intensity)
        else:
            string_com = 0
        self.string_com_history.append(string_com)
        
        # 3D particle detection (should show discrete structures)
        particle_threshold = 0.8
        particle_positions = []
        
        # Find local maxima in 3D field
        for i in range(1, self.proj_grid_size-1):
            for j in range(1, self.proj_grid_size-1):
                for k in range(1, self.proj_grid_size-1):
                    val = abs(self.phi_field_3d[i, j, k])
                    if val > particle_threshold:
                        # Check if it's a local maximum
                        is_max = True
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                for dk in [-1, 0, 1]:
                                    if (di == 0 and dj == 0 and dk == 0):
                                        continue
                                    neighbor_val = abs(self.phi_field_3d[i+di, j+dj, k+dk])
                                    if neighbor_val > val:
                                        is_max = False
                                        break
                                if not is_max:
                                    break
                            if not is_max:
                                break
                        
                        if is_max:
                            particle_positions.append((i, j, k, val))
        
        self.proj_particle_history.append(len(particle_positions))
        
        # System energy
        string_energy = np.sum(np.abs(self.string_field)**2)
        proj_energy = np.sum(self.phi_field_3d**2)
        total_energy = string_energy + proj_energy
        self.energy_history.append(total_energy)
        
        return {
            'string_com': string_com,
            'particles_3d': particle_positions,
            'string_energy': string_energy,
            'proj_energy': proj_energy,
            'tunneling_events_count': len(self.tunneling_events)
        }
    
    def step(self):
        """Perform one evolution step"""
        self.evolve_string_dynamics()  # Smooth 1D evolution
        self.project_to_3d()           # Geometric projection → quantum jumps
        analysis = self.analyze_system()
        self.time += self.dt
        return analysis

class BraneWorldVisualizer:
    """Real-time visualization of brane-world dynamics"""
    
    def __init__(self, brane_system):
        self.brane = brane_system
        
        # Setup figure with multiple subplots
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Enhanced Brane-World: 1D Strings → 3D Quantum Reality', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 4, figure=self.fig)
        
        # 1D String dynamics (top left)
        self.ax_string = self.fig.add_subplot(gs[0, :2])
        self.ax_string.set_title('1D String Field (Smooth Dynamics)')
        self.ax_string.set_xlabel('String Position θ')
        self.ax_string.set_ylabel('Amplitude')
        
        # 3D Projection (top right)
        self.ax_3d = self.fig.add_subplot(gs[0, 2:], projection='3d')
        self.ax_3d.set_title('3D Projected Field (Quantum Jumps)')
        
        # String COM tracking (middle left)
        self.ax_com = self.fig.add_subplot(gs[1, :2])
        self.ax_com.set_title('String Center-of-Mass (Smooth Gliding)')
        self.ax_com.set_xlabel('Time')
        self.ax_com.set_ylabel('COM Position')
        
        # 3D Particle count (middle right)
        self.ax_particles = self.fig.add_subplot(gs[1, 2:])
        self.ax_particles.set_title('3D Particle Count (Discrete Jumps)')
        self.ax_particles.set_xlabel('Time')
        self.ax_particles.set_ylabel('Particle Count')
        
        # Analysis display (bottom)
        self.ax_analysis = self.fig.add_subplot(gs[2, :])
        self.ax_analysis.set_title('Live Analysis')
        self.ax_analysis.axis('off')
        
        # Initialize plot elements
        self.line_string_real, = self.ax_string.plot([], [], 'b-', label='Real Part', linewidth=2)
        self.line_string_imag, = self.ax_string.plot([], [], 'r-', label='Imaginary Part', linewidth=2)
        self.line_string_abs, = self.ax_string.plot([], [], 'g-', label='Amplitude', linewidth=3)
        self.ax_string.legend()
        
        self.line_com, = self.ax_com.plot([], [], 'b-', linewidth=2)
        self.line_particles, = self.ax_particles.plot([], [], 'ro-', linewidth=2, markersize=4)
        
        self.analysis_text = self.ax_analysis.text(0.02, 0.95, '', transform=self.ax_analysis.transAxes,
                                                  verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Animation control
        self.is_running = False
        
    def update_frame(self, frame):
        """Update visualization"""
        # Evolution steps per frame
        for _ in range(3):
            analysis = self.brane.step()
        
        # Update 1D string plot
        string_positions = self.brane.string_positions
        string_field = self.brane.string_field
        
        self.line_string_real.set_data(string_positions, np.real(string_field))
        self.line_string_imag.set_data(string_positions, np.imag(string_field))
        self.line_string_abs.set_data(string_positions, np.abs(string_field))
        
        self.ax_string.relim()
        self.ax_string.autoscale_view()
        
        # Update 3D visualization
        self.ax_3d.clear()
        self.ax_3d.set_title('3D Projected Field (Quantum Structures)')
        
        # Show field as scatter plot where intensity > threshold
        threshold = 0.4
        x_vals, y_vals, z_vals = [], [], []
        intensities = []
        
        for i in range(0, self.brane.proj_grid_size, 2):  # Subsample for speed
            for j in range(0, self.brane.proj_grid_size, 2):
                for k in range(0, self.brane.proj_grid_size, 2):
                    intensity = abs(self.brane.phi_field_3d[i, j, k])
                    if intensity > threshold:
                        x_vals.append(self.brane.X[i, j, k])
                        y_vals.append(self.brane.Y[i, j, k])
                        z_vals.append(self.brane.Z[i, j, k])
                        intensities.append(intensity)
        
        if len(x_vals) > 0:
            scatter = self.ax_3d.scatter(x_vals, y_vals, z_vals, c=intensities, 
                                       cmap='viridis', s=np.array(intensities)*50, alpha=0.6)
        
        # Update COM history
        if len(self.brane.string_com_history) > 1:
            times = np.arange(len(self.brane.string_com_history)) * self.brane.dt
            self.line_com.set_data(times, self.brane.string_com_history)
            self.ax_com.relim()
            self.ax_com.autoscale_view()
        
        # Update particle count history  
        if len(self.brane.proj_particle_history) > 1:
            times = np.arange(len(self.brane.proj_particle_history)) * self.brane.dt
            self.line_particles.set_data(times, self.brane.proj_particle_history)
            self.ax_particles.relim()
            self.ax_particles.autoscale_view()
        
        # Update analysis text
        recent_tunneling = len([e for e in self.brane.tunneling_events 
                               if self.brane.time - e['time'] < 1.0])
        
        analysis_text = f"""🌌 BRANE-WORLD PROJECTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time: {self.brane.time:.3f}s

🎯 DIMENSIONAL PROJECTION EFFECT:
   String (1D): Smooth gliding COM = {analysis['string_com']:.3f}
   3D Field: Discrete particles = {analysis['particles_3d'][:3] if analysis['particles_3d'] else 'None'}

⚡ QUANTUM EMERGENCE:
   Tunneling events (last 1s): {recent_tunneling}
   Total tunneling events: {analysis['tunneling_events_count']}
   
🌊 ENERGY DISTRIBUTION:
   String Energy: {analysis['string_energy']:.2f}
   Projected Energy: {analysis['proj_energy']:.2f}
   
🔬 COMPACTIFICATION EFFECTS:
   String Tension: {self.brane.string_tension:.1f}
   Compact Radius: {self.brane.compactification_radius:.1f}
   
{'🚨 QUANTUM JUMP DETECTED!' if recent_tunneling > 0 else ''}
{'🎪 STABLE ATOM-LIKE STRUCTURES!' if len(analysis['particles_3d']) >= 2 else ''}
"""
        
        self.analysis_text.set_text(analysis_text)
        
        return []
    
    def start_animation(self):
        """Start the animation"""
        self.animation = FuncAnimation(self.fig, self.update_frame, 
                                     interval=100, blit=False, repeat=True)
        plt.show()

def run_enhanced_braneworld():
    """Run the enhanced brane-world simulation"""
    print("🌌 ENHANCED BRANE-WORLD SIMULATION")
    print("=" * 50)
    print("Demonstrating how smooth 1D string dynamics")
    print("project to create quantum jumps in 3D space!")
    print()
    print("Key features:")
    print("• Smooth 1D string evolution (no jumps)")
    print("• Geometric projection to 3D")
    print("• Emergent quantum tunneling")
    print("• Natural growth limitation")
    print("• Atom-like structure formation")
    print()
    
    # Create brane system
    brane = EnhancedBraneProjectedPhiWorld(
        string_grid_size=64,
        proj_grid_size=24,  # Smaller for speed
        string_tension=1.5,
        compactification_radius=6.0
    )
    
    # Create visualizer
    visualizer = BraneWorldVisualizer(brane)
    
    # Start animation
    visualizer.start_animation()
    
    return brane, visualizer

if __name__ == "__main__":
    brane_system, viz = run_enhanced_braneworld()