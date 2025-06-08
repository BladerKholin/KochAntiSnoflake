import jax
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
import numpy as np
import time

class JAXKochSnowflake:
    def __init__(self):
        # Pre-compile the core functions
        self._koch_step_compiled = jit(self._koch_iteration_step)
    
    @staticmethod
    @jit
    def _koch_transform_segment(p1, p2):
        """Transform a single segment into Koch curve (returns 4 new segments)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate the key points
        p_1_3 = jnp.array([p1[0] + dx/3, p1[1] + dy/3])
        p_2_3 = jnp.array([p1[0] + 2*dx/3, p1[1] + 2*dy/3])
        
        # Calculate the peak point
        mid_x = (p_1_3[0] + p_2_3[0]) / 2
        mid_y = (p_1_3[1] + p_2_3[1]) / 2
        
        segment_length = jnp.sqrt(dx**2 + dy**2)
        height = jnp.sqrt(3) / 6 * segment_length
        
        # Perpendicular vector
        perp_x = -dy / segment_length * height
        perp_y = dx / segment_length * height
        peak = jnp.array([mid_x + perp_x, mid_y + perp_y])
        
        # Return 4 new segments as a 4x2x2 array
        return jnp.array([
            [p1, p_1_3],
            [p_1_3, peak],
            [peak, p_2_3],
            [p_2_3, p2]
        ])
    
    def _koch_iteration_step(self, segments):
        """Apply Koch transformation to all segments vectorized"""
        # Apply transformation to all segments in parallel
        transformed = vmap(self._koch_transform_segment)(segments[:, 0], segments[:, 1])
        
        # Reshape from (N, 4, 2, 2) to (N*4, 2, 2)
        n_segments = segments.shape[0]
        return transformed.reshape(n_segments * 4, 2, 2)
    
    def generate_snowflake_segments(self, iterations=4, size=1.0):
        """Generate Koch snowflake segments using pure JAX operations"""
        # Initial triangle vertices
        height = size * jnp.sqrt(3) / 2
        vertices = jnp.array([
            [-size/2, -height/3],      # Bottom left
            [size/2, -height/3],       # Bottom right  
            [0, 2*height/3]            # Top
        ])
        
        # Initial 3 segments (triangle sides)
        segments = jnp.array([
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]]
        ])
        
        # Apply Koch transformation iteratively
        # Use Python loop with pre-compiled JAX function
        for _ in range(iterations):
            segments = self._koch_step_compiled(segments)
        
        return segments
    
    def segments_to_points(self, segments):
        """Convert segments to ordered points (JIT compiled)"""
        return segments[:, 0]  # Take start point of each segment
    
    def generate_snowflake_points(self, iterations=4, size=1.0):
        """Generate final snowflake points"""
        segments = self.generate_snowflake_segments(iterations, size)
        return self.segments_to_points(segments)

def benchmark_implementations():
    """Benchmark original vs JAX implementations"""
    print("Performance Comparison: Original vs JAX")
    print("=" * 50)
    
    # Try to import original implementation
    original = None
    try:
        from ClassicKoch import KochSnowflake
        original = KochSnowflake()
        print("✓ Original KochSnowflake loaded successfully")
    except ImportError as e:
        print(f"✗ Could not import original KochSnowflake: {e}")
        print("  Make sure 'ClassicKoch.py' is in the same directory")
    except Exception as e:
        print(f"✗ Error loading original KochSnowflake: {e}")
    
    jax_simple = JAXKochSnowflake()
    
    test_iterations = [3, 4, 5, 6, 7, 8, 9, 10, 11]  # Extended test range
    
    for iterations in test_iterations:
        print(f"\nTesting {iterations} iterations:")
        
        # Original implementation (if available)
        if original is not None:
            try:
                start = time.time()
                orig_points = original.generate_snowflake(iterations, 3.0)
                orig_time = time.time() - start
                print(f"  Original:   {orig_time:.6f}s ({len(orig_points)} points)")
            except Exception as e:
                print(f"  Original:   ERROR - {e}")
                orig_time = None
        else:
            orig_time = None
        
        # JAX version
        try:
            start = time.time()
            jax_points = jax_simple.generate_snowflake_points(iterations, 3.0)
            jax_time = time.time() - start
            print(f"  JAX:        {jax_time:.6f}s ({len(jax_points)} points)")
        except Exception as e:
            print(f"  JAX:        ERROR - {e}")
            jax_time = None
        
        # Calculate speedup
        if orig_time and jax_time:
            print(f"  Speedup (JAX vs Original): {orig_time/jax_time:.2f}x")

def check_classickoch_structure():
    """Helper function to check the structure of ClassicKoch file"""
    print("Checking ClassicKoch.py structure...")
    print("=" * 40)
    
    try:
        import ClassicKoch
        print("✓ ClassicKoch module imported successfully")
        
        # Check available classes and functions
        available_items = [item for item in dir(ClassicKoch) if not item.startswith('_')]
        print(f"Available items: {available_items}")
        
        # Try to find KochSnowflake class
        if hasattr(ClassicKoch, 'KochSnowflake'):
            koch_class = getattr(ClassicKoch, 'KochSnowflake')
            print(f"✓ KochSnowflake class found")
            
            # Check methods
            methods = [method for method in dir(koch_class) if not method.startswith('_')]
            print(f"Available methods: {methods}")
            
            # Try to create instance
            try:
                instance = koch_class()
                print("✓ KochSnowflake instance created successfully")
                
                # Check if it has generate_snowflake method
                if hasattr(instance, 'generate_snowflake'):
                    print("✓ generate_snowflake method found")
                else:
                    print("✗ generate_snowflake method not found")
                    print("Available methods:", [m for m in dir(instance) if not m.startswith('_')])
                    
            except Exception as e:
                print(f"✗ Could not create KochSnowflake instance: {e}")
        else:
            print("✗ KochSnowflake class not found")
            
    except ImportError as e:
        print(f"✗ Could not import ClassicKoch: {e}")
        print("Make sure ClassicKoch.py is in the same directory as this script")
    except Exception as e:
        print(f"✗ Error checking ClassicKoch: {e}")
    
    print("-" * 40)

class OptimizedKochGenerator:
    """Production-ready optimized Koch snowflake generator"""
    
    def __init__(self):
        self.jax_generator = JAXKochSnowflake()
    
    def save_snowflake(self, filename="koch_jax.png", iterations=4, size=3.0,
                      dpi=300, figsize=(10, 10), line_width=1.0, color='blue'):
        """Generate and save Koch snowflake (optimized version)"""
        
        start_time = time.time()
        print(f"Generating JAX Koch Snowflake with {iterations} iterations...")
        
        # Generate points
        points = self.jax_generator.generate_snowflake_points(iterations, size)
        generation_time = time.time() - start_time
        
        # Convert to numpy for matplotlib
        points_np = np.array(points)
        x_coords = points_np[:, 0]
        y_coords = points_np[:, 1]
        
        # Close the shape
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        
        # Create plot
        plt.figure(figsize=figsize, facecolor='white')
        plt.plot(x_coords, y_coords, color=color, linewidth=line_width)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save
        plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                   pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        
        end_time = time.time()
        
        print(f"Saved as '{filename}'")
        print(f"Points: {len(points)}")
        print(f"Generation time: {generation_time:.4f}s")
        print(f"Total time: {end_time - start_time:.4f}s")
        print("-" * 40)

def main():
    """Main function with simple JAX implementation"""
    
    generator = OptimizedKochGenerator()
    
    # Generate series of snowflakes
    iterations_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print("JAX-Optimized Koch Snowflake Generation")
    print("=" * 50)
    
    total_start = time.time()
    
    for iterations in iterations_list:
        filename = f"koch_jax_iter_{iterations}.png"
        generator.save_snowflake(
            filename=filename,
            iterations=iterations,
            size=3.0,
            dpi=300,
            figsize=(8, 8),
            line_width=1.5,
            color='darkblue'
        )
    
    total_time = time.time() - total_start
    print(f"\nTotal time for all generations: {total_time:.4f}s")
    
    # Check ClassicKoch structure first
    print("\n" + "="*50)
    check_classickoch_structure()
    
    # Run benchmark
    print("\n" + "="*50)
    benchmark_implementations()

if __name__ == "__main__":
    # Use GPU by default (remove # to use CPU)
    # jax.config.update('jax_platform_name', 'cpu')
    main()