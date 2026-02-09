# Cloth_Simulator

A real-time physics-based cloth simulation built with OpenGL and Python, demonstrating advanced computer graphics concepts including 3D transformations, lighting, collision detection, and interactive rendering.

##  Project Overview

This project simulates a cloth dropping onto a sphere with realistic physics. The cloth wraps around the ball naturally, showing different levels of smoothness based on mesh resolution. Built as a Computer Graphics (COMP 342) course project at Kathmandu University.

###  Key Features

- **Real-time Physics Simulation**: Spring-mass system with Verlet integration
- **Rigid Body Collision**: Cloth wraps around sphere without penetration
- **Interactive Resolution Control**: 8 different mesh resolutions (4×4 to 32×32)
- **Dynamic Lighting**: Phong shading with smooth surface rendering
- **White Cloth with Black Grid**: Clean visualization of mesh structure
- **Camera Controls**: Rotate, zoom, and view from any angle
- **Pause/Resume**: Control simulation flow

## Computer Graphics Concepts Implemented

Based on COMP 342 syllabus, this project demonstrates:

### 1. **Geometric Transformations (3D)** - Chapter 6
- Translation of cloth particles in 3D space
- Rotation for camera control
- Scaling for zoom functionality
- Matrix transformations for view positioning

### 2. **3D Viewing** - Chapter 6
- Perspective projection using `gluPerspective()`
- Viewing pipeline with model-view transformations
- Interactive camera rotation and zoom
- Viewport to screen mapping

### 3. **Visible Surface Detection** - Chapter 7
- Z-Buffer method (depth testing) for proper rendering
- Back-face culling for performance optimization
- Depth buffer clearing and testing

### 4. **Illumination Models and Surface Rendering** - Chapter 8
- **Phong Shading**: Smooth per-vertex normal interpolation
- **Lighting Model**: Ambient, diffuse, and specular components
- **Light Sources**: Positioned light with GL_LIGHT0
- **Normal Computation**: Vertex normals calculated from face normals
- **Material Properties**: Specular highlights and shininess

### 5. **3D Mesh Rendering**
- Quad-based mesh representation
- Grid structure with structural, shear, and bend springs
- Wireframe and solid rendering modes
- Dynamic mesh deformation

### 6. **Color Models** - Chapter 9
- RGB color model for cloth (white: 1.0, 1.0, 1.0)
- RGB for sphere (red: 1.0, 0.3, 0.3)
- Color interpolation through lighting

### 7. **Advanced Topics** - Chapter 10
- **Computer Animation**: Real-time cloth simulation at 60 FPS
- **Physics-Based Animation**: Spring-mass dynamics
- **Collision Response**: Rigid body interactions

##  Technical Implementation

### Physics Engine
- **Verlet Integration**: Stable numerical integration for particle motion
- **Spring Constraints**: Maintains cloth structure with adjustable stiffness
- **Collision Detection**: Sphere-particle distance checking with rigid response
- **Damping**: Velocity damping for realistic energy loss

### Rendering Pipeline
1. Apply transformations (camera positioning)
2. Compute vertex normals for lighting
3. Enable depth testing and lighting
4. Render sphere using GLU quadrics
5. Render cloth mesh with grid lines
6. Switch to 2D orthographic projection for UI
7. Draw buttons and text overlays

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries
```bash
pip install pygame PyOpenGL PyOpenGL-accelerate numpy
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/ArSoham/cloth-drop-simulation.git
cd cloth-drop-simulation

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python cloth_drop_simulation.py
```

##  Controls

### Mouse Controls
- **Left Click + Drag**: Rotate camera around the scene
- **Scroll Wheel**: Zoom in/out
- **Click Buttons**: Change cloth resolution or reset

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `1-8` | Quick resolution change (4×4 to 32×32) |
| `SPACE` | Pause/Resume simulation |
| `W` | Toggle wireframe mode |
| `R` | Reset cloth to starting position |
| `ESC` | Exit simulation |

### Resolution Buttons
- **4×4**: 16 particles (fastest, less detail)
- **6×6**: 36 particles
- **8×8**: 64 particles
- **10×10**: 100 particles
- **12×12**: 144 particles (recommended)
- **14×14**: 196 particles
- **16×16**: 256 particles
- **32×32**: 1024 particles (slowest, highest detail)

## Performance

- **Recommended Resolution**: 12×12 for balanced performance and visual quality
- **High Performance**: 4×4 to 8×8 for older systems
- **High Quality**: 16×16 to 32×32 for modern GPUs
- **Target FPS**: 60 FPS with adaptive timestep limiting

##  Project Structure

```
cloth-drop-simulation/
│
├── cloth_drop_simulation.py    # Main simulation file
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── screenshots/                 # Project screenshots
│   ├── simulation_running.png
│   ├── wireframe_mode.png
│   └── different_resolutions.png
└── docs/                        # Additional documentation
    └── REPORT.md
```

##  Learning Outcomes

This project helped me understand:
1. How 3D transformations work in real-time graphics
2. Implementation of lighting and shading models
3. Physics simulation using numerical integration
4. Collision detection and response algorithms
5. OpenGL rendering pipeline and state management
6. Interactive 3D application development
7. Performance optimization in real-time graphics

##  Known Limitations

- Performance may drop with 32×32 resolution on older hardware
- Cloth may occasionally clip through sphere at very high speeds (rare)
- UI rendering uses glDrawPixels (compatibility mode)

##  Future Enhancements

- [ ] Add wind force controls
- [ ] Multiple sphere obstacles
- [ ] Cloth tearing simulation
- [ ] Export animation to video
- [ ] GPU-accelerated physics
- [ ] Self-collision detection

##  Development

### Built With
- **Python 3.13.5**: Core programming language
- **PyGame**: Window management and input handling
- **PyOpenGL**: OpenGL bindings for Python
- **NumPy**: Numerical computations and vector math

### Code Highlights
- Object-oriented design with Particle, Spring, Cloth, and Ball classes
- Efficient collision detection with spatial proximity checks
- Modular rendering pipeline
- Event-driven UI system

##  Academic Information

**Course**: COMP 342 - Computer Graphics  
**Institution**: Kathmandu University, Department of Computer Science and Engineering  
**Level**: B.Sc 3rd Year / 1st Semester  
**Project Type**: 3D Graphics Simulation


**Note**: Performance may vary based on system specifications.
