"""
OpenGL Cloth and Ball Physics Simulation
Computer Graphics Project

Demonstrates:
- 3D Mesh Rendering (Cloth)
- Physics Simulation (Spring-Mass System)
- Collision Detection with Wrapping
- Lighting and Shading
- Transformations and Projections
- Real-time Animation
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

class Particle:
    """Represents a single particle in the cloth mesh"""
    def __init__(self, x, y, z, mass=1.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.old_pos = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.force = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass = mass
        self.pinned = False
        self.normal = np.array([0.0, 0.0, 1.0], dtype=float)
    
    def apply_force(self, f):
        """Add force to the particle"""
        if not self.pinned:
            self.force += f
    
    def update(self, dt):
        """Update particle position using Verlet integration"""
        if self.pinned:
            return
        
        # Verlet integration
        acceleration = self.force / self.mass
        new_pos = 2 * self.pos - self.old_pos + acceleration * dt * dt
        self.velocity = (new_pos - self.pos) / dt
        self.old_pos = self.pos.copy()
        self.pos = new_pos
        
        # Reset force
        self.force = np.array([0.0, 0.0, 0.0], dtype=float)
    
    def add_damping(self, damping=0.99):
        """Add velocity damping"""
        if not self.pinned:
            self.velocity *= damping

class Spring:
    """Spring constraint between two particles"""
    def __init__(self, p1, p2, stiffness=50.0):
        self.p1 = p1
        self.p2 = p2
        self.rest_length = np.linalg.norm(p1.pos - p2.pos)
        self.stiffness = stiffness
    
    def satisfy_constraint(self):
        """Apply spring force to maintain rest length - RIGID behavior"""
        delta = self.p2.pos - self.p1.pos
        current_length = np.linalg.norm(delta)
        
        if current_length == 0:
            return
        
        # Calculate correction needed to restore rest length
        correction = (current_length - self.rest_length) / current_length
        # Use full correction for rigid behavior (was 0.5)
        correction_vector = delta * correction * 0.8  # Stronger correction for rigidity
        
        if not self.p1.pinned:
            self.p1.pos += correction_vector
        if not self.p2.pinned:
            self.p2.pos -= correction_vector

class Cloth:
    """Cloth mesh using spring-mass system"""
    def __init__(self, width, height, resolution, start_pos=(0, 8, 0)):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.start_pos = start_pos
        
        # Create particle grid
        self.particles = []
        self.springs = []
        
        self.create_mesh()
        self.create_springs()
    
    def create_mesh(self):
        """Create grid of particles"""
        step_x = self.width / (self.resolution - 1)
        step_y = self.height / (self.resolution - 1)
        
        for i in range(self.resolution):
            row = []
            for j in range(self.resolution):
                x = self.start_pos[0] - self.width/2 + j * step_x
                y = self.start_pos[1]
                z = self.start_pos[2] - self.height/2 + i * step_y
                
                particle = Particle(x, y, z, mass=0.1)
                # No pinned particles - cloth freely falls
                
                row.append(particle)
            self.particles.append(row)
    
    def create_springs(self):
        """Create spring constraints between particles"""
        # Structural springs (horizontal and vertical)
        for i in range(self.resolution):
            for j in range(self.resolution):
                if j < self.resolution - 1:
                    spring = Spring(self.particles[i][j], self.particles[i][j+1])
                    self.springs.append(spring)
                
                if i < self.resolution - 1:
                    spring = Spring(self.particles[i][j], self.particles[i+1][j])
                    self.springs.append(spring)
        
        # Shear springs (diagonals)
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                spring1 = Spring(self.particles[i][j], self.particles[i+1][j+1])
                spring2 = Spring(self.particles[i][j+1], self.particles[i+1][j])
                self.springs.append(spring1)
                self.springs.append(spring2)
        
        # Bend springs (skip one particle)
        for i in range(self.resolution):
            for j in range(self.resolution):
                if j < self.resolution - 2:
                    spring = Spring(self.particles[i][j], self.particles[i][j+2], stiffness=30.0)
                    self.springs.append(spring)
                
                if i < self.resolution - 2:
                    spring = Spring(self.particles[i][j], self.particles[i+2][j], stiffness=30.0)
                    self.springs.append(spring)
    
    def apply_gravity(self, gravity=(0, -9.8, 0)):
        """Apply gravitational force to all particles"""
        g = np.array(gravity, dtype=float)
        for row in self.particles:
            for particle in row:
                particle.apply_force(g * particle.mass)
    
    def apply_wind(self, wind_force=(0, 0, 0)):
        """Apply wind force"""
        w = np.array(wind_force, dtype=float)
        for row in self.particles:
            for particle in row:
                particle.apply_force(w)
    
    def update(self, dt, constraint_iterations=4):
        """Update cloth simulation"""
        # Apply forces
        self.apply_gravity()
        
        # Update particles
        for row in self.particles:
            for particle in row:
                particle.update(dt)
                particle.add_damping(0.99)
        
        # Satisfy constraints (more iterations for rigid behavior)
        for _ in range(constraint_iterations):
            for spring in self.springs:
                spring.satisfy_constraint()
    
    def compute_normals(self):
        """Compute vertex normals for lighting"""
        # Reset normals
        for row in self.particles:
            for particle in row:
                particle.normal = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Compute face normals and accumulate
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                p1 = self.particles[i][j]
                p2 = self.particles[i][j+1]
                p3 = self.particles[i+1][j]
                p4 = self.particles[i+1][j+1]
                
                # Triangle 1
                v1 = p2.pos - p1.pos
                v2 = p3.pos - p1.pos
                normal1 = np.cross(v1, v2)
                if np.linalg.norm(normal1) > 0:
                    normal1 = normal1 / np.linalg.norm(normal1)
                
                # Triangle 2
                v3 = p3.pos - p2.pos
                v4 = p4.pos - p2.pos
                normal2 = np.cross(v3, v4)
                if np.linalg.norm(normal2) > 0:
                    normal2 = normal2 / np.linalg.norm(normal2)
                
                # Accumulate normals
                p1.normal += normal1
                p2.normal += normal1 + normal2
                p3.normal += normal1 + normal2
                p4.normal += normal2
        
        # Normalize
        for row in self.particles:
            for particle in row:
                norm = np.linalg.norm(particle.normal)
                if norm > 0:
                    particle.normal /= norm
    
    def collide_with_sphere(self, center, radius):
        """Handle collision with sphere - cloth wraps around it WITHOUT penetration"""
        center = np.array(center, dtype=float)
        for row in self.particles:
            for particle in row:
                if particle.pinned:
                    continue
                
                delta = particle.pos - center
                distance = np.linalg.norm(delta)
                
                # Check if particle is inside or touching the sphere
                # Add small offset to ensure NO penetration
                collision_radius = radius + 0.02  # Small buffer to prevent penetration
                
                if distance < collision_radius:
                    # Push particle OUT to sphere surface immediately (RIGID)
                    if distance > 0.001:
                        # Calculate normal direction from sphere center
                        normal = delta / distance
                        # Place particle EXACTLY on sphere surface (rigid collision)
                        particle.pos = center + normal * collision_radius
                        # Kill velocity component going into sphere (rigid response)
                        velocity_normal = np.dot(particle.velocity, normal)
                        if velocity_normal < 0:  # Moving into sphere
                            particle.velocity -= velocity_normal * normal * 1.5  # Bounce back
                        # Apply strong friction
                        particle.velocity *= 0.5
                    else:
                        # If particle is exactly at center, push it out in random direction
                        random_dir = np.random.randn(3)
                        random_dir = random_dir / np.linalg.norm(random_dir)
                        particle.pos = center + random_dir * collision_radius
                        particle.velocity *= 0.1
    
    def render(self, wireframe=False):
        """Render the cloth mesh"""
        self.compute_normals()
        
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            # Black lines for wireframe on white cloth
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.5)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_QUADS)
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                p1 = self.particles[i][j]
                p2 = self.particles[i][j+1]
                p3 = self.particles[i+1][j+1]
                p4 = self.particles[i+1][j]
                
                if not wireframe:
                    # White cloth
                    glColor3f(1.0, 1.0, 1.0)
                
                glNormal3fv(p1.normal)
                glVertex3fv(p1.pos)
                glNormal3fv(p2.normal)
                glVertex3fv(p2.pos)
                glNormal3fv(p3.normal)
                glVertex3fv(p3.pos)
                glNormal3fv(p4.normal)
                glVertex3fv(p4.pos)
        glEnd()
        
        if wireframe:
            glEnable(GL_LIGHTING)
            glLineWidth(1.0)
        
        # Draw grid lines on cloth (black lines on white cloth)
        if not wireframe:
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            
            # Draw horizontal lines
            glBegin(GL_LINES)
            for i in range(self.resolution):
                for j in range(self.resolution - 1):
                    p1 = self.particles[i][j]
                    p2 = self.particles[i][j+1]
                    glVertex3fv(p1.pos)
                    glVertex3fv(p2.pos)
            glEnd()
            
            # Draw vertical lines
            glBegin(GL_LINES)
            for i in range(self.resolution - 1):
                for j in range(self.resolution):
                    p1 = self.particles[i][j]
                    p2 = self.particles[i+1][j]
                    glVertex3fv(p1.pos)
                    glVertex3fv(p2.pos)
            glEnd()
            
            glEnable(GL_LIGHTING)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

class Ball:
    """Sphere object for collision"""
    def __init__(self, position, radius, color=(1.0, 0.3, 0.3)):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.color = color
    
    def render(self):
        """Render the ball using GLU sphere"""
        glPushMatrix()
        glTranslatef(*self.position)
        glColor3f(*self.color)
        
        # Create quadric object
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, self.radius, 32, 32)
        gluDeleteQuadric(quad)
        
        glPopMatrix()

class Button:
    """Simple 2D button for UI"""
    def __init__(self, x, y, width, height, text, value):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.value = value
        self.hovered = False
        self.active = False
    
    def check_hover(self, mouse_pos):
        """Check if mouse is hovering over button"""
        self.hovered = self.rect.collidepoint(mouse_pos)
        return self.hovered
    
    def check_click(self, mouse_pos):
        """Check if button is clicked"""
        return self.rect.collidepoint(mouse_pos)
    
    def draw(self, screen, font):
        """Draw button on screen"""
        # Button background
        if self.active:
            color = (100, 200, 100)
        elif self.hovered:
            color = (80, 80, 120)
        else:
            color = (50, 50, 80)
        
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)
        
        # Button text
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class ClothSimulation:
    """Main simulation class"""
    def __init__(self, cloth_resolution=12):
        # Initialize Pygame
        pygame.init()
        self.display = (1200, 800)
        # Create window with both OpenGL and regular pygame capabilities
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption('Cloth Drop Simulation - Interactive')
        
        # Create a separate 2D surface for UI overlay
        self.ui_surface = pygame.Surface(self.display, pygame.SRCALPHA)
        
        # Setup OpenGL
        self.setup_opengl()
        
        # Create objects
        self.cloth = Cloth(width=4, height=4, resolution=cloth_resolution, start_pos=(0, 8, 0))
        self.ball = Ball(position=(0, 0, 0), radius=1.2)
        
        # Simulation parameters
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.wireframe = False
        self.rotation = [20, 30]
        self.zoom = -12
        
        # Camera
        self.mouse_down = False
        self.last_mouse_pos = None
        self.ui_active = False  # Track if mouse is over UI
        
        # Create resolution buttons
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.buttons = []
        button_width = 70
        button_height = 40
        button_spacing = 10
        start_x = 20
        start_y = 20
        
        # 8 resolution options in two rows
        resolutions = [
            (4, "4x4"), (6, "6x6"), (8, "8x8"), (10, "10x10"),
            (12, "12x12"), (14, "14x14"), (16, "16x16"), (32, "32x32")
        ]
        
        for i, (res, label) in enumerate(resolutions):
            # Calculate position - 4 buttons per row
            row = i // 4
            col = i % 4
            button = Button(
                start_x + col * (button_width + button_spacing),
                start_y + row * (button_height + button_spacing),
                button_width,
                button_height,
                label,
                res
            )
            if res == cloth_resolution:
                button.active = True
            self.buttons.append(button)
        
        # Add Reset button below the resolution buttons
        self.reset_button = Button(
            start_x,
            start_y + 2 * (button_height + button_spacing) + 10,
            180,
            button_height,
            "RESET",
            None
        )
    
    def setup_opengl(self):
        """Configure OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Lighting
        glLight(GL_LIGHT0, GL_POSITION, (5, 10, 5, 1))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        glLight(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
        
        # Material
        glMaterial(GL_FRONT, GL_SPECULAR, (1, 1, 1, 1))
        glMaterial(GL_FRONT, GL_SHININESS, 50)
        
        # Projection
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
    
    def handle_events(self):
        """Handle user input"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Check if mouse is over any UI element
        self.ui_active = False
        for button in self.buttons:
            if button.check_hover(mouse_pos):
                self.ui_active = True
        if self.reset_button.check_hover(mouse_pos):
            self.ui_active = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_w:
                    self.wireframe = not self.wireframe
                elif event.key == pygame.K_r:
                    # Reset cloth
                    self.reset_cloth()
                elif event.key == pygame.K_1:
                    self.change_resolution(4)
                elif event.key == pygame.K_2:
                    self.change_resolution(6)
                elif event.key == pygame.K_3:
                    self.change_resolution(8)
                elif event.key == pygame.K_4:
                    self.change_resolution(10)
                elif event.key == pygame.K_5:
                    self.change_resolution(12)
                elif event.key == pygame.K_6:
                    self.change_resolution(14)
                elif event.key == pygame.K_7:
                    self.change_resolution(16)
                elif event.key == pygame.K_8:
                    self.change_resolution(32)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check button clicks
                    clicked_button = False
                    for button in self.buttons:
                        if button.check_click(mouse_pos):
                            self.change_resolution(button.value)
                            clicked_button = True
                            break
                    
                    if self.reset_button.check_click(mouse_pos):
                        self.reset_cloth()
                        clicked_button = True
                    
                    # Only enable camera rotation if no button was clicked
                    if not clicked_button and not self.ui_active:
                        self.mouse_down = True
                        self.last_mouse_pos = mouse_pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down and not self.ui_active:
                    if self.last_mouse_pos:
                        dx = mouse_pos[0] - self.last_mouse_pos[0]
                        dy = mouse_pos[1] - self.last_mouse_pos[1]
                        self.rotation[0] += dy * 0.5
                        self.rotation[1] += dx * 0.5
                    self.last_mouse_pos = mouse_pos
            
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom += event.y * 0.5
                self.zoom = max(-30, min(-5, self.zoom))
    
    def change_resolution(self, new_resolution):
        """Change cloth resolution"""
        self.cloth = Cloth(width=4, height=4, resolution=new_resolution, start_pos=(0, 8, 0))
        for button in self.buttons:
            button.active = (button.value == new_resolution)
    
    def reset_cloth(self):
        """Reset cloth to starting position"""
        current_res = self.cloth.resolution
        self.cloth = Cloth(width=4, height=4, resolution=current_res, start_pos=(0, 8, 0))
    
    def render(self):
        """Render the scene"""
        # Clear and setup 3D scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera positioning
        glTranslatef(0, -1, self.zoom)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        
        # Draw grid floor
        self.draw_grid()
        
        # Render ball
        self.ball.render()
        
        # Render cloth
        self.cloth.render(wireframe=self.wireframe)
        
        # Switch to 2D mode for UI
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Render UI elements using OpenGL
        self.render_ui_opengl()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        pygame.display.flip()
    
    def draw_grid(self):
        """Draw ground grid"""
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i, -2, -10)
            glVertex3f(i, -2, 10)
            glVertex3f(-10, -2, i)
            glVertex3f(10, -2, i)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def render_ui_opengl(self):
        """Render 2D UI elements using OpenGL"""
        # Draw semi-transparent background for button area
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Background panel (adjusted for 2 rows of buttons + reset button)
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(360, 10)
        glVertex2f(360, 160)
        glVertex2f(10, 160)
        glEnd()
        
        # Draw buttons
        for button in self.buttons:
            self.draw_button_opengl(button)
        self.draw_button_opengl(self.reset_button)
        
        # Draw text overlays using pygame
        self.draw_text_overlay()
        
        glDisable(GL_BLEND)
    
    def draw_button_opengl(self, button):
        """Draw a button using OpenGL"""
        # Determine color
        if button.active:
            color = (0.4, 0.8, 0.4, 0.9)
        elif button.hovered:
            color = (0.35, 0.35, 0.5, 0.9)
        else:
            color = (0.2, 0.2, 0.35, 0.9)
        
        # Draw filled rectangle
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(button.rect.left, button.rect.top)
        glVertex2f(button.rect.right, button.rect.top)
        glVertex2f(button.rect.right, button.rect.bottom)
        glVertex2f(button.rect.left, button.rect.bottom)
        glEnd()
        
        # Draw border
        glColor4f(0.8, 0.8, 0.8, 1.0)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(button.rect.left, button.rect.top)
        glVertex2f(button.rect.right, button.rect.top)
        glVertex2f(button.rect.right, button.rect.bottom)
        glVertex2f(button.rect.left, button.rect.bottom)
        glEnd()
    
    def draw_text_overlay(self):
        """Draw text using pygame and blit to OpenGL"""
        # Create temporary surface for text
        text_surface = pygame.Surface(self.display, pygame.SRCALPHA)
        text_surface.fill((0, 0, 0, 0))
        
        # Draw button labels with white text
        button_font = pygame.font.Font(None, 30)
        for button in self.buttons:
            # Draw white text for button labels
            text_surf = button_font.render(button.text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=button.rect.center)
            # Add black shadow for better visibility
            shadow_surf = button_font.render(button.text, True, (0, 0, 0))
            shadow_rect = shadow_surf.get_rect(center=(button.rect.centerx + 1, button.rect.centery + 1))
            text_surface.blit(shadow_surf, shadow_rect)
            text_surface.blit(text_surf, text_rect)
        
        # Draw reset button label
        reset_font = pygame.font.Font(None, 32)
        text_surf = reset_font.render(self.reset_button.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.reset_button.rect.center)
        shadow_surf = reset_font.render(self.reset_button.text, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(self.reset_button.rect.centerx + 1, self.reset_button.rect.centery + 1))
        text_surface.blit(shadow_surf, shadow_rect)
        text_surface.blit(text_surf, text_rect)
        
        # Draw info text
        info_texts = [
            f"Particles: {self.cloth.resolution * self.cloth.resolution}",
            f"Springs: {len(self.cloth.springs)}",
            "Keys: SPACE=Pause | W=Wireframe | R=Reset | 1-8=Resolution"
        ]
        
        y_offset = self.display[1] - 75
        for text in info_texts:
            # Shadow
            shadow = self.small_font.render(text, True, (0, 0, 0))
            text_surface.blit(shadow, (21, y_offset + 1))
            # Text
            surface = self.small_font.render(text, True, (255, 255, 255))
            text_surface.blit(surface, (20, y_offset))
            y_offset += 23
        
        # Status
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 100, 100) if self.paused else (100, 255, 100)
        status_font = pygame.font.Font(None, 36)
        shadow = status_font.render(status, True, (0, 0, 0))
        text_surface.blit(shadow, (self.display[0] - 121, 21))
        surface = status_font.render(status, True, status_color)
        text_surface.blit(surface, (self.display[0] - 120, 20))
        
        # Convert to OpenGL texture and draw
        texture_data = pygame.image.tostring(text_surface, "RGBA", True)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glRasterPos2f(0, 0)
        glDrawPixels(self.display[0], self.display[1], GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glDisable(GL_BLEND)
    
    def update(self, dt):
        """Update simulation"""
        if not self.paused:
            # More substeps for rigid collision handling
            substeps = 5
            sub_dt = dt / substeps
            
            for _ in range(substeps):
                self.cloth.update(sub_dt)
                # Apply collision AFTER each physics step for rigid response
                self.cloth.collide_with_sphere(self.ball.position, self.ball.radius)
    
    def run(self):
        """Main simulation loop"""
        while self.running:
            # Cap at 60 FPS, use fixed timestep for stability
            dt = min(self.clock.tick(60) / 1000.0, 0.02)  # Max 0.02s to prevent huge jumps
            
            self.handle_events()
            self.update(dt)
            self.render()
        
        pygame.quit()

def main():
    """Main entry point"""
    print("=" * 60)
    print("CLOTH DROP SIMULATION")
    print("Computer Graphics Project")
    print("=" * 60)
    print("\nStarting simulation with smooth cloth physics...")
    print("\nControls:")
    print("- Resolution Buttons: Click buttons for different resolutions")
    print("  4x4, 6x6, 8x8, 10x10, 12x12, 14x14, 16x16, 32x32")
    print("- Keyboard: Press 1-8 for quick resolution change")
    print("- RESET: Click RESET button or press R to drop cloth again")
    print("- SPACE: Pause/Resume simulation")
    print("- W: Toggle Wireframe mode")
    print("- Mouse Drag: Rotate camera view")
    print("- Mouse Scroll: Zoom in/out")
    print("- ESC: Exit simulation")
    print("\nWatch the white cloth with black grid drop and wrap!")
    print("Higher resolutions = smoother, more realistic wrapping")
    print("=" * 60)
    
    sim = ClothSimulation(cloth_resolution=12)
    sim.run()

if __name__ == "__main__":
    main()