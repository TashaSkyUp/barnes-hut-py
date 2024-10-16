import pygame
import numpy as np
from simulation import Simulation

class Renderer:
    def __init__(self):
        pygame.init()
        self.width, self.height = 900, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('N-Body Simulation')
        self.clock = pygame.time.Clock()
        self.simulation = Simulation()
        self.scale = 3600.0
        self.view_pos = np.zeros(2, dtype=float)
        self.running = True
        self.paused = False

        self.dragging = False
        self.last_mouse_pos = None

    def world_to_screen(self, pos):
        screen_pos = ((pos - self.view_pos) / self.scale) * self.height / 2 + np.array([self.width / 2, self.height / 2], dtype=float)
        return screen_pos.astype(int)

    def screen_to_world(self, pos):
        world_pos = ((pos - np.array([self.width / 2, self.height / 2], dtype=float)) * self.scale * 2 / self.height) + self.view_pos
        return world_pos

    def run(self):
        while self.running:
            self.handle_events()
            if not self.paused:
                self.simulation.step()
            self.render()
            self.clock.tick(60)  # Limit to 60 FPS

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
                elif event.button == 4:  # Mouse wheel up
                    # Zoom in
                    mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
                    world_pos_before_zoom = self.screen_to_world(mouse_pos)
                    self.scale /= 1.1
                    world_pos_after_zoom = self.screen_to_world(mouse_pos)
                    self.view_pos += (world_pos_before_zoom - world_pos_after_zoom)
                elif event.button == 5:  # Mouse wheel down
                    # Zoom out
                    mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
                    world_pos_before_zoom = self.screen_to_world(mouse_pos)
                    self.scale *= 1.1
                    world_pos_after_zoom = self.screen_to_world(mouse_pos)
                    self.view_pos += (world_pos_before_zoom - world_pos_after_zoom)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                    self.last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
                    delta = mouse_pos - self.last_mouse_pos
                    self.view_pos -= (delta * self.scale * 2 / self.height)
                    self.last_mouse_pos = mouse_pos

    def render(self):
        self.screen.fill((0, 0, 0))
        for body in self.simulation.bodies:
            screen_pos = self.world_to_screen(body.pos)
            radius = int(body.radius * self.height / self.scale)
            if 0 <= screen_pos[0] < self.width and 0 <= screen_pos[1] < self.height:
                pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, max(radius, 1))
        pygame.display.flip()
