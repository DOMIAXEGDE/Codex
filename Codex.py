#!/usr/bin/env python3
"""
Event Sequencer - Modern Core
-----------------------------
Refactored engine with dynamic resolution, robust pointer logic,
and a modern HUD interface.

Author: Refactored by Gemini
Version: 2.0 (Sleek Edition)
Date: 2025-12-06
"""

import pygame
import os
import sys
import tkinter as tk
from tkinter import simpledialog, ttk
import json
import pickle
from datetime import datetime

# --- CONFIGURATION & AESTHETICS ---
CONFIG = {
    "WINDOW_WIDTH": 1280,
    "WINDOW_HEIGHT": 800,
    "GRID_ROWS": 16,
    "GRID_COLS": 16,
    "FPS": 60,
    "FONT_SIZE": 20,
    "SIDEBAR_WIDTH": 300,
}

# Cyberpunk / Dark Mode Palette
COLORS = {
    "BG": (20, 24, 28),           # Dark Slate
    "GRID_BG": (30, 34, 38),      # Slightly lighter
    "GRID_LINES": (50, 55, 60),
    "CELL_EMPTY": (40, 44, 48),
    "CELL_ACTIVE": (0, 120, 215), # Blue
    "CURSOR": (0, 255, 150),      # Neon Green
    "SELECTION": (255, 50, 80),   # Neon Red
    "TEXT": (220, 220, 220),
    "TEXT_DIM": (120, 120, 120),
    "ACCENT": (255, 180, 0)       # Amber
}

# --- DATA MANAGEMENT ---

class SessionManager:
    """Handles persistence of the matrix and history."""
    def __init__(self):
        self.filename = "Command_Matrix_Data.json"
        self.data = {
            "commands": {},  # Format: "index_id": "code"
            "history": [],
            "last_pos": [0, 0]
        }
        self.load()

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=4)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Load failed: {e}")

    def get_command(self, index):
        return self.data["commands"].get(str(index), "")

    def set_command(self, index, code):
        self.data["commands"][str(index)] = code
        self.save()

    def log_history(self, index):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.data["history"].append(f"[{timestamp}] Exec Cmd {index}")
        if len(self.data["history"]) > 20:
            self.data["history"].pop(0)

# --- UI COMPONENTS ---

class ModernUI:
    """Handles all drawing operations."""
    def __init__(self, surface):
        self.surface = surface
        self.font = pygame.font.Font(None, CONFIG["FONT_SIZE"])
        self.title_font = pygame.font.Font(None, 30)
        
        # Calculate Layout
        self.grid_area = pygame.Rect(
            20, 20, 
            CONFIG["WINDOW_WIDTH"] - CONFIG["SIDEBAR_WIDTH"] - 40, 
            CONFIG["WINDOW_HEIGHT"] - 40
        )
        
        # Determine Cell Size (Keep aspect ratio square)
        avail_w = self.grid_area.width
        avail_h = self.grid_area.height
        self.cell_size = min(avail_w // CONFIG["GRID_COLS"], avail_h // CONFIG["GRID_ROWS"])
        
        # Re-center grid exactly
        actual_w = self.cell_size * CONFIG["GRID_COLS"]
        actual_h = self.cell_size * CONFIG["GRID_ROWS"]
        self.grid_offset_x = self.grid_area.x + (avail_w - actual_w) // 2
        self.grid_offset_y = self.grid_area.y + (avail_h - actual_h) // 2

    def to_screen(self, grid_x, grid_y):
        """Converts Grid Index -> Screen Pixel."""
        sx = self.grid_offset_x + (grid_x * self.cell_size)
        sy = self.grid_offset_y + (grid_y * self.cell_size)
        return sx, sy

    def to_grid(self, screen_x, screen_y):
        """Converts Screen Pixel -> Grid Index."""
        rel_x = screen_x - self.grid_offset_x
        rel_y = screen_y - self.grid_offset_y
        
        gx = rel_x // self.cell_size
        gy = rel_y // self.cell_size
        
        if 0 <= gx < CONFIG["GRID_COLS"] and 0 <= gy < CONFIG["GRID_ROWS"]:
            return int(gx), int(gy)
        return None

    def draw_grid(self, session, cursor_pos, selection_pos=None):
        # Draw Background
        pygame.draw.rect(self.surface, COLORS["GRID_BG"], 
                        (self.grid_offset_x, self.grid_offset_y, 
                         self.cell_size * CONFIG["GRID_COLS"], 
                         self.cell_size * CONFIG["GRID_ROWS"]))

        # Draw Cells
        for y in range(CONFIG["GRID_ROWS"]):
            for x in range(CONFIG["GRID_COLS"]):
                sx, sy = self.to_screen(x, y)
                
                # Check if command exists
                # Logic: We treat the grid as one layer for simplicity in this view
                # Ideally, this maps to the Session structure
                global_idx = (y * CONFIG["GRID_COLS"]) + x
                has_cmd = str(global_idx) in session.data["commands"]
                
                color = COLORS["CELL_ACTIVE"] if has_cmd else COLORS["CELL_EMPTY"]
                
                # Draw Cell
                pygame.draw.rect(self.surface, color, 
                                (sx + 2, sy + 2, self.cell_size - 4, self.cell_size - 4))
                
                # Draw ID (Small)
                if self.cell_size > 30:
                    text = self.font.render(str(global_idx), True, (60, 60, 60))
                    self.surface.blit(text, (sx + 5, sy + 5))

        # Draw Selection (The "Anchor" or First Click)
        if selection_pos:
            sx, sy = self.to_screen(*selection_pos)
            pygame.draw.rect(self.surface, COLORS["SELECTION"], 
                            (sx, sy, self.cell_size, self.cell_size), 4)

        # Draw Cursor (The Player)
        cx, cy = self.to_screen(*cursor_pos)
        pygame.draw.rect(self.surface, COLORS["CURSOR"], 
                        (cx - 2, cy - 2, self.cell_size + 4, self.cell_size + 4), 2)
        
        # Crosshair effect
        pygame.draw.line(self.surface, COLORS["CURSOR"], (cx + self.cell_size//2, cy), (cx + self.cell_size//2, cy + self.cell_size), 1)
        pygame.draw.line(self.surface, COLORS["CURSOR"], (cx, cy + self.cell_size//2), (cx + self.cell_size, cy + self.cell_size//2), 1)

    def draw_sidebar(self, status_msg, mode, history):
        x = CONFIG["WINDOW_WIDTH"] - CONFIG["SIDEBAR_WIDTH"]
        y = 20
        
        # Panel BG
        pygame.draw.rect(self.surface, (25, 30, 35), (x, 0, CONFIG["SIDEBAR_WIDTH"], CONFIG["WINDOW_HEIGHT"]))
        pygame.draw.line(self.surface, COLORS["ACCENT"], (x, 0), (x, CONFIG["WINDOW_HEIGHT"]), 2)

        # Header
        title = self.title_font.render("COMMAND MATRIX", True, COLORS["ACCENT"])
        self.surface.blit(title, (x + 20, y))
        y += 50

        # Mode Indicator
        mode_lbl = self.font.render(f"INPUT: {mode}", True, COLORS["TEXT"])
        self.surface.blit(mode_lbl, (x + 20, y))
        y += 40
        
        # Status
        status_lbl = self.font.render("STATUS:", True, COLORS["TEXT_DIM"])
        self.surface.blit(status_lbl, (x + 20, y))
        y += 25
        
        # Wrap status text
        words = status_msg.split(' ')
        line = ""
        for word in words:
            test_line = line + word + " "
            if self.font.size(test_line)[0] < CONFIG["SIDEBAR_WIDTH"] - 40:
                line = test_line
            else:
                self.surface.blit(self.font.render(line, True, COLORS["CURSOR"]), (x + 20, y))
                y += 20
                line = word + " "
        self.surface.blit(self.font.render(line, True, COLORS["CURSOR"]), (x + 20, y))
        y += 50

        # History
        hist_lbl = self.font.render("EXECUTION LOG:", True, COLORS["TEXT_DIM"])
        self.surface.blit(hist_lbl, (x + 20, y))
        y += 25
        
        for log_entry in reversed(history[-10:]):
            entry_surf = self.font.render(log_entry, True, (150, 150, 150))
            self.surface.blit(entry_surf, (x + 20, y))
            y += 20

# --- CORE ENGINE ---

class Engine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]))
        pygame.display.set_caption("Codex // Event Sequencer 2.0")
        self.clock = pygame.time.Clock()
        
        self.session = SessionManager()
        self.ui = ModernUI(self.screen)
        
        # State
        self.running = True
        self.cursor = [0, 0] # Grid X, Grid Y
        self.selection_anchor = None # For 2-step selection
        self.mode = "KEYBOARD" # KEYBOARD, MOUSE
        self.status = "Ready. Press Enter to Select."
        
        # Tkinter hidden root for dialogs
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # --- MOUSE INPUT ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mode = "MOUSE"
                mx, my = pygame.mouse.get_pos()
                grid_pos = self.ui.to_grid(mx, my)
                
                if grid_pos:
                    self.cursor = list(grid_pos)
                    self.process_activation()

            elif event.type == pygame.MOUSEMOTION:
                if self.mode == "MOUSE":
                    mx, my = pygame.mouse.get_pos()
                    grid_pos = self.ui.to_grid(mx, my)
                    if grid_pos:
                        self.cursor = list(grid_pos)

            # --- KEYBOARD INPUT ---
            elif event.type == pygame.KEYDOWN:
                self.mode = "KEYBOARD"
                
                if event.key == pygame.K_LEFT:
                    self.cursor[0] = max(0, self.cursor[0] - 1)
                elif event.key == pygame.K_RIGHT:
                    self.cursor[0] = min(CONFIG["GRID_COLS"] - 1, self.cursor[0] + 1)
                elif event.key == pygame.K_UP:
                    self.cursor[1] = max(0, self.cursor[1] - 1)
                elif event.key == pygame.K_DOWN:
                    self.cursor[1] = min(CONFIG["GRID_ROWS"] - 1, self.cursor[1] + 1)
                
                elif event.key == pygame.K_RETURN:
                    self.process_activation()
                
                elif event.key == pygame.K_ESCAPE:
                    if self.selection_anchor:
                        self.selection_anchor = None
                        self.status = "Selection Cancelled."
                    else:
                        self.status = "Press Ctrl+Q to Quit"

                elif event.key == pygame.K_q and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.running = False

    def process_activation(self):
        """Unified logic for handling selection (2-step process)."""
        current_idx = (self.cursor[1] * CONFIG["GRID_COLS"]) + self.cursor[0]
        
        if self.selection_anchor is None:
            # Step 1: Set Anchor (Index)
            self.selection_anchor = tuple(self.cursor)
            self.status = f"Index {self.cursor} Selected. Select Command relative to this."
        else:
            # Step 2: Determine Command ID based on relative offset
            # (Simplification for this demo: We use absolute ID calculation 
            # based on Index * 256 + Offset, similar to original logic)
            
            # Calculate logic as per original Codex requirements
            anchor_id = (self.selection_anchor[1] * CONFIG["GRID_COLS"]) + self.selection_anchor[0]
            
            # Relative selection in a virtual 16x16 subgrid
            rel_x = abs(self.cursor[0] - self.selection_anchor[0])
            rel_y = abs(self.cursor[1] - self.selection_anchor[1])
            sub_id = (rel_y * 16) + rel_x
            
            final_cmd_id = (anchor_id * 256) + sub_id
            
            self.open_action_menu(final_cmd_id)
            self.selection_anchor = None # Reset

    def open_action_menu(self, cmd_id):
        """Opens the Tkinter dialog to Edit or Execute."""
        self.status = f"Querying Command ID: {cmd_id}..."
        self.draw() # Force a redraw so status updates visually before freezing for Tkinter
        
        existing_code = self.session.get_command(cmd_id)
        
        choice = simpledialog.askinteger(
            "Command Action", 
            f"Command ID: {cmd_id}\n\n1. Edit Code\n2. Execute Code\n3. Clear Slot",
            parent=self.tk_root,
            minvalue=1, maxvalue=3
        )
        
        if choice == 1:
            self.open_editor(cmd_id, existing_code)
        elif choice == 2:
            self.execute_command(cmd_id, existing_code)
        elif choice == 3:
            self.session.set_command(cmd_id, "")
            self.status = f"Command {cmd_id} cleared."

    def open_editor(self, cmd_id, code):
        """Opens a simple Tkinter Text editor."""
        editor = tk.Toplevel(self.tk_root)
        editor.title(f"Editing Command {cmd_id}")
        editor.geometry("600x400")
        
        text_area = tk.Text(editor, bg="#202020", fg="#00FF00", insertbackground="white")
        text_area.pack(fill=tk.BOTH, expand=True)
        text_area.insert(tk.END, code)
        
        def save():
            new_code = text_area.get("1.0", tk.END).strip()
            self.session.set_command(cmd_id, new_code)
            self.status = f"Command {cmd_id} saved."
            editor.destroy()
            
        btn = ttk.Button(editor, text="SAVE", command=save)
        btn.pack(fill=tk.X)
        
        self.tk_root.wait_window(editor)

    def execute_command(self, cmd_id, code):
        if not code:
            self.status = f"Command {cmd_id} is empty."
            return
            
        try:
            # SAFETY: In a real app, sanitize this or use a restricted scope
            exec_globals = {}
            exec(code, exec_globals)
            self.session.log_history(cmd_id)
            self.status = f"Command {cmd_id} executed successfully."
        except Exception as e:
            self.status = f"Execution Error: {str(e)}"

    def draw(self):
        self.screen.fill(COLORS["BG"])
        
        self.ui.draw_grid(self.session, self.cursor, self.selection_anchor)
        self.ui.draw_sidebar(self.status, self.mode, self.session.data["history"])
        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input()
            self.draw()
            self.clock.tick(CONFIG["FPS"])
        
        self.session.save()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = Engine()
    app.run()