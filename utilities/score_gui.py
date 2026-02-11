import tkinter as tk

class ScoreGui:
    def __init__(self, root, score_min=1, score_max=324):
        self.root = root
        self.app_height = 480
        self.app_width = 800
        self.score_min = score_min
        self.score_max = score_max
        self.score = 0
        
        # --- Create Screen Layout --- #
        # Full-Screen Setup
        self.root.overrideredirect(True)  # Removes the title bar and borders
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")  # Forces screen resolution
        self.root.config(cursor="none")  # Hide the cursor
        self.root.bind('<Escape>', lambda event: self.exit_fullscreen())
        self.root.title("Coherence Visualizer")
        self.root.configure(bg="#111111")
        # Create the canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.root.winfo_screenwidth(),  # Match canvas to screen width
            height=self.root.winfo_screenheight() - (self.root.winfo_screenheight() * 0.2),  # Match height
            bg="#111111",
            highlightthickness=0
        )
        self.canvas.pack(side=tk.TOP, expand=True)
        self.root.oval_radius = 200
        self.canvas.create_oval(
            (self.root.winfo_screenwidth() - self.root.oval_radius) * 0.5,
            (self.root.winfo_screenheight() - self.root.oval_radius) * 0.5,
            (self.root.winfo_screenwidth() - self.root.oval_radius) * 0.5 + self.root.oval_radius,
            (self.root.winfo_screenheight() - self.root.oval_radius) * 0.5 + self.root.oval_radius,
            fill="white", 
            outline="white"
        )
        # Add numerical display
        self.score_label = tk.Label(
            self.root, 
            text="0", 
            font=("Piboto Condensed Bold", 48),
            bg="white"
        )
        self.score_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- Add Controls --- #
        # Start button
        self.start_button = tk.Button(
            self.root, 
            text="Start", 
            font=("Piboto", 24), 
            command=self.toggle_process,
            fg="white", 
            bg="#121212",
            padx=30,
            pady=20
        )
        self.start_button.pack(side=tk.RIGHT, expand=False, pady=24, padx=24)
        # Quit button
        self.quit_button = tk.Button(
            self.root, 
            text="Quit", 
            font=("Piboto", 24), 
            command=self.close,
            fg="white", 
            bg="#121212",
            padx=30,
            pady=20
        )
        self.quit_button.pack(side=tk.LEFT, expand=False, pady=24, padx=24)

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)
        self.root.geometry(f"{self.app_width}x{self.app_height}")

    def scale_score(self, score):
        adj = 0 - self.score_min # Calculate different from 0 and min
        score_max = self.score_max + adj # Add adjustment to max score
        scale_factor = 100 / score_max # Create ratio to convert score to a 100 point scale
        adjusted_score = adj + score # adjust to ensure min possible score is 0
        scaled_score = adjusted_score * scale_factor # Scale the adjusted score
        return int(scaled_score) # Return scaled score as an integer

    def score_to_color(self, score):
        if score >= 40:
            t = (score - 40) / (100 - 40)
            r = int(247 + t * (247 - 247))
            g = int(227 + t * (77 - 227))
            b = int(118 + t * (32 - 118))
        else:
            t = score / 40
            r = int(210 + t * (247 - 210))
            g = int(210 + t * (227 - 210))
            b = int(208 + t * (118 - 208))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def update_score(self, score):
        self.score = score
        scaled_sc = self.scale_score(self.score)
        self.score_label.config(text=f"{scaled_sc}")
        color = self.score_to_color(scaled_sc)
        self.root.configure(bg=color)
        self.canvas.configure(bg=color)

    def toggle_process(self):
        if self.running:
            self.running = False
            self.button.config(text="Start")
        else:
            self.running = True
            self.button.config(text="Stop")

    def close(self):
        self.root.destroy()