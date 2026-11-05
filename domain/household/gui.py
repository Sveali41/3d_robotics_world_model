"""Live PARTNR robot view with semantic skill commands, using Tk and Pillow."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .cli import parse_action


class HouseholdWindow:
    def __init__(self, env):
        self.env = env
        self.root = tk.Tk()
        self.root.title(f"PARTNR Household — {env.robot.title()} — Skill interaction")
        self.closed = False
        self.busy = False
        self.exit_code = 0
        self.photo = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        state = env.state()
        self.status = tk.StringVar(value="Ready — enter a skill command below")
        ttk.Label(self.root, text=f"Scene {state['scene_id']} · Episode {state['episode_id']}").pack(pady=6)
        self.image = ttk.Label(self.root)
        self.image.pack(padx=12)
        ttk.Label(self.root, textvariable=self.status, wraplength=730).pack(padx=12, pady=8)
        ttk.Label(self.root, text="Navigate <entity>   |   Pick <object>   |   Place <object> <furniture> on"
                  "   |   Open / Close <furniture>").pack(padx=12, pady=4)
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=8)
        objects = [node["name"] for node in state["nodes"] if node["kind"] == "Object"]
        self.command = tk.StringVar(value=f"Navigate {objects[0]}" if objects else "")
        self.entry = ttk.Entry(controls, textvariable=self.command, width=60)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _: self.execute())
        self.execute_button = ttk.Button(controls, text="Execute", command=self.execute)
        self.execute_button.pack(side="left", padx=6)
        self.reset_button = ttk.Button(controls, text="Reset", command=self.reset)
        self.reset_button.pack(side="left")
        targets = ", ".join(node["name"] for node in state["nodes"]
                            if node["kind"] in {"Object", "Furniture"})
        ttk.Label(self.root, text="Targets: " + targets, wraplength=730).pack(padx=12, pady=6)
        self.env.frame_callback = self.render_frame
        self.render_frame(env.interface.get_observations(), {})
        self.entry.focus_set()

    def render_frame(self, observations, actions):
        if self.closed:
            raise InterruptedError("Interactive window closed during skill execution")
        # Display the existing robot third-person sensor; no extra simulator step.
        frame = observations["agent_0_third_rgb"][0].detach().cpu().numpy()
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((900, 600))
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.image.configure(image=self.photo)
        self.root.update_idletasks()
        self.root.update()
        if self.closed:
            raise InterruptedError("Interactive window closed during skill execution")

    def set_busy(self, busy):
        self.busy = busy
        for widget in (self.entry, self.execute_button, self.reset_button):
            widget.configure(state="disabled" if busy else "normal")

    def execute(self, action=None):
        if self.busy or self.closed:
            return None
        try:
            if action is None:
                action = parse_action(self.command.get())
        except ValueError as error:
            self.status.set(str(error))
            return None
        self.set_busy(True)
        self.status.set(f"Executing {action['skill']} {action['target']}…")
        self.root.update_idletasks()
        try:
            result = self.env.execute_skill(**action)
            outcome = "Succeeded" if result["success"] else "Failed"
            self.status.set(f"{outcome}: {result['response'] or 'Skill timed out'}"
                            f" · {result['control_steps']} control steps")
            return result
        except InterruptedError:
            if not self.closed:
                raise
        except Exception as error:
            self.status.set(f"Error: {error}. Reset before trying another skill.")
        finally:
            if not self.closed:
                self.set_busy(False)
                self.entry.focus_set()

    def reset(self):
        if self.busy or self.closed:
            return
        self.set_busy(True)
        self.status.set("Resetting episode…")
        self.root.update_idletasks()
        try:
            self.env.reset()
            self.render_frame(self.env.interface.get_observations(), {})
            self.status.set("Episode reset — ready")
        except InterruptedError:
            if not self.closed:
                raise
        except Exception as error:
            self.status.set(f"Reset failed: {error}")
        finally:
            if not self.closed:
                self.set_busy(False)

    def close(self):
        self.closed = True
        self.root.destroy()

    def run(self, commands=None):
        if commands is not None:
            def run_commands():
                for action in commands:
                    if self.closed:
                        return
                    result = self.execute(action)
                    if result is None or not result["success"]:
                        self.exit_code = 2
                        break
            self.root.after(100, run_commands)
        try:
            self.root.mainloop()
        finally:
            self.env.frame_callback = None
        return self.exit_code
