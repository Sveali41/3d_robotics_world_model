"""Live ego/map views and upstream iTHOR actions; no forced interactions."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


class ThorWindow:
    def __init__(self, env):
        self.env = env
        self.root = tk.Tk()
        self.root.title("AI2-THOR / ProcTHOR — Kitchen interaction")
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.photos = []
        views = ttk.Frame(self.root)
        views.pack(padx=8, pady=8)
        self.ego = ttk.Label(views)
        self.ego.pack(side="left")
        self.map = ttk.Label(views)
        self.map.pack(side="left", padx=8)
        self.status = tk.StringVar(value="Ready. Select a target or click an object in the left image.")
        ttk.Label(self.root, textvariable=self.status, wraplength=1000).pack(padx=8, pady=6)
        targets = ttk.Frame(self.root)
        targets.pack(fill="x", padx=8)
        self.visible_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(targets, text="Visible targets only", variable=self.visible_only,
                        command=self.refresh).pack(side="left")
        self.target = ttk.Combobox(targets, state="readonly", width=90)
        self.target.pack(side="left", fill="x", expand=True)
        actions = ttk.Frame(self.root)
        actions.pack(pady=8)
        for label, action in (("Open", "OpenObject"), ("Close", "CloseObject"),
                              ("Pick up", "PickupObject"), ("Put on / in target", "PutObject")):
            ttk.Button(actions, text=label, command=lambda a=action: self.interact(a)).pack(side="left", padx=3)
        ttk.Button(actions, text="Reset", command=self.reset).pack(side="left", padx=12)
        moves = ttk.Frame(self.root)
        moves.pack(pady=5)
        for label, action in (("Forward ↑", "MoveAhead"), ("Back ↓", "MoveBack"),
                              ("Turn left ←", "RotateLeft"), ("Turn right →", "RotateRight"),
                              ("Look up", "LookUp"), ("Look down", "LookDown"),
                              ("Crouch", "Crouch"), ("Stand", "Stand")):
            ttk.Button(moves, text=label, command=lambda a=action: self.act(a)).pack(side="left", padx=2)
        for key, action in (("Up", "MoveAhead"), ("Down", "MoveBack"),
                            ("Left", "RotateLeft"), ("Right", "RotateRight")):
            self.root.bind(key, lambda event, a=action: self.key_action(event, a))
        self.ego.bind("<Button-1>", self.select_pixel)
        ttk.Label(self.root, text="Arrow keys: move / turn. Open / Pick require proximity and visibility. "
                  "Put targets the receiving table/container. Abstract agent; no Fetch arm simulation.",
                  wraplength=1000).pack(padx=8, pady=6)
        ttk.Label(self.root, text=f"State/action records: {env.output_dir}").pack(pady=4)
        self.refresh()

    def refresh(self):
        event = self.env.controller.last_event
        self.photos = []
        frames = (event.frame, event.third_party_camera_frames[0])
        for widget, frame in zip((self.ego, self.map), frames):
            picture = Image.fromarray(frame)
            if widget is self.map:
                picture.thumbnail((400, 480))
            photo = ImageTk.PhotoImage(picture, master=self.root)
            self.photos.append(photo)
            widget.configure(image=photo)
        selected = self.target.get()
        objects = [o for o in event.metadata["objects"]
                   if (not self.visible_only.get() or o["visible"])
                   and (o["openable"] or o["pickupable"] or o["receptacle"])]
        self.targets = {f"{o['objectType']} | {o['objectId']}": o["objectId"]
                        for o in sorted(objects, key=lambda o: (not o["openable"], not o["pickupable"],
                                                               o["objectType"], o["objectId"]))}
        self.target["values"] = list(self.targets)
        self.target.set(selected if selected in self.targets else next(iter(self.targets), ""))

    def key_action(self, event, action):
        if event.widget is not self.target:
            self.act(action)
            return "break"

    def act(self, action, **parameters):
        self.status.set(f"Executing {action}…")
        self.root.update_idletasks()
        try:
            event = self.env.step(action, **parameters)
            held = [o["objectType"] for o in event.metadata.get("inventoryObjects", [])]
            message = "Succeeded" if event.metadata["lastActionSuccess"] else event.metadata["errorMessage"]
            self.status.set(f"{action}: {message} | Holding: {', '.join(held) or 'nothing'}")
            self.refresh()
        except Exception as error:
            self.status.set(f"Error: {error}")

    def interact(self, action):
        target = self.targets.get(self.target.get())
        if target is None:
            self.status.set("Select a target first; approach it and turn the camera toward it.")
            return
        parameters = {"objectId": target, "forceAction": False}
        if action == "OpenObject":
            parameters["openness"] = 1.0
        elif action == "PutObject":
            parameters["placeStationary"] = True
        self.act(action, **parameters)

    def select_pixel(self, event):
        # Ego image is displayed at native resolution; IDs come from the existing segmentation.
        frame = self.env.controller.last_event
        color = tuple(int(c) for c in frame.instance_segmentation_frame[event.y, event.x])
        object_id = frame.color_to_object_id.get(color)
        for label, candidate in self.targets.items():
            if candidate == object_id:
                self.target.set(label)
                self.status.set(f"Selected {object_id}")
                return
        self.status.set("This pixel is not an available interaction target. Try moving closer.")

    def reset(self):
        try:
            self.env.reset()
            self.refresh()
            self.status.set("Scene reset")
        except Exception as error:
            self.status.set(f"Reset failed: {error}")

    def run(self):
        self.root.mainloop()
