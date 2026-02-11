from typing import Self


class ColorTheme:
    def __init__(self):
        self.system = {
            "darkamber": (255, 176, 0),
            "lightamber": (255, 204, 0),
            "yellow": (173, 173, 30),
            "green": (95, 222, 70),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "darkgray": (40, 40, 40),
            "gray": (140, 140, 140),
            "heaven": (249, 247, 180)
        }
        self.ui = {
            "highlight": self.system["darkamber"],
            "foreground": self.system["gray"],
            "background": self.system["black"]
        }
        self.nvar_def = [
            (1, 140, "#5f81eb",(95, 129, 235), "Normal"),
            (140, 166, "#aa5feb",(170, 95, 235), "Elevated"),
            (166, 219, "#e8851a", (232, 133, 26), "High"),
            (219, 279, "#b6b300", (219, 216, 2), "Very High"),
            (279, 325, "#ffa7be", (255, 167, 190), "Extreme") # True cap is 324
        ]
        
    def get_color_by_nvar(self, nvar, color_type="rgb"):
        nv = int(nvar)
        for low, hi, hex, rgb, label in self.nvar_def:
            if int(nv) < hi:
                if color_type == "rgb":
                    return rgb
                elif color_type == "hex":
                    return hex
                else:
                    print("No valid color type specified")
                    return
        print(f"No valid match for {nv}")

    def get_label_by_nvar(self, nvar):
        nv = int(nvar)
        for low, hi, hex, rgb, label in self.nvar_def:
            if nv < hi:
                return label
        print(f"No valid match for {nv}")

    def color(self, label):
        colors = self.system | self.ui
        if label in colors:
            return colors[label]
        else:
            print(f"{label} not found")

    def get_nvar_def(self):
        return self.nvar_def

