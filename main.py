import torch
import torchvision.transforms.functional as TF
from PIL import Image
import json
import math
import cv2
import numpy as np

from OptimizerEngine import OptimizerEngine
from GPUColorAndLoss import GPUColorAndLoss
from GPUShapes import GPUShapes


class VectorRenderer:
    def __init__(self, image_path, device=None):
        self.target_alpha = None
        self.error_map = None
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.image_path = image_path

        # Globale Vektor-Liste (Hier liegt unser eigentliches "Meisterwerk")
        self.vector_data = []

        # Wir berechnen die Hintergrundfarbe einmal am Anfang
        temp_img = self._load_target_image(64)
        self.mean_color = temp_img.mean(dim=(1, 2), keepdim=True)

        # Aktuelle Arbeitsvariablen (werden pro LOD überschrieben)
        self.resolution = 0
        self.target_img = None
        self.canvas_img = None

    def _load_target_image(self, resolution):
        """Lädt das Bild frisch von der Festplatte in der gewünschten Auflösung."""
        img = Image.open(self.image_path).convert('RGB')
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        return TF.to_tensor(img).to(self.device)

    def _redraw_all_shapes(self):
        """DIE MAGIE: Zeichnet alle bisherigen Formen messerscharf auf die aktuelle Leinwand."""
        for shape in self.vector_data:
            # Parameter aus dem JSON-Format zurück in PyTorch-Tensoren verwandeln
            params = torch.tensor([
                shape["cx"],
                shape["cy"],
                shape["rw"],
                shape["rh"],
                shape["angle"] / (180.0 / math.pi),  # Grad zurück in Bogenmaß
                shape["alpha"]
            ], device=self.device)

            color = torch.tensor(shape["color"], device=self.device) / 255.0
            shape_type = 0 if shape["type"] == "ellipse" else 2

            # Zeichnen!
            self._update_canvas(params, color, shape_type)

    # Füge die Pinselgrößen (min_brush_px, max_brush_px) als Argumente hinzu
    def render(self, preview_interval=10, min_brush_px=5.0, max_brush_px=50.0):
        # Unser Masterplan: Nur noch Auflösung und Anzahl der Formen!
        lods = [
            {"name": "LOD 1 (Grob)", "res": 128, "shapes": 300, "brush_px": 80.0},  # Malt dicke Farbblöcke
            {"name": "LOD 2 (Mittel)", "res": 256, "shapes": 500, "brush_px": 30.0},  # Definiert Schattierungen
            {"name": "LOD 3 (Fein)", "res": 1024, "shapes": 700, "brush_px": 15.0},  # Zeichnet harte Kanten
            {"name": "LOD 4 (Makro)", "res": 2000, "shapes": 200, "brush_px": 10.0},
            {"name": "LOD 5 (Super-Makro)", "res": 4000, "shapes": 200, "brush_px": 10.0}
        ]

        print(f"🚀 Starte echtes Multi-Scale Rendering auf {self.device}")
        print(f"🎨 Feste Pinselgröße: {min_brush_px}px bis {max_brush_px}px\n")

        MAX_BAD_SCORS = 25
        for lod_idx, lod in enumerate(lods):
            bad_cores_pro_LOD = 0
            print(f"{'=' * 50}")
            print(f"🌟 WECHSEL ZU {lod['name']} | Auflösung: {lod['res']}x{lod['res']}")
            print(f"{'=' * 50}\n")

            # 1. Canvas und Target für diese Stufe vorbereiten
            self.resolution = lod['res']
            self.target_img = self._load_target_image(self.resolution)
            self.canvas_img = self.mean_color.expand_as(self.target_img).clone()

            # 2. Alte Formen in neuer Schärfe neu zeichnen
            if len(self.vector_data) > 0:
                print(f"Vektorisiere {len(self.vector_data)} alte Formen auf neue Leinwand...")
                self._redraw_all_shapes()

            # --- DIE PINSEL-MATHEMATIK ---
            # Wir berechnen die relative Größe (0.0 bis 1.0) für DIESE Auflösung

            current_max_s = lod["brush_px"] / self.resolution
            current_min_s = current_max_s / 4
            print(f"Relative Pinselgröße für dieses LOD: {current_min_s:.3f} bis {current_max_s:.3f}\n")

            # 3. Der eigentliche Optimierungs-Loop für diese LOD
            for step in range(lod['shapes']):
                shape_type = 0 if step % 10 < 5 else 0

                # Aufmerksamkeitskarte (Error Map)
                self.error_map = torch.mean(torch.abs(self.target_img - self.canvas_img), dim=0)
                self.target_alpha = (self.error_map > 0.008).float()

                # Engine abfeuern
                best_params, best_color, best_score = OptimizerEngine.find_best_shape(
                    self.target_img, self.canvas_img, self.target_alpha,
                    shape_type=shape_type,
                    n_samples=8000,
                    n_mutate=40,
                    min_size=current_min_s,  # <-- Die dynamisch berechnete Grenze
                    max_size=current_max_s  # <-- Die dynamisch berechnete Grenze
                )

                # Filter schlechte scores

                if best_score > -0.5:
                    bad_cores_pro_LOD +=1
                    if best_score >=0:
                        print(f"Form mit score{best_score} gefiltert. Bereits {bad_cores_pro_LOD} schlechte Shapes gefiltert")
                        if bad_cores_pro_LOD >= MAX_BAD_SCORS:
                            break
                        continue


                # Einbrennen & Speichern
                self._update_canvas(best_params, best_color, shape_type)
                self._save_to_memory(best_params, best_color, shape_type)

                # OpenCV Live Vorschau
                if (step + 1) % preview_interval == 0 or step == 0:
                    self._show_preview(lod['name'], step + 1, lod['shapes'], best_score, shape_type)

        cv2.destroyAllWindows()

    def _show_preview(self, phase_name, step, total_shapes, best_score, shape_type):
        """Kapselt die OpenCV Logik sicher ein."""
        np_img = (self.target_alpha.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        np_img_2 = (self.canvas_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr_img_2 = cv2.cvtColor(np_img_2, cv2.COLOR_RGB2BGR)

        # Egal wie groß das Bild intern ist (selbst bei 2048x2048),
        # wir zeigen das Vorschau-Fenster immer angenehm in 512x512 an!
        display_img = cv2.resize(bgr_img, (512, 512), interpolation=cv2.INTER_AREA)
        display_img_2 = cv2.resize(bgr_img_2, (512, 512), interpolation=cv2.INTER_AREA)

        cv2.imshow("Vector Renderer - Live Preview", display_img)
        cv2.imshow("Vector Renderer - Live Preview_2", display_img_2)
        cv2.waitKey(1)

        shape_name = 'Ellipse' if shape_type == 0 else 'Dreieck'
        print(f"[{phase_name}] Form {step:>4}/{total_shapes} | {shape_name:<8} | Score: {best_score:.2f}")

    def _update_canvas(self, params: torch.Tensor, color: torch.Tensor, shape_type: int):
        with torch.no_grad():
            grid = GPUShapes.create_relative_grid(self.resolution, self.resolution, self.device)
            params_exp = params.unsqueeze(0)

            if shape_type == 0:
                sdfs = GPUShapes.sdf_ellipse(grid.unsqueeze(0), params_exp)
            else:
                sdfs = GPUShapes.sdf_triangle(grid.unsqueeze(0), params_exp)

            mask = torch.sigmoid(-sdfs * 100.0).squeeze(0)
            color_exp = color.view(3, 1, 1)
            effective_alpha = mask * params[5]

            self.canvas_img = (color_exp * effective_alpha) + (self.canvas_img * (1.0 - effective_alpha))

    def _save_to_memory(self, params: torch.Tensor, color: torch.Tensor, shape_type: int):
        shape_data = {
            "type": "ellipse" if shape_type == 0 else "triangle",
            "cx": params[0].item(),
            "cy": params[1].item(),
            "rw": params[2].item(),
            "rh": params[3].item(),
            "angle": params[4].item() * (180.0 / math.pi),
            "alpha": params[5].item(),
            "color": [
                int(color[0].item() * 255),
                int(color[1].item() * 255),
                int(color[2].item() * 255)
            ]
        }
        self.vector_data.append(shape_data)

    def export_results(self, json_path="output.json", img_path="output.png"):
        with open(json_path, 'w') as f:
            json.dump(self.vector_data, f, indent=4)

        final_image = TF.to_pil_image(self.canvas_img.cpu())
        final_image.save(img_path)
        print(f"\n🎉 Fertig! Vektordaten: {json_path} | Bild-Auflösung: {self.resolution}x{self.resolution}")


if __name__ == "__main__":
    IMAGE_PATH = "frierenHeart.jpg"

    # Der Renderer bekommt nur noch den Pfad, er steuert die Auflösung jetzt selbst!
    renderer = VectorRenderer(IMAGE_PATH)

    # 10er Intervalle für das Live-Fenster sind angenehm flüssig
    renderer.render(preview_interval=1)

    renderer.export_results("frieren_vektor.json", "frieren_vektor.png")