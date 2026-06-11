import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import json
import math
import cv2
import numpy as np
from OptimizerEngine import OptimizerEngine
from GPUShapes import GPUShapes


class VectorRenderer:
    def __init__(self, image_path, device=None):
        self.last_score = -999
        self.target_alpha = None
        self.error_map = None
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.image_path = image_path

        # Globale Vektor-Liste (Hier liegt unser eigentliches "Meisterwerk")
        self.vector_data = []

        # Wir berechnen die Hintergrundfarbe einmal am Anfang
        temp_img = self._load_target_image(128)

        # 1. Pixel flachklopfen und in 0-255 Ganzzahlen (Long) umwandeln
        # Form: von (3, 64, 64) -> (4096, 3) | Jede Zeile ist ein [R, G, B] Pixel
        pixels = (temp_img.permute(1, 2, 0) * 255).long().view(-1, 3)

        # 2. Bit-Packing: Wir codieren R, G, B in eine einzige, eindeutige Zahl
        # R * 256^2 + G * 256 + B. Das macht aus der 3D-Farbe einen einfachen 1D-Wert.
        encoded_pixels = pixels[:, 0] * 65536 + pixels[:, 1] * 256 + pixels[:, 2]

        # 3. Zähle, welche codierte Farbe am häufigsten im Bild vorkommt
        unique_colors, counts = torch.unique(encoded_pixels, return_counts=True)
        dominant_color_idx = torch.argmax(counts)
        winner_encoded = unique_colors[dominant_color_idx]

        # 4. Die Gewinner-Zahl wieder zurück in R, G, B decodieren und auf 0.0-1.0 normalisieren
        r = (winner_encoded // 65536).float() / 255.0
        g = ((winner_encoded % 65536) // 256).float() / 255.0
        b = (winner_encoded % 256).float() / 255.0

        # 5. Als (3, 1, 1) Tensor speichern, damit expand_as() im Loop perfekt matcht
        self.mean_color = torch.tensor([r, g, b], device=self.device).view(3, 1, 1)

        # Aktuelle Arbeitsvariablen (werden pro LOD überschrieben)
        self.resolution = 0
        self.target_img = None
        self.canvas_img = None

    def _load_target_image(self, resolution, blur_radius=0):
        """Lädt das Bild und verschmiert es (blinzeln) für grobe LODs!"""
        img = Image.open(self.image_path).convert('RGB')
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # NEU: Der Blinzel-Trick
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))

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
            if shape["type"] == "ellipse":
                shape_type = 0
            elif shape["type"] == "rectangle":
                shape_type = 1
            else:
                shape_type = 2

            # Zeichnen!
            self._update_canvas(params, color, shape_type)

    # Füge die Pinselgrößen (min_brush_px, max_brush_px) als Argumente hinzu
    def render(self, preview_interval=10, min_brush_px=5.0,
               total_shapes_target=2000, smart = False):


        self.resolution = 1024
        print(f"Starte Darwin-Renderer auf {self.device} | Auflösung: {self.resolution}x{self.resolution}")
        print(f"Ziel-Budget: {total_shapes_target} Formen insgesamt")

        MAX_BAD_SCORES = 50  # Nach 100 Fehlversuchen in Folge wird das LOD gewechselt
        global_shapes_drawn = 0  # Unser globaler Meister-Zähler


        consecutive_bad_scores = 0  # Zählt die Fehlversuche für das AKTUELLE LOD


        # ---- EMA FILTER SETUP ----
        ema_score = None
        ema_alpha = 0.15  # Einfluss einer neuen Form (15 %)
        tolerance = 0.15  # Akzeptierte Verschlechterung (15 %)
        decay_factor = 0.98  # Abschwächung bei Fehlversuchen (2 %)
        WARMUP_SHAPES = 30


        #---- SMARTE PINSEL-MATHEMATIK ----

        if global_shapes_drawn == 25 or global_shapes_drawn == 100:
            ema_score = None

        if global_shapes_drawn < 25 :
            # Phase 1: Fundament (Form 0 bis 24) -> Pinsel fest auf 50%
            current_brush_px = self.resolution * 0.50

        elif global_shapes_drawn < 70 :
            # Phase 2: Struktur (Form 25 bis 99) -> Pinsel fest auf 30%
            current_brush_px = self.resolution * 0.30

        else:
            # Phase 3: Details (Form 100 bis Ziel) -> Stetig sinkend von 25% auf Minimum
            shapes_left = max(1, total_shapes_target - 100)
            progress = (global_shapes_drawn - 100) / shapes_left

            start_brush = self.resolution * 0.25
            end_brush = min_brush_px

            current_brush_px = start_brush * (1.0 - progress) + end_brush * progress

        # In relatives Maß (0.0 bis 1.0) umwandeln
        current_max_s = current_brush_px / self.resolution

        # Die Zange: Das Minimum ist immer strikt 50% vom Maximum!
        current_min_s = current_max_s * 0.5


        patch_fov_px = (current_brush_px * 2.0) + 48.0
        current_tile_size = int(math.ceil(patch_fov_px / 32.0) * 32)
        current_tile_size = max(64, current_tile_size)
        current_tile_size = min(128, current_tile_size)

        self.target_img = self._load_target_image(self.resolution)
        self.canvas_img = self.mean_color.expand_as(self.target_img).clone()


        # 3. Der eigentliche Optimierungs-Loop
        # Er läuft so lange, bis unser globales Budget leer ist ODER dieses LOD ausgereizt ist.
        self.target_alpha = torch.ones(self.resolution, self.resolution, device=self.device)
        while global_shapes_drawn < total_shapes_target:

            best_params, best_color, best_score = OptimizerEngine.find_best_shape(
                self.target_img, self.canvas_img, self.target_alpha,
                n_samples=1024*10,
                n_mutate=64,
                min_size=current_min_s,
                max_size=current_max_s,
                chunk_size=1024,
                tile_size=current_tile_size,
                patch_fov_px=patch_fov_px,
                top_k=32

            )
            shape_type = int(best_params[6].item())

            # --- DER SMARTE FILTER ---
            if global_shapes_drawn < WARMUP_SHAPES:
                current_limit = 0
            elif ema_score is None:
                # Bei der allerersten Form haben wir noch keinen Durchschnitt.
                # Alles was das Bild verbessert (< 0.0) wird akzeptiert.
                current_limit = 0.0
            else:
                # Da Scores negativ sind: -100 * (1.0 - 0.15) = -85.0
                current_limit = ema_score * (1.0 - tolerance)

            if best_score > current_limit:
                consecutive_bad_scores += 1

                # DEADLOCK-SCHUTZ: Wir weichen den Durchschnitt auf!
                # Wenn wir nichts finden, passen wir unsere Ansprüche langsam an den kleineren Pinsel an.
                if ema_score is not None:
                    ema_score *= decay_factor

                if consecutive_bad_scores % 50 == 0:
                    print(f"⚠️ {consecutive_bad_scores} Fehlversuche. Senke Anspruch auf: {current_limit:.2f}")

                if consecutive_bad_scores >= MAX_BAD_SCORES:
                    print(f"\n🛑 Nichts mehr zu verbessern! Breche ab.")
                    break
                continue

            # --- WENN WIR HIER SIND, WAR DIE FORM EIN ERFOLG! ---

            # Bad shapes counter halbieren anstelle von auf 0 setzen
            consecutive_bad_scores = math.ceil(consecutive_bad_scores/2)
            global_shapes_drawn += 1  # Budget um 1 verringern

            if ema_score is None:
                ema_score = best_score
            else:
                ema_score = (ema_alpha * best_score) + ((1.0 - ema_alpha) * ema_score)

            # Einbrennen & Speichern
            geom_only_final = best_params[ :6]
            self._update_canvas(geom_only_final, best_color, shape_type)
            self._save_to_memory(geom_only_final, best_color, shape_type)

            # OpenCV Live Vorschau
            if global_shapes_drawn % preview_interval == 0 or global_shapes_drawn == 1:
                self._show_preview("Darwin Mode", global_shapes_drawn, total_shapes_target, best_score, shape_type)

            # Wenn wir hier ankommen und das Budget voll ist, brechen wir auch die äußere (LOD) Schleife ab!
            if global_shapes_drawn >= total_shapes_target:
                print(f" Globales Ziel-Budget von {total_shapes_target} Formen erreicht! Beende Rendering.")
                break

        cv2.destroyAllWindows()

    def _show_preview(self, phase_name, step, total_shapes, best_score, shape_type):
        """Kapselt die OpenCV Logik sicher ein."""

        np_img_2 = (self.canvas_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr_img_2 = cv2.cvtColor(np_img_2, cv2.COLOR_RGB2BGR)

        # Egal wie groß das Bild intern ist (selbst bei 2048x2048),
        # wir zeigen das Vorschau-Fenster immer angenehm in 512x512 an!
        display_img_2 = cv2.resize(bgr_img_2, (512, 512), interpolation=cv2.INTER_AREA)


        cv2.imshow("Vector Renderer - Live Preview_2", display_img_2)
        cv2.waitKey(1)

        if shape_type == 0:
            shape_name = "Ellipse"
        if shape_type == 1:
            shape_name = "Triangle"
        if shape_type == 2:
            shape_name = "Rectangle"

        print(f"[{phase_name}] Form {step:>4}/{total_shapes} | {shape_name:<8} | Score: {best_score:.2f}")

    def _update_canvas(self, params: torch.Tensor, color: torch.Tensor, shape_type: int):
        with torch.no_grad():
            grid = GPUShapes.create_relative_grid(self.resolution, self.resolution, self.device)
            params_exp = params.unsqueeze(0)

            if shape_type == 0:
                sdfs = GPUShapes.sdf_ellipse(grid.unsqueeze(0), params_exp)
            elif shape_type == 1:
                sdfs = GPUShapes.sdf_rectangle(grid.unsqueeze(0), params_exp)
            else:
                sdfs = GPUShapes.sdf_triangle(grid.unsqueeze(0), params_exp)

            mask  = torch.sigmoid(-sdfs * 1000.0)
            color_exp = color.view(3, 1, 1)
            effective_alpha = mask * params[5]

            self.canvas_img = (color_exp * effective_alpha) + (self.canvas_img * (1.0 - effective_alpha))

    def _save_to_memory(self, params: torch.Tensor, color: torch.Tensor, shape_type: int):
        # Den Typen-Namen anhand deines Codes (0=ellipse, 1=rectangle, 2=triangle) zuweisen
        if shape_type == 0:
            type_str = "ellipse"
        elif shape_type == 1:
            type_str = "rectangle"
        else:
            type_str = "triangle"

        shape_data = {
            "type": type_str,
            "cx": params[0].item(),
            "cy": params[1].item(),
            "rw": params[2].item(),
            "rh": params[3].item(),
            "angle": params[4].item() * (180.0 / math.pi),  # Bogenmaß zurück in Grad
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
    IMAGE_PATH = "bilder/frierenAuto.webp"

    # Der Renderer bekommt nur noch den Pfad, er steuert die Auflösung jetzt selbst!
    renderer = VectorRenderer(IMAGE_PATH)

    # 10er Intervalle für das Live-Fenster sind angenehm flüssig
    time_start = time.time()
    renderer.render(preview_interval=20, total_shapes_target=3000, smart=False)
    time_end = time.time()
    print(f"Dauer: {time_end - time_start}")
    renderer.export_results("frieren_vektor_2.json", "frieren_vektor_2.png")