import time

import torch
import torchvision.transforms.functional as TF

from profil import RenderPreset

torch._dynamo.config.cache_size_limit = 64

from PIL import Image, ImageFilter
import json
import math
import cv2
import numpy as np
from OptimizerEngine import OptimizerEngine
from GPUShapes import GPUShapes


class VectorRenderer:
    def __init__(self, image_path, device=None):
        self.telemetry_data = []
        self.deleted_scores = []
        self.full_grid = None
        self.last_pinsel = False
        self.ema_pinsel = 0
        self.last_score = -999
        self.target_alpha = None
        self.error_map = None
        self.flat_error_map = None
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
    def render(self,preset: RenderPreset ,preview_interval=10, total_shapes_target=2000, telemetry = False,
               wait_at_finisch=True):

        cfg = preset.value

        self.resolution = cfg["resolution"]
        print(f"\nStarte Darwin-Renderer im Modus: [{preset.name}]")
        print(f"   Mutations: {cfg['n_mutate']} | Samples: {1024 * cfg['sample_multi']} | Patience: {cfg['patience_factor']}")

        global_shapes_drawn = 0
        bad_shapes_count = 0

        global_shapes_drawn = 0
        bad_shapes_count = 0

        # ====================================================================
        # EMA Parameter (jetzt dynamisch aus dem Enum)
        # ====================================================================
        ema_score = None
        first_ema_score = True
        patience_factor = cfg["patience_factor"]
        self.spaghetti_unlocked = False
        ema_negativ_reaction = cfg["ema_negativ_reaction"]
        ema_positiv_reaction = cfg["ema_positiv_reaction"]
        consecutive_bad_scores = 0
        MAX_BAD_SCORES = cfg["MAX_BAD_SCORES"]
        best_rejected_score = float('inf')




        # ====================================================================
        # Setup Leinwand
        # ====================================================================
        self.target_img = self._load_target_image(self.resolution)
        self.canvas_img = self.mean_color.expand_as(self.target_img).clone()
        self.target_alpha = torch.ones(self.resolution, self.resolution, device=self.device)

        self.full_grid = GPUShapes.create_relative_grid(self.resolution, self.resolution, self.device)




        phase_score_sum = 0.0
        phase_shapes_accepted = 0



        # 2. Die EINE saubere Render-Schleife
        while global_shapes_drawn < total_shapes_target:

            # ====================================================================
            # PinselLogik
            # ====================================================================
            start_brush = 0.9
            end_brush = 0.015 if preset == RenderPreset.ULTRA else 0.01  # Luft nach oben für Ultra
            max_virtual_progress = 2000
            pinsel_error_step_size = 1

            progress = (global_shapes_drawn + (
                        min(bad_shapes_count, max_virtual_progress) // pinsel_error_step_size)) / (
                                   total_shapes_target + max_virtual_progress)

            # Steilheit wird aus dem Enum gezogen!
            adjusted_progress = math.pow(progress, cfg["pinsel_steilheit"])
            current_max_s = start_brush * math.pow((end_brush / start_brush), adjusted_progress)

            # ErrorMap gewicht anpassen
            error_map_weight = min(0.4 + (0.7 * progress), 0.9)


            # ====================================================================
            # 2. DYNAMISCHES LIMIT (EMA-Filter)
            # ====================================================================
            if ema_score is None:
                filter_hard_limit = 0.0  # Am Anfang alles erlauben
            else:
                filter_hard_limit = ema_score * patience_factor

            # ====================================================================
            # 3. KACHEL-GRÖSSE & UNTERGRENZEN
            # ====================================================================
            patch_fov_px = (current_max_s * 2.0 * float(self.resolution)) + 48.0
            current_tile_size = max(64, min(128, int(math.ceil(patch_fov_px / 32.0) * 32)))

            # Das Klobig-Minimum (1/3) und das absolute GPU-Minimum (1.5 Pixel)
            pixel_per_grid_cell = patch_fov_px / current_tile_size
            gpu_safe_min_s = max(1.5, pixel_per_grid_cell * 1.5) / float(self.resolution)
            chunky_min_s = current_max_s * 0.33

            # ====================================================================
            # 4. DIE SPAGHETTI-ZANGE & MILESTONE-RESET
            # ====================================================================
            spaghetti_threshold = 0.2
            if current_max_s > spaghetti_threshold:
                # Phase 1: Blockout (Beide Seiten klobig)
                min_w = chunky_min_s
                min_h = chunky_min_s
            else:
                # Phase 2: Details (Dicke darf auf 1.5 Pixel kollabieren)
                min_w = chunky_min_s
                min_h = gpu_safe_min_s

                # EMA-Reset beim Werkzeug-Wechsel!
                if not self.spaghetti_unlocked:
                    print("\n" + "=" * 60)
                    print("SPAGHETTI MODE UNLOCKED!")
                    print("Resette EMA-Score für Neu-Kalibrierung...")
                    print("=" * 60 + "\n")
                    ema_score = None
                    self.spaghetti_unlocked = True

            min_w = min(min_w, current_max_s * 0.5)
            min_h = min(min_h, current_max_s * 0.5)

            min_size_t = torch.tensor([[min_w, min_h]], device=self.device)
            max_size_t = torch.tensor([[current_max_s, current_max_s]], device=self.device)
            patch_fov_px_t = torch.tensor([patch_fov_px], device=self.device)
            # ====================================================================
            # ENGINE START
            # ====================================================================
            best_params, best_color, best_score = OptimizerEngine.find_best_shape(
                self.target_img, self.canvas_img, self.target_alpha,
                n_samples=1024 * cfg["sample_multi"],
                n_mutate=cfg["n_mutate"],
                min_size=min_size_t,
                max_size=max_size_t,
                chunk_size=1024 * cfg["batch_multi"],
                tile_size=current_tile_size,
                patch_fov_px=patch_fov_px_t,
                top_k=cfg["top_k"],
                resolution=self.resolution,
                heat_map=self.flat_error_map,
                alpha_base=min(current_max_s, 0.5)
            )
            shape_type = int(best_params[6].item())

            # ====================================================================
            # FILTER (REJECTION SAMPLING)
            # ====================================================================
            if best_score > filter_hard_limit:
                consecutive_bad_scores += 1
                bad_shapes_count+=1
                if best_score < best_rejected_score:
                    best_rejected_score = best_score
                # Wenn wir absolut feststecken (z.B. 50x in Folge Müll gefunden),
                # weichen wir den Maßstab auf bestes der letzten 50
                if consecutive_bad_scores > MAX_BAD_SCORES:
                    if ema_score is not None:
                        ema_score = best_rejected_score
                    consecutive_bad_scores = 0
                    best_rejected_score = float('inf') #speicher für lokales maximum zurücksetzen
                    print(f"    [Warnung] Stecke fest! EMA-Limit auf {ema_score} gesetzt.")

                if telemetry:
                    # Wir zwingen best_score und ema_score explizit zu normalen Floats
                    s = float(best_score)
                    e = float(ema_score) if ema_score is not None else None
                    self.deleted_scores.append([s, e])
                continue  #Form wegwerfen

            # ====================================================================
            # Erfolg - Filter Anpassen
            # ====================================================================
            consecutive_bad_scores = math.ceil(consecutive_bad_scores / 2)
            global_shapes_drawn += 1
            best_rejected_score = float('inf')  # speicher für lokales maximum zurücksetzen

            # --- NEU: Asymmetrisches EMA Update ---
            if ema_score is None:
                if first_ema_score:
                    first_ema_score = False
                    ema_score = None
                else:
                    ema_score = best_score
            else:
                # WICHTIG: best_score ist negativ. Ein kleinerer Wert (z.B. -200) ist BESSER als -50.
                if best_score < ema_score:
                    # GIERIG: Wir haben eine super Form gefunden! Standard schnell anheben.
                    ema_score = ((1- ema_positiv_reaction) * ema_score) + (ema_positiv_reaction * best_score)
                else:
                    # ZÖGERLICH: Form war schlechter als der Schnitt. Standard nur extrem langsam senken.
                    ema_score = ((1-ema_negativ_reaction) * ema_score) + (ema_negativ_reaction * best_score)


            # ====================================================================
            # FORM EINBRENNEN
            # ====================================================================

            if global_shapes_drawn % 100 == 0:
                self._update_error_map(error_map_weight)


            phase_score_sum += best_score
            phase_shapes_accepted += 1

            geom_only_final = best_params[:6]
            self._update_canvas(geom_only_final, best_color, shape_type)
            self._save_to_memory(geom_only_final, best_color, shape_type)

            # OpenCV Live Vorschau
            if global_shapes_drawn % preview_interval == 0:
                self._show_preview(self.canvas_img, "Vector Renderer - Live Preview")
                if ema_score is not None:
                    print(f"    Form {global_shapes_drawn:>4}/{total_shapes_target}  | Score: {best_score:.2f}"
                      f"| EMA: {ema_score:.2f} | PinselMax: {current_max_s}")
            if global_shapes_drawn >= total_shapes_target:

                print(f"Ziel-Budget von {total_shapes_target} Formen erreicht! Beende Rendering.")
                print(f"{bad_shapes_count} Shapes Weg geworfen. ({bad_shapes_count/(total_shapes_target+bad_shapes_count)*100:.2f}%)")
                break

            if telemetry:
                self.telemetry_data.append({
                    "geometry": geom_only_final.cpu().tolist(),  # <-- HIER FIXEN
                    "score": float(best_score),
                    "ema": float(ema_score) if ema_score is not None else 0.0,
                    "pinsel_max": float(current_max_s),
                    "color": best_color.cpu().tolist(),  # <-- HIER FIXEN
                    "shape_type": shape_type,
                })

        if wait_at_finisch:
            cv2.waitKey()
        cv2.destroyAllWindows()

    def _show_preview(self, img, window_name):
        """Kapselt die OpenCV Logik sicher ein."""

        if img.ndim == 3:
            np_img_2 = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            np_img_2 = (img.cpu().numpy() * 255).astype(np.uint8)
        bgr_img_2 = cv2.cvtColor(np_img_2, cv2.COLOR_RGB2BGR)

        # wir zeigen das Vorschau-Fenster immer in 512x512 an!
        display_img_2 = cv2.resize(bgr_img_2, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, display_img_2)
        cv2.waitKey(1)




    def _update_canvas(self, params: torch.Tensor, color: torch.Tensor, shape_type: int):
        with torch.no_grad():
            grid = self.full_grid
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

        p_list = params.cpu().tolist()
        c_list = color.cpu().tolist()

        shape_data = {
            "type": type_str,
            "cx": p_list[0],
            "cy": p_list[1],
            "rw": p_list[2],
            "rh": p_list[3],
            "angle": p_list[4] * (180.0 / math.pi),  # Bogenmaß zurück in Grad
            "alpha": p_list[5],
            "color": [
                int(c_list[0] * 255),
                int(c_list[1] * 255),
                int(c_list[2] * 255)
            ]
        }
        self.vector_data.append(shape_data)

    def export_results(self, json_path="output.json", img_path="output.png", telemetry_path="telemetry.json"):
        # 1. Das normale Bild-JSON für den Viewer/die SVG
        with open(json_path, 'w') as f:
            json.dump(self.vector_data, f, indent=4)

        # 2. Das Bild abspeichern
        final_image = TF.to_pil_image(self.canvas_img.cpu())
        final_image.save(img_path)

        # 3. NEU: Die Telemetrie für unser Dashboard exportieren!
        telemetry_export = {
            "accepted": self.telemetry_data,
            "rejected": self.deleted_scores
        }
        with open(telemetry_path, 'w') as f:
            json.dump(telemetry_export, f, indent=4)

        print(f"\n🎉 Fertig! Vektordaten: {json_path} | Bild-Auflösung: {self.resolution}x{self.resolution}")
        print(f"📊 Deep Research Daten gespeichert unter: {telemetry_path}")

    def _update_error_map(self, gewichtung):
        with torch.no_grad():
            # 1. Differenz pro Kanal
            diff = torch.abs(self.target_img - self.canvas_img)  # (3, 2048, 2048)

            # 2. Luminanz-Gewichtung (Helle Bereiche sind wichtiger)
            # Wir nehmen das Zielbild als Referenz für Helligkeit
            luminance = torch.mean(self.target_img, dim=0)

            # 3. Farbdifferenz + Helligkeits-Boost
            # Wir multiplizieren den Fehler mit der Luminanz, um Highlights zu pushen
            # Highlights (hell) im Original sollen mehr Fehlermeldung erzeugen
            color_error = torch.mean(diff, dim=0)
            self.error_map = color_error * (1.0 + luminance * 2.0)

            # optional: Sättigungs-Boost für die Augen/Ohrringe
            # Dies ist ein "Pro-Feature": Je gesättigter das Original, desto wichtiger
            saturation = torch.std(self.target_img, dim=0)
            self.error_map = self.error_map * (1.0 + saturation * 3.0)

            self._show_preview(self.error_map, "Color-Aware Error-Map")
            self.flat_error_map = self.error_map.view(-1) + (1 - gewichtung)


if __name__ == "__main__":
    IMAGE_PATH = "bilder/frierenHeart.jpg"

    # Der Renderer bekommt nur noch den Pfad, er steuert die Auflösung jetzt selbst!
    renderer = VectorRenderer(IMAGE_PATH)

    # 10er Intervalle für das Live-Fenster sind angenehm flüssig
    time_start = time.time()
    renderer.render(
        preset=RenderPreset.ULTRA_FAST,
        preview_interval=1,
        total_shapes_target=3000,
        telemetry=True,
        wait_at_finisch=False
    )
    time_end = time.time()
    print(f"Dauer: {time_end - time_start}")
    renderer.export_results("frierenHeart.json", "frierenHeart_vektor_UltraFast.png")