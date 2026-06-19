import random
import time

import torch
import torchvision.transforms.functional as TF
from math import ceil

from cpuUtils import filter_top_shapes
from profil import RenderPreset

torch._dynamo.config.cache_size_limit = 64

from PIL import Image, ImageFilter
import json
import math
import cv2
import numpy as np
from OptimizerEngine import OptimizerEngine
from GPUShapes import GPUShapes
import logging
import torch._logging
torch._logging.set_logs(
    dynamo=logging.WARNING,

    # Die beiden WICHTIGSTEN Logs für dich:
    graph_breaks=True,  # Warnt dich, wenn er die Optimierung abbrechen muss (z.B. wegen In-Place Operationen)
    recompiles=True  # Warnt dich, wenn er neu kompilieren muss (weil sich die Input-Größe unerwartet geändert hat)
)
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
        self.all_samples = []


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

    def _build_static_grid(self, current_max_s):
        """
        Baut das statische Bucket-Grid für die aktuelle Detailstufe (LOD).
        Wird nur aufgerufen, wenn der Pinsel signifikant kleiner wird.
        """
        brush_px = current_max_s * self.resolution

        # 1. Kachel-Größe bestimmen (Kernbereich)
        # Wir runden auf 32er Schritte. Max 1/4 des Bildes, Min 32 Pixel.
        raw_core = max(32, min(self.resolution // 4, int(brush_px * 2)))
        core_size = math.ceil(raw_core / 64) * 64

        stride = core_size // 2  # 50% Überlappung

        # 2. Das FOV für die GPU (Kern + Pinselüberhang)
        # Das ist exakt die Breite/Höhe in Pixeln, die wir aus dem großen Bild schneiden.
        buffer_padding = math.ceil(brush_px * 1.42) + 4
        patch_fov_px = core_size + (2 * buffer_padding)

        buckets = []
        bucket_id = 0

        # 3. Grid über das Bild legen
        for y in range(0, self.resolution - stride, stride):
            for x in range(0, self.resolution - stride, stride):

                # --- A: KERNBEREICH (Für die Spawns der cx, cy) ---
                x_start, y_start = x, y
                x_end = min(x + core_size, self.resolution)
                y_end = min(y + core_size, self.resolution)

                # Wenn der Kernbereich am Rand winzig wird (kleiner als 16px), überspringen wir ihn
                if (x_end - x_start) < 16 or (y_end - y_start) < 16:
                    continue

                # --- B: GPU BERECHNUNGEN (Das FOV-Zentrum) ---
                # Die GPU braucht für theta (affine_grid) das genaue Zentrum des Buffers.
                # WICHTIG: Das Zentrum des Buffers ist das Zentrum des Kerns!
                center_x_px = x_start + (core_size / 2.0)
                center_y_px = y_start + (core_size / 2.0)

                # Umrechnen in -1.0 bis +1.0 für PyTorch grid_sample
                # 0px -> -1.0 | resolution -> +1.0
                tx = (center_x_px / self.resolution) * 2.0 - 1.0
                ty = (center_y_px / self.resolution) * 2.0 - 1.0

                buckets.append({
                    "id": bucket_id,
                    "core_x_start": x_start,
                    "core_x_end": x_end,
                    "core_y_start": y_start,
                    "core_y_end": y_end,
                    "gpu_tx": tx,  # Perfekt vorbereitet für die GPU!
                    "gpu_ty": ty,  # Perfekt vorbereitet für die GPU!
                    "shape_count": 0,
                    "failed_attempts": 0,  # NEU: Strike-Zähler
                    "locked_until_brush": 0.0,  # NEU: Sperr-Grenze

                })
                bucket_id += 1

        # Wir merken uns das aktuelle Grid und die FOV-Größe in der Klasse
        self.active_grid = buckets
        self.active_fov_px = patch_fov_px
        self.active_core_size = core_size

        print(
            f"\n[LOD Update] Pinsel: {brush_px:.1f}px | Core: {core_size}px | FOV: {patch_fov_px}px | Buckets: {len(buckets)}")



    """
    bucket_tiers ist ein dickt mit 4 tier stufen finisched, low, normal, high.
    jede dieser stufen ist None oder eine liste mit den zu stufe gehörenden bucket IDs
    
    buckets_per_it muss durch 4 Teilbar sein
    samples_per_it und buckets_per_it mussen ein verhltnis von:
        samples_per_it = n * samples_per_it * 8
    
    """

    def _generate_samples_from_buckets(self,current_max_s, bucket_tiers: dict, buckets_per_it:int, samples_per_it:int):
        if bucket_tiers is None:
            bucket_tiers = {"normal": range(len(self.active_grid))}

        valid_high, valid_normal, valid_low = [], [], []

        for b_id, b in enumerate(self.active_grid):
            # Sperre prüfen
            if b["locked_until_brush"] > 0 and current_max_s >= b["locked_until_brush"]: continue
            # Soft-Cap prüfen (max 15 Shapes pro LOD-Stufe)
            if b["shape_count"] >= 15: continue

            # Sortieren (Dummy-Logik, solange LPIPS-Heatmap fehlt)
            if bucket_tiers and b_id in bucket_tiers.get("high", []):
                valid_high.append(b_id)
            elif bucket_tiers and b_id in bucket_tiers.get("low", []):
                valid_low.append(b_id)
            else:
                valid_normal.append(b_id)

        target_high = buckets_per_it //2
        target_normal = buckets_per_it //4
        target_low = buckets_per_it //4

        # HIGH auswerten & Überlauf berechnen
        high_count = min(target_high, len(valid_high))
        leftover_high = target_high - high_count

        # NORMAL auswerten (Bekommt den Rest von HIGH)
        target_normal += leftover_high
        normal_count = min(target_normal, len(valid_normal))
        leftover_normal = target_normal - normal_count

        # LOW auswerten (Bekommt den Rest von NORMAL)
        target_low += leftover_normal
        low_count = min(target_low, len(valid_low))

        # 2. Buckets zufällig ziehen
        blocked_set = set()

        selected_high = self._smart_sample(valid_high, high_count, blocked_set)
        selected_normal = self._smart_sample(valid_normal, normal_count, blocked_set)
        selected_low = self._smart_sample(valid_low, low_count, blocked_set)

        # 3. Wir nutzen eine lokale Liste statt self.all_samples (sauberer)
        all_samples_tensors = []

        budget_h, budget_m, budget_l = self.calculate_attention(buckets_per_it, samples_per_it)

        # 4. Samples generieren (mit den richtigen Variablen!)
        all_samples_tensors.extend(self.generate_samples(budget_h, selected_high))
        all_samples_tensors.extend(self.generate_samples(budget_m, selected_normal))
        all_samples_tensors.extend(self.generate_samples(budget_l, selected_low))

        # Wenn gar keine Buckets gefunden wurden (Ende des Bildes)
        if not all_samples_tensors:
            return None, None

        # 5. Alles zu einem einzigen (N, 5) Tensor zusammenkleben
        final_samples_tensor = torch.cat(all_samples_tensors, dim=0)

        # Das Paket für die GPU schnüren
        active_bucket_count = high_count + normal_count + low_count
        gpu_package = {
            "patch_fov_px": self.active_fov_px,
            "b_count": active_bucket_count
        }

        return final_samples_tensor, gpu_package

    def generate_samples(self, budget: int, selected_buckets: list[int]):
        """Generiert die Tensoren für eine Liste von Buckets hoch-vektorisiert ohne Python-Loop!"""
        if budget <= 0 or not selected_buckets:
            return []

        num_buckets = len(selected_buckets)
        total_samples = num_buckets * budget

        # 1. Daten aus dem Dictionary flachklopfen (Harmlose Python-Listen)
        x_starts = [self.active_grid[b]["core_x_start"] for b in selected_buckets]
        x_ends   = [self.active_grid[b]["core_x_end"] for b in selected_buckets]
        y_starts = [self.active_grid[b]["core_y_start"] for b in selected_buckets]
        y_ends   = [self.active_grid[b]["core_y_end"] for b in selected_buckets]
        b_ids    = [self.active_grid[b]["id"] for b in selected_buckets]
        txs      = [self.active_grid[b]["gpu_tx"] for b in selected_buckets]
        tys      = [self.active_grid[b]["gpu_ty"] for b in selected_buckets]

        # 2. In EINEN einzigen Tensor auf der GPU werfen und direkt auf das Budget "aufblasen"
        # repeat_interleave macht aus [A, B] bei budget=3 -> [A, A, A, B, B, B]
        x_s_t = torch.tensor(x_starts, device=self.device, dtype=torch.float32).repeat_interleave(budget)
        x_e_t = torch.tensor(x_ends, device=self.device, dtype=torch.float32).repeat_interleave(budget)
        y_s_t = torch.tensor(y_starts, device=self.device, dtype=torch.float32).repeat_interleave(budget)
        y_e_t = torch.tensor(y_ends, device=self.device, dtype=torch.float32).repeat_interleave(budget)
        b_id_t = torch.tensor(b_ids, device=self.device, dtype=torch.float32).repeat_interleave(budget)
        tx_t = torch.tensor(txs, device=self.device, dtype=torch.float32).repeat_interleave(budget)
        ty_t = torch.tensor(tys, device=self.device, dtype=torch.float32).repeat_interleave(budget)

        # 3. Exakt ZWEI Random-Aufrufe statt Hunderter!
        rand_x = (torch.rand(total_samples, device=self.device) * (x_e_t - x_s_t) + x_s_t) / float(self.resolution)
        rand_y = (torch.rand(total_samples, device=self.device) * (y_e_t - y_s_t) + y_s_t) / float(self.resolution)

        # 4. Relative Grenzen berechnen (alles Vektor-Mathe!)
        t_xmin = x_s_t / float(self.resolution)
        t_xmax = x_e_t / float(self.resolution)
        t_ymin = y_s_t / float(self.resolution)
        t_ymax = y_e_t / float(self.resolution)

        # 5. Ein einziger GPU-Stack-Befehl
        bucket_tensor = torch.stack((rand_x, rand_y, b_id_t, tx_t, ty_t, t_xmin, t_xmax, t_ymin, t_ymax), dim=1)

        return [bucket_tensor]

    def _smart_sample(self, pool: list[int], target_count: int, blocked_set: set):
        if not pool: return []
        random.shuffle(pool)
        selected = []
        stride = self.active_core_size // 2

        # 1. Versuch: Räumlich getrennt (Spatial NMS)
        for b_id in pool:
            b = self.active_grid[b_id]
            gx, gy = b["core_x_start"] // stride, b["core_y_start"] // stride
            if (gx, gy) in blocked_set: continue

            selected.append(b_id)
            # Blockiere das 3x3 Umfeld
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]: blocked_set.add((gx + dx, gy + dy))
            if len(selected) >= target_count: break

        # 2. Fallback: Wenn wir das Target durch Blockaden nicht erreicht haben, fülle blind auf!
        if len(selected) < target_count:
            remaining_pool = [b for b in pool if b not in selected]
            needed = target_count - len(selected)
            selected.extend(random.sample(remaining_pool, min(needed, len(remaining_pool))))

        return selected

    def calculate_attention(self, total_objects, total_attention):
        # 1. Objekte auf Tiers verteilen (50%, 25%, 25%)
        target_high = total_objects // 2
        target_normal = total_objects // 4
        target_low = total_objects // 4

        # 2. Die relativen Gewichte festlegen (abgeleitet von 26, 8, 4)
        weight_high = 13
        weight_normal = 4
        weight_low = 2

        # 3. Berechnen der gewichteten Gesamtsumme
        weighted_sum = (target_high * weight_high) + \
                       (target_normal * weight_normal) + \
                       (target_low * weight_low)

        # 4. Den Basiswert ermitteln
        base_unit = total_attention / weighted_sum

        # 5. Die finale Punkteverteilung pro Objekt ermitteln
        attention_high = base_unit * weight_high
        attention_normal = base_unit * weight_normal
        attention_low = base_unit * weight_low

        return int(attention_high), int(attention_normal), int(attention_low)

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
               wait_at_finisch=True,max_shapes_per_iteration=3):

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

            # ====================================================================
            # GRID UPDATE LOGIK
            # ====================================================================
            # Wir bauen das Grid nur am Anfang ODER wenn der Pinsel 20% kleiner geworden ist
            if not hasattr(self, 'last_grid_brush') or current_max_s < self.last_grid_brush * 0.8:
                self._build_static_grid(current_max_s)
                self.last_grid_brush = current_max_s

            # ErrorMap gewicht anpassen
            error_map_weight = min(0.4 + (0.7 * progress), 0.9)

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

            num_samples = 1024 * cfg["sample_multi"]

            # 1. POSITIONEN GENERIEREN (Heatmap oder Random)
            final_samples_tensor, gpu_package = self._generate_samples_from_buckets(
                bucket_tiers=None,
                buckets_per_it=128,  # Unser Trichter-Limit
                samples_per_it=num_samples,  # Gesamtes Sample-Budget
                current_max_s=current_max_s,
            )

            # WICHTIG: Fallback, falls mal wirklich nichts zurückkommt
            if final_samples_tensor is None:
                continue

            best_params, best_color, best_score = None, None, None
            elite_tensor = OptimizerEngine.find_best_shape(
                target_img=self.target_img,
                canvas_img=self.canvas_img,
                random_samples=final_samples_tensor,
                target_alpha=self.target_alpha,
                n_mutate=cfg["n_mutate"],
                min_size=min_size_t,
                max_size=max_size_t,
                chunk_size=1024 * cfg["batch_multi"],
                tile_size=current_tile_size,
                patch_fov_px=patch_fov_px_t,
                top_k=cfg["top_k"],
                resolution=self.resolution,
                alpha_base=min(current_max_s, 0.5),
                gpu_package=gpu_package,

            )
            elite_shapes = elite_tensor.detach().cpu().numpy()


            # ====================================================================
            # DYNAMISCHES LIMIT (EMA-Filter)
            # ====================================================================
            if ema_score is None:
                filter_hard_limit = None  # Signal an cpuUtils: Nichts filtern!
                current_max_return = 1  # WICHTIG: Am Anfang nur exakt 1 Form zulassen, um die Baseline zu setzen
            else:
                filter_hard_limit = ema_score * patience_factor
                current_max_return = max_shapes_per_iteration  # Wenn der Filter läuft, dürfen es auch 3 gleichzeitig sein

            final_shapes = filter_top_shapes(elites_tensor=elite_shapes,
                                             max_return_shapes=current_max_return,
                                             filter_hard_limit=filter_hard_limit,
                                             overlap_margin=0.95
                                             )

            # ====================================================================
            # BUCKET REWARD & PUNISHMENT
            # ====================================================================
            queried_buckets = torch.unique(final_samples_tensor[:, 2]).cpu().numpy().astype(int).tolist()
            accepted_ids = [int(shape_data[12]) for shape_data in final_shapes] if len(final_shapes) > 0 else []

            if len(final_shapes) == 0:
                consecutive_bad_scores += 1
                bad_shapes_count+=1
                best_rejected_candidate = elite_shapes[0, 10].item()

                if best_rejected_candidate < best_rejected_score:
                    best_rejected_score = best_rejected_candidate

                if consecutive_bad_scores > MAX_BAD_SCORES:
                    if ema_score is not None:
                        ema_score = best_rejected_score
                    consecutive_bad_scores = 0
                    best_rejected_score = float('inf') #speicher für lokales maximum zurücksetzen
                continue  #Form wegwerfen

            # ====================================================================
            # FILTER (REJECTION SAMPLING)
            # ====================================================================
            if len(final_shapes) == 0:
                consecutive_bad_scores += 1
                bad_shapes_count += 1
                best_rejected_candidate = elite_shapes[0, 10].item()

                if best_rejected_candidate < best_rejected_score:
                    best_rejected_score = best_rejected_candidate

                if consecutive_bad_scores > MAX_BAD_SCORES:
                    if ema_score is not None:
                        ema_score = best_rejected_score
                    consecutive_bad_scores = 0
                    best_rejected_score = float('inf')  # speicher für lokales maximum zurücksetzen
                continue  # Form wegwerfen

            # ====================================================================
            # Erfolg - Filter Anpassen
            # ====================================================================
            consecutive_bad_scores = math.ceil(consecutive_bad_scores / 2)
            best_rejected_score = float('inf')  # speicher für lokales maximum zurücksetzen
            mean_batch_score = None

            for shape_data in final_shapes:



                # Abbruch-Bedingung prüfen
                if global_shapes_drawn >= total_shapes_target:
                    print(f"\nZiel-Budget von {total_shapes_target} Formen erreicht! Beende Rendering.")

                    # Sicherer Prozent-Rechner (verhindert Division by Zero)
                    total_attempts = total_shapes_target + bad_shapes_count
                    reject_rate = (bad_shapes_count / total_attempts * 100) if total_attempts > 0 else 0.0

                    print(f"{bad_shapes_count} Shapes weggeworfen. ({reject_rate:.2f}% Rejection Rate)")
                    break  # Bricht die for-Schleife ab. Die äußere while-Schleife beendet sich danach automatisch.



                # ====================================================================
                # TELEMETRIE & LOGGING (Innerhalb der for-Schleife!)
                # ====================================================================
                if telemetry:
                    self.telemetry_data.append({
                        "geometry": best_params.cpu().tolist(),
                        # <-- GEFIXT: Nutzt jetzt best_params (alle 7 Werte)
                        "score": float(best_score),
                        "ema": float(ema_score) if ema_score is not None else 0.0,
                        "pinsel_max": float(current_max_s),
                        "color": best_color.cpu().tolist(),  # <-- GEFIXT: Das hat schon gepasst
                        "shape_type": shape_type,
                    })

            best_params = torch.tensor(shape_data[:7], device=self.device)
            best_color = torch.tensor(shape_data[7:10], device=self.device)
            # ====================================================================
            # EMA score agg
            # ====================================================================

            batch_scores = [s[10] for s in final_shapes]
            mean_batch_score = sum(batch_scores) / len(batch_scores)

            best_score = shape_data[10]
            shape_type = int(shape_data[11])

            global_shapes_drawn += 1

            # --- GANZ SAUBERES EMA UPDATE ---
            if ema_score is not None:
                # OpenCV Live Vorschau
                if global_shapes_drawn % preview_interval == 0:
                    self._show_preview(self.canvas_img, "Vector Renderer - Live Preview")
                    if ema_score is not None:
                        print(
                            f"    Form {global_shapes_drawn:>4}/{total_shapes_target}  | Score: {best_score:.2f} "
                            f"| EMA: {ema_score:.2f} | PinselMax: {current_max_s:.4f}")
            # ====================================================================
            # FORM EINBRENNEN
            # ====================================================================


                if global_shapes_drawn % 100 == 0:
                    self._update_error_map(error_map_weight)


            self._update_canvas(best_params, best_color, shape_type)
            self._save_to_memory(best_params, best_color, shape_type)

            # ====================================================================
            # UPDATE EMA
            # ====================================================================
            if ema_score is None:
                ema_score = mean_batch_score  # Der Startschuss!
            else:
                if mean_batch_score < ema_score:
                    ema_score = ((1 - ema_positiv_reaction) * ema_score) + (ema_positiv_reaction * mean_batch_score)
                else:
                    ema_score = ((1 - ema_negativ_reaction) * ema_score) + (ema_negativ_reaction * mean_batch_score)


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

            math_params = params_exp[:, :6]

            if shape_type == 0:
                sdfs = GPUShapes.sdf_ellipse(grid.unsqueeze(0), math_params)
            elif shape_type == 1:
                sdfs = GPUShapes.sdf_rectangle(grid.unsqueeze(0), math_params)
            else:
                sdfs = GPUShapes.sdf_triangle(grid.unsqueeze(0), math_params)

            mask = torch.sigmoid(-sdfs * 1000.0)
            color_exp = color.view(3, 1, 1)

            effective_alpha = mask * params[6]

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
            "skew": p_list[5],
            "alpha": p_list[6],
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
    preset=RenderPreset.SMALL
    renderer.render(
        preset=preset,
        preview_interval=100,
        total_shapes_target=3000,
        telemetry=False,
        wait_at_finisch=False,
        max_shapes_per_iteration = preset.value["max_shapes_per_iteration"],
    )
    time_end = time.time()
    print(f"Dauer: {time_end - time_start}")
    renderer.export_results("frierenHeart.json", "frierenHeart_vektor_UltraFast.png")