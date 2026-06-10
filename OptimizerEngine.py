import torch
import torch.nn.functional as F
import math

from GPUColorAndLoss import GPUColorAndLoss
from GPUShapes import GPUShapes


class OptimizerEngine:
    """
    Modul 3: Das Herzstück. Sucht, filtert und optimiert Formen im Round-Robin-Verfahren.
    Nutzt Delta-Loss gegen tote Formen und friert Alpha ein.
    """

    @staticmethod
    def find_best_shape(target_img: torch.Tensor, canvas_img: torch.Tensor, target_alpha: torch.Tensor,
                        shape_type: int, n_samples: int = 2000, top_k: int = 64,
                        n_mutate: int = 40, tile_size: int = 112, chunk_size: int = 512,
                        min_size: float = 0.02, max_size: float = 0.8) -> tuple:

        device = target_img.device




        # 1. FUNKTIONS-ZEIGER: Compiler-freundliche Auswahl der Mathematik
        if shape_type == 0:
            sdf_function = GPUShapes.sdf_ellipse
        elif shape_type == 1:
            sdf_function = GPUShapes.sdf_rectangle
        elif shape_type == 2:
            sdf_function = GPUShapes.sdf_triangle
        else:
            raise ValueError("Unbekannter Shape-Typ!")

        # ==========================================
        # PHASE 1: DIE SCHROTFLINTE (No Grad)
        # ==========================================
        with torch.no_grad():
            # 1. Alle Parameter einmalig generieren
            size_range = max_size - min_size
            params = torch.rand((n_samples, 6), device=device)  #Position
            params[:, 2:4] = params[:, 2:4] * size_range + min_size    #Größe
            params[:, 4] = params[:, 4] * math.pi   #Rotation
            params[:, 5] /= 2.0  #Alpha

            # Liste für die Ergebnisse
            all_scores = []

            # 2. CHUNKING-LOOP: Wir verarbeiten die Samples in sicheren Blöcken
            for i in range(0, n_samples, chunk_size):
                # Schneide den aktuellen 500er Block aus den Parametern heraus
                chunk_params = params[i: i + chunk_size]

                # Kacheln nur für diese 500 ausstanzen
                T_target, T_canvas, T_alpha, local_grids = OptimizerEngine._extract_tiles(
                    chunk_params, target_img, canvas_img, target_alpha, tile_size
                )



                # Rohe Bewertung für diese 500
                sdfs = sdf_function(local_grids, chunk_params)
                masks = torch.sigmoid(-sdfs * 5.0)
                alphas = chunk_params[:, 5]

                colors = GPUColorAndLoss.compute_optimal_color(T_target, T_canvas, masks, alphas, T_alpha)
                blended = GPUColorAndLoss.blend_shape(T_canvas, colors, masks, alphas)

                # Scores für diese 500 berechnen
                scores = GPUColorAndLoss.compute_score(blended, T_target, T_alpha, T_canvas, masks, alphas,chunk_params)

                # Die 500 Scores an die Gesamtliste anhängen
                all_scores.append(scores)

            # 3. Alle Scores zu einem einzigen großen (n_samples,) Tensor zusammenkleben
            final_scores_phase1 = torch.cat(all_scores, dim=0)


        # ==========================================
        # PHASE 2: DER FILTER (Top-K & Gewaltenteilung)
        # ==========================================
        # Wir suchen die negativsten Scores (die größten Verbesserungen!)

        best_scores, best_indices = torch.topk(final_scores_phase1, top_k, largest=False)

        # WICHTIG: Wir frieren Alpha ein!
        # Adam bekommt nur die ersten 5 Parameter (Geometrie) mit requires_grad=True
        best_params = params[best_indices]
        geom_params = torch.cat([best_params[:, 0:2], best_params[:, 3:5]], dim=1).requires_grad_(True)

        fixed_widths = best_params[:, 2:3].clone()
        fixed_alphas = best_params[:, 5:6].clone()  # Das bleibt unangetastet!

        # Kacheln nur für die Elite behalten
        T_target_k, T_canvas_k, T_alpha_k, local_grids_k = OptimizerEngine._extract_tiles(
            best_params, target_img, canvas_img, target_alpha, tile_size
        )

        # ==========================================
        # PHASE 3: DAS SKALPELL (Adam Loop)
        # ==========================================
        optimizer = torch.optim.Adam([geom_params], lr=0.1)
        resolution = target_img.shape[2]
        MAX_SHIFT = 10.0 / resolution
        MAX_SCALE = 0.05
        MAX_ROT = math.radians(10.0)

        for step in range(1, n_mutate + 1):
            optimizer.zero_grad()

            # Zustand vor dem Schritt speichern ---
            params_before = geom_params.clone().detach()

            # Annealing
            progress = (step - 1) / max(1, n_mutate - 1)
            sharpness = 20.0 + (progress**2 * 195.0)

            #Aus 4 Variablen wieder 6 zusammenkleben!
            # geom_params ist: [cx, cy, rh, angle]
            cx_cy = geom_params[:, 0:2]
            rh = geom_params[:, 2:3]
            angle = geom_params[:, 3:4]

            # Zusammenfügen zu [cx, cy, rw, rh, angle, alpha]
            current_params = torch.cat([cx_cy, fixed_widths, rh, angle, fixed_alphas], dim=1)

            # Forward Pass
            sdfs = sdf_function(local_grids_k, current_params)
            masks = torch.sigmoid(-sdfs * sharpness)
            alphas = current_params[:, 5]
            h_x = current_params[:, 3]

            # Farbe berechnen & Mischen
            colors = GPUColorAndLoss.compute_optimal_color(T_target_k, T_canvas_k, masks, alphas,T_alpha_k)
            blended = GPUColorAndLoss.blend_shape(T_canvas_k, colors, masks, alphas)

            # Delta-Loss berechnen
            step_scores = GPUColorAndLoss.compute_score(blended, T_target_k, T_alpha_k, T_canvas_k, masks, alphas,current_params)
            loss = step_scores.mean()

            # Backward & Parameter Update
            loss.backward()
            optimizer.step()
            current_lr = 0.1 - (progress * 0.09)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Harte Grenzen
            with torch.no_grad():
                delta = geom_params - params_before
                # Begrenze die Sprünge für jeden Parameter einzeln
                # geom_params enthält aktuell 5 Werte: [cx, cy, rw, rh, angle]
                delta[:, 0:2].clamp_(-MAX_SHIFT, MAX_SHIFT)  # cx, cy
                delta[:, 2:3].clamp_(-MAX_SCALE, MAX_SCALE)  #rh
                delta[:, 4:5].clamp_(-MAX_ROT, MAX_ROT)  # angle

                geom_params.copy_(params_before + delta)
                # 2. Größe (rh)
                geom_params[:, 2:3].clamp_(min_size, max_size)
                # 3. Winkel (angle) lassen wir unangetastet rotieren

        # ==========================================
        # PHASE 4: DEN SIEGER KRÖNEN
        # ==========================================
        with torch.no_grad():

            cx_cy_final = geom_params[:, 0:2]
            rh_final = geom_params[:, 2:3]
            angle_final = geom_params[:, 3:4]

            final_params = torch.cat([cx_cy_final, fixed_widths, rh_final, angle_final, fixed_alphas], dim=1)

            final_sdfs = sdf_function(local_grids_k, final_params)
            final_masks = torch.sigmoid(-final_sdfs * 100.0)  # Absolute Schärfe

            final_colors = GPUColorAndLoss.compute_optimal_color(T_target_k, T_canvas_k, final_masks,
                                                                 fixed_alphas.squeeze(-1),T_alpha_k)
            final_blended = GPUColorAndLoss.blend_shape(T_canvas_k, final_colors, final_masks, fixed_alphas.squeeze(-1))
            final_scores = GPUColorAndLoss.compute_score(final_blended, T_target_k, T_alpha_k, T_canvas_k, final_masks,
                                                         fixed_alphas.squeeze(-1),final_params)

            # Finde den absoluten Champion
            winner_idx = torch.argmin(final_scores)

            # Gib die 6 Parameter, die 3 Farben und den Score (für den Türsteher) zurück
            return final_params[winner_idx], final_colors[winner_idx], final_scores[winner_idx]

    @staticmethod
    def _extract_tiles(params, target_img, canvas_img, target_alpha, tile_size):
        """
        Der PyTorch Stanz-Automat. Schneidet B Kacheln in einem einzigen Takt aus.
        """
        B = params.shape[0]
        H, W = target_img.shape[1], target_img.shape[2]
        device = params.device

        cx = params[:, 0]
        cy = params[:, 1]

        # grid_sample erwartet Koordinaten von -1.0 bis +1.0 (Clip-Space)
        tx = cx * 2.0 - 1.0
        ty = cy * 2.0 - 1.0

        # Skalierungsfaktor für die Kachel (z.B. 112 / 1024)
        scale = tile_size / H

        # Wir bauen eine affine Transformations-Matrix für jede der B Formen
        # Form: [Scale, 0, Translate_X] und [0, Scale, Translate_Y]
        theta = torch.zeros((B, 2, 3), device=device)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        # 1. Generiere das Kachel-Gitter im Clip-Space (-1 bis +1)
        # align_corners=False ist wichtig für perfekte Pixel-Übereinstimmung
        grid = F.affine_grid(theta, (B, 1, tile_size, tile_size), align_corners=False)

        # 2. Stanze die Bilder aus! (Bilder auf Batch-Size aufblasen)
        target_exp = target_img.unsqueeze(0).expand(B, -1, -1, -1)
        canvas_exp = canvas_img.unsqueeze(0).expand(B, -1, -1, -1)
        alpha_exp = target_alpha.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        # padding_mode='zeros' sorgt dafür, dass Kacheln am Bildrand außen schwarz werden
        T_target = F.grid_sample(target_exp, grid, align_corners=False, padding_mode='zeros')
        T_canvas = F.grid_sample(canvas_exp, grid, align_corners=False, padding_mode='zeros')
        T_alpha = F.grid_sample(alpha_exp, grid, align_corners=False, padding_mode='zeros')

        # 3. Genialer Trick: Wir wandeln das Clip-Space-Gitter zurück in 0.0 bis 1.0
        # Das wird unser 'local_grids' für Modul 1! Es enthält jetzt die ECHTEN
        # globalen Koordinaten jedes einzelnen Pixels in der Kachel.
        local_grids = (grid + 1.0) / 2.0

        return T_target, T_canvas, T_alpha, local_grids