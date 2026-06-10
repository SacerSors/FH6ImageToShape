import torch
import torch.nn.functional as F
import math

from GPUColorAndLoss import GPUColorAndLoss
from GPUShapes import GPUShapes

class OptimizerEngine:
    """
    Modul 3: Das Herzstück. Sucht, filtert und optimiert Formen im Round-Robin-Verfahren.
    Beherrscht den "smart" (Adam) und "dumb" (Evolutionär) Mode.
    """

    @staticmethod
    def find_best_shape(target_img: torch.Tensor, canvas_img: torch.Tensor, target_alpha: torch.Tensor,
                        shape_type: int, n_samples: int = 2000, top_k: int = 64,
                        n_mutate: int = 40, tile_size: int = 112, chunk_size: int = 512,
                        min_size: float = 0.02, max_size: float = 0.8,
                        optimizer_mode: str = "smart") -> tuple:

        device = target_img.device

        # 1. FUNKTIONS-ZEIGER
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
            size_range = max_size - min_size
            params = torch.rand((n_samples, 6), device=device)  # Position
            params[:, 2:4] = params[:, 2:4] * size_range + min_size  # Größe
            params[:, 4] = params[:, 4] * math.pi  # Rotation
            params[:, 5] = params[:, 5] * 0.5 + 0.5  # Alpha zwischen 0.5 und 1.0

            all_scores = []

            for i in range(0, n_samples, chunk_size):
                chunk_params = params[i: i + chunk_size]

                T_target, T_canvas, T_alpha, local_grids = OptimizerEngine._extract_tiles(
                    chunk_params, target_img, canvas_img, target_alpha, tile_size
                )

                sdfs = sdf_function(local_grids, chunk_params)
                masks = torch.sigmoid(-sdfs * 1000.0)
                alphas = chunk_params[:, 5]

                colors = GPUColorAndLoss.compute_optimal_color(T_target, T_canvas, masks, alphas, T_alpha)
                blended = GPUColorAndLoss.blend_shape(T_canvas, colors, masks, alphas)
                scores = GPUColorAndLoss.compute_score(blended, T_target, T_alpha, T_canvas, masks, alphas,
                                                       chunk_params)

                all_scores.append(scores)

            final_scores_phase1 = torch.cat(all_scores, dim=0)

        # ==========================================
        # PHASE 2: DER FILTER (Top-K)
        # ==========================================
        best_scores, best_indices = torch.topk(final_scores_phase1, top_k, largest=False)
        best_params = params[best_indices]

        # Für den Smart-Mode trennen wir die Breiten und Alphas ab
        fixed_widths = best_params[:, 2:3].clone()
        fixed_alphas = best_params[:, 5:6].clone()

        T_target_k, T_canvas_k, T_alpha_k, local_grids_k = OptimizerEngine._extract_tiles(
            best_params, target_img, canvas_img, target_alpha, tile_size
        )

        # ==========================================
        # PHASE 3: DER WEICHENSTELLER (Smart vs. Dumb)
        # ==========================================
        if optimizer_mode == "smart":
            # ------------------------------------------
            # SMART MODE (Adam Optimizer)
            # ------------------------------------------
            geom_params = torch.cat([best_params[:, 0:2], best_params[:, 3:5]], dim=1).requires_grad_(True)
            optimizer = torch.optim.Adam([geom_params], lr=0.1)
            resolution = target_img.shape[2]

            MAX_STEP_SHIFT = 10.0 / resolution
            MAX_TOTAL_TRAVEL = 16.0 / resolution
            MAX_SCALE = 0.05
            MAX_ROT = math.radians(10.0)

            anchor_positions = geom_params[:, 0:2].clone().detach()

            for step in range(1, n_mutate + 1):
                optimizer.zero_grad()
                params_before = geom_params.clone().detach()

                progress = (step - 1) / max(1, n_mutate - 1)
                sharpness = 40.0 + (progress ** 2 * 195.0)

                cx_cy = geom_params[:, 0:2]
                rh = geom_params[:, 2:3]
                angle = geom_params[:, 3:4]
                current_params = torch.cat([cx_cy, fixed_widths, rh, angle, fixed_alphas], dim=1)

                sdfs = sdf_function(local_grids_k, current_params)
                masks = torch.sigmoid(-sdfs * sharpness)
                alphas = current_params[:, 5]

                colors = GPUColorAndLoss.compute_optimal_color(T_target_k, T_canvas_k, masks, alphas, T_alpha_k)
                blended = GPUColorAndLoss.blend_shape(T_canvas_k, colors, masks, alphas)
                step_scores = GPUColorAndLoss.compute_score(blended, T_target_k, T_alpha_k, T_canvas_k, masks,
                                                            alphas, current_params)

                loss = step_scores.mean()
                loss.backward()
                optimizer.step()

                current_lr = 0.1 - (progress * 0.09)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                with torch.no_grad():
                    delta = geom_params - params_before
                    delta[:, 0:2].clamp_(-MAX_STEP_SHIFT, MAX_STEP_SHIFT)
                    delta[:, 2:3].clamp_(-MAX_SCALE, MAX_SCALE)
                    delta[:, 4:5].clamp_(-MAX_ROT, MAX_ROT)
                    geom_params.copy_(params_before + delta)

                    geom_params[:, 0:2] = torch.max(
                        torch.min(geom_params[:, 0:2], anchor_positions + MAX_TOTAL_TRAVEL),
                        anchor_positions - MAX_TOTAL_TRAVEL
                    )
                    geom_params[:, 0:2].clamp_(0.0, 1.0)
                    geom_params[:, 2:3].clamp_(min_size, max_size)

            # Zusammenkleben für Phase 4
            with torch.no_grad():
                final_elites = torch.cat(
                    [geom_params[:, 0:2], fixed_widths, geom_params[:, 2:3], geom_params[:, 3:4], fixed_alphas],
                    dim=1)

        elif optimizer_mode == "dumb":
            # ------------------------------------------
            # DUMB MODE (Evolutionäre Mutation)
            # ------------------------------------------
            elites = best_params.clone()
            n_elites = top_k
            n_mutants = 32
            n_generations = n_mutate  # Wir nutzen n_mutate als Anzahl der Generationen

            resolution = target_img.shape[2]
            sharpness = 1000.0  # Absolute Härte für den Dumb-Mode

            with torch.no_grad():
                for gen in range(1, n_generations + 1):
                    progress = 1.0 - (gen - 1) / n_generations

                    # Wir feuern den komplett durchkompilierten Kernel ab!
                    elites = OptimizerEngine._evolution_step(
                        elites=elites,
                        progress=progress,
                        resolution=resolution,
                        min_size=min_size,
                        max_size=max_size,
                        shape_type=shape_type,
                        T_target_k=T_target_k,
                        T_canvas_k=T_canvas_k,
                        T_alpha_k=T_alpha_k,
                        local_grids_k=local_grids_k
                    )

            final_elites = elites

        else:
            raise ValueError(f"Unbekannter Optimizer-Mode: {optimizer_mode}. Erlaubt sind 'smart' oder 'dumb'.")

        # ==========================================
        # PHASE 4: DEN SIEGER KRÖNEN
        # ==========================================
        with torch.no_grad():
            final_sdfs = sdf_function(local_grids_k, final_elites)
            final_masks = torch.sigmoid(-final_sdfs * 1000.0)  # Absolute Schärfe
            final_alphas = final_elites[:, 5]

            final_colors = GPUColorAndLoss.compute_optimal_color(T_target_k, T_canvas_k, final_masks, final_alphas,
                                                                 T_alpha_k)
            final_blended = GPUColorAndLoss.blend_shape(T_canvas_k, final_colors, final_masks, final_alphas)
            final_scores = GPUColorAndLoss.compute_score(final_blended, T_target_k, T_alpha_k, T_canvas_k,
                                                         final_masks, final_alphas, final_elites)

            winner_idx = torch.argmin(final_scores)

            return final_elites[winner_idx], final_colors[winner_idx], final_scores[winner_idx]
    @staticmethod
    @torch.compile(fullgraph=True)
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

    @staticmethod
    @torch.compile(fullgraph=True)
    def _evolution_step(elites: torch.Tensor, progress: float, resolution: int,
                        min_size: float, max_size: float, shape_type: int,
                        T_target_k: torch.Tensor, T_canvas_k: torch.Tensor,
                        T_alpha_k: torch.Tensor, local_grids_k: torch.Tensor) -> torch.Tensor:
        """
        Kompilierter Turbo-Kernel für EINE Generation.
        Erzeugt Mutanten, bewertet sie und gibt die neuen Eliten zurück.
        """
        device = elites.device
        n_elites = elites.shape[0]
        n_mutants = 32
        sharpness = 1000.0  # Absolute Schärfe

        # 1. Mutations-Amplituden berechnen
        shift_amplitude = (20.0 / resolution) * progress
        scale_amplitude = 0.10 * progress
        rot_amplitude = math.radians(30.0) * progress

        # 2. Eliten aufblasen und Rauschen addieren
        mutant_params = elites.unsqueeze(1).expand(n_elites, n_mutants, 6).clone().view(-1, 6)
        noise = torch.rand((n_elites * n_mutants, 6), device=device) * 2.0 - 1.0

        mutant_params[:, 0:2] += noise[:, 0:2] * shift_amplitude
        mutant_params[:, 2:4] += noise[:, 2:4] * scale_amplitude
        mutant_params[:, 4] += noise[:, 4] * rot_amplitude

        # 3. Grenzen sichern
        mutant_params[:, 0:2].clamp_(0.0, 1.0)
        mutant_params[:, 2:4].clamp_(min_size, max_size)

        # 4. Kacheln aufblasen
        tile_size = T_target_k.shape[2]
        T_target_m = T_target_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(-1, 3, tile_size,
                                                                                             tile_size)
        T_canvas_m = T_canvas_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(-1, 3, tile_size,
                                                                                             tile_size)
        T_alpha_m = T_alpha_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(-1, 1, tile_size, tile_size)
        local_grids_m = local_grids_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(-1, tile_size,
                                                                                                   tile_size, 2)

        # 5. SDFs berechnen (Direkter Aufruf für perfekten Compiler-Fluss)
        if shape_type == 0:
            sdfs = GPUShapes.sdf_ellipse(local_grids_m, mutant_params)
        elif shape_type == 1:
            sdfs = GPUShapes.sdf_rectangle(local_grids_m, mutant_params)
        else:
            sdfs = GPUShapes.sdf_triangle(local_grids_m, mutant_params)

        # 6. Bewertung
        masks = torch.sigmoid(-sdfs * sharpness)
        alphas = mutant_params[:, 5]

        colors = GPUColorAndLoss.compute_optimal_color(T_target_m, T_canvas_m, masks, alphas, T_alpha_m)
        blended = GPUColorAndLoss.blend_shape(T_canvas_m, colors, masks, alphas)
        mutant_scores = GPUColorAndLoss.compute_score(blended, T_target_m, T_alpha_m, T_canvas_m, masks, alphas,
                                                      mutant_params)

        # 7. Die besten Kinder auswählen
        score_matrix = mutant_scores.view(n_elites, n_mutants)
        best_mutant_indices = torch.argmin(score_matrix, dim=1)

        row_offsets = torch.arange(n_elites, device=device) * n_mutants
        flat_best_indices = row_offsets + best_mutant_indices

        return mutant_params[flat_best_indices]