from typing import Any, Callable

import torch
import torch.nn.functional as F
import math
from torch import Tensor

from GPUColorAndLoss import GPUColorAndLoss
from GPUShapes import GPUShapes

class OptimizerEngine:
    """
    Modul 3: Das Herzstück. Sucht, filtert und optimiert Formen im Round-Robin-Verfahren.
    Beherrscht den "smart" (Adam) und "dumb" (Evolutionär) Mode.
    """

    @staticmethod
    def find_best_shape(target_img: torch.Tensor,
                        canvas_img: torch.Tensor,
                        target_alpha: torch.Tensor,
                        min_size: Tensor,
                        max_size: Tensor,
                        patch_fov_px: Tensor,
                        n_samples: int = 2000,
                        top_k: int = 64,
                        n_mutate: int = 40,
                        tile_size: int = 112,
                        chunk_size: int = 512,
                        heat_map: Tensor = None, #flat
                        resolution=1024,
                        alpha_base=0.5) -> tuple:

        device = target_img.device
        # ==========================================
        # PHASE 1: DIE SCHROTFLINTE (No Grad)
        # ==========================================
        with torch.no_grad():

            if heat_map is None:
                pos = torch.rand((n_samples, 2), device=device)  # Position
            else:
                sampled_indices = torch.multinomial(heat_map, num_samples=n_samples, replacement=True)

                # 4. Indices zurück in X und Y Koordinaten (0.0 bis 1.0) umrechnen
                pos_y = (sampled_indices // resolution).float() / resolution
                pos_x = (sampled_indices % resolution).float() / resolution

                # 5. Zu einem Tensor zusammenfügen
                pos = torch.stack((pos_x, pos_y), dim=1)

            size_w = torch.rand((n_samples, 1), device=device) * (max_size[0, 0] - min_size[0, 0]) + min_size[
                0, 0]  # Spalte 2
            size_h = torch.rand((n_samples, 1), device=device) * (max_size[0, 1] - min_size[0, 1]) + min_size[
                0, 1]  # Spalte 3
            rot = torch.rand((n_samples, 1), device=device) * math.pi  # Spalte 4
            alpha = torch.rand((n_samples, 1), device=device) * alpha_base + 0.5  # Spalte 5
            stype = torch.floor(torch.rand((n_samples, 1), device=device) * 3.0)  # Spalte 6

            # Einmaliges Zusammenfügen
            params = torch.cat([pos, size_w, size_h, rot, alpha, stype], dim=1)

            all_scores = []

            for i in range(0, n_samples, chunk_size):
                chunk_params = params[i: i + chunk_size]

                T_target, T_canvas, T_alpha, local_grids = OptimizerEngine._extract_tiles(
                    chunk_params, target_img, canvas_img, target_alpha, tile_size, patch_fov_px
                )

                scores = OptimizerEngine.shotgun_score(
                    chunk_params,min_size,max_size, T_target, T_canvas, T_alpha, local_grids
                )

                all_scores.append(scores.clone())
            final_scores_phase1 = torch.cat(all_scores, dim=0)

        # ==========================================
        # PHASE 2: DER FILTER (Top-K)
        # ==========================================
        best_scores, best_indices = torch.topk(final_scores_phase1, top_k, largest=False)
        best_params = params[best_indices]


        T_target_k, T_canvas_k, T_alpha_k, local_grids_k = OptimizerEngine._extract_tiles(
            best_params, target_img, canvas_img, target_alpha, tile_size,patch_fov_px
        )

        # ------------------------------------------
        # DUMB MODE (Evolutionäre Mutation)
        # ------------------------------------------
        elites = best_params.clone()

        n_generations = n_mutate  # Wir nutzen n_mutate als Anzahl der Generationen

        progress_tensor = torch.zeros(1, device=device)

        resolution = target_img.shape[2]

        with torch.no_grad():
            for gen in range(1, n_generations + 1):
                progress = 1.0 - (gen - 1) / n_generations
                progress_tensor.fill_(progress)
                # Wir feuern den komplett durchkompilierten Kernel ab!
                elites = OptimizerEngine._evolution_step(
                    elites=elites,
                    progress=progress_tensor,
                    resolution=resolution,
                    min_size=min_size,
                    max_size=max_size,
                    T_target_k=T_target_k,
                    T_canvas_k=T_canvas_k,
                    T_alpha_k=T_alpha_k,
                    local_grids_k=local_grids_k
                ).clone()

        final_elites = elites


        # ==========================================
        # PHASE 4: DEN SIEGER KRÖNEN
        # ==========================================
        with torch.no_grad():

            geom_only_final = final_elites[:, :6]

            shape_types = final_elites[:, 6].view(-1, 1, 1)
            sdfs_e = GPUShapes.sdf_ellipse(local_grids_k, geom_only_final)
            sdfs_r = GPUShapes.sdf_rectangle(local_grids_k, geom_only_final)
            sdfs_t = GPUShapes.sdf_triangle(local_grids_k, geom_only_final)

            final_sdfs = torch.where(shape_types == 0, sdfs_e, torch.where(shape_types == 1, sdfs_r, sdfs_t))
            final_masks = (final_sdfs <= 0.0).float()
            final_alphas = final_elites[:, 5]

            final_colors = GPUColorAndLoss.compute_optimal_color(T_target_k, T_canvas_k, final_masks, final_alphas,
                                                                 T_alpha_k)
            final_blended = GPUColorAndLoss.blend_shape(T_canvas_k, final_colors, final_masks, final_alphas)
            final_scores = GPUColorAndLoss.compute_score(final_blended, T_target_k, T_alpha_k, T_canvas_k, final_masks,
                                                         final_alphas, final_elites)

            winner_idx = torch.argmin(final_scores)

            # Wir geben den ermittelten finalen Shape-Typ einfach mit dem Tensor zurück!
            # Er sitzt in best_params[6]
            return final_elites[winner_idx], final_colors[winner_idx], final_scores[winner_idx]

    @staticmethod
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def shotgun_score(chunk_params: torch.Tensor,
                        min_size: Tensor, max_size: Tensor,
                        T_target: torch.Tensor, T_canvas: torch.Tensor,
                        T_alpha: torch.Tensor, local_grids: torch.Tensor) -> torch.Tensor:

        # Die 7. Spalte ist der Form-Typ (0=Ellipse, 1=Rechteck, 2=Dreieck)
        # .view(-1, 1, 1) macht die Dimensionen kompatibel für das Grid
        shape_types = chunk_params[:, 6].view(-1, 1, 1)

        geom_only = chunk_params[:, :6]
        # 1. Wir berechnen alle 3 Mathematiken gleichzeitig (extrem billig auf der GPU)
        sdfs_e = GPUShapes.sdf_ellipse(local_grids, geom_only)
        sdfs_r = GPUShapes.sdf_rectangle(local_grids, geom_only)
        sdfs_t = GPUShapes.sdf_triangle(local_grids, geom_only)

        # 2. Der Multiplexer: Wählt den exakten SDF-Wert basierend auf Spalte 7!
        sdfs = torch.where(shape_types == 0, sdfs_e,
               torch.where(shape_types == 1, sdfs_r, sdfs_t))

        # 3. Ab hier geht alles seinen normalen Weg
        masks = (sdfs <= 0.0).float()
        alphas = chunk_params[:, 5]

        # Das schwere Heben (Farbe & Loss) passiert nur 1x pro Shape!
        colors = GPUColorAndLoss.compute_optimal_color(T_target, T_canvas, masks, alphas, T_alpha)
        blended = GPUColorAndLoss.blend_shape(T_canvas, colors, masks, alphas)
        scores = GPUColorAndLoss.compute_score(blended, T_target, T_alpha, T_canvas, masks, alphas, chunk_params)

        return scores

    @staticmethod
    @torch.compile(fullgraph=True)
    def _extract_tiles(params, target_img, canvas_img, target_alpha, tile_size,patch_fov_px):
        B = params.shape[0]
        H, W = target_img.shape[1], target_img.shape[2]
        device = params.device

        cx = params[:, 0]
        cy = params[:, 1]

        # grid_sample erwartet Koordinaten von -1.0 bis +1.0
        tx = cx * 2.0 - 1.0
        ty = cy * 2.0 - 1.0

        # 1. Das Sichtfeld (FOV): Wie viel Prozent des Bildes schneiden wir aus?
        scale = patch_fov_px[0] / H

        theta = torch.zeros((B, 2, 3), device=device)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        # 2. Die GPU-Auflösung (tile_sizey): Egal wie groß das Sichtfeld ist,
        # der Tensor wird NIE größer als z.B. 128x128! Das rettet den VRAM.
        grid = F.affine_grid(theta, (B, 1, tile_size, tile_size), align_corners=False)

        target_exp = target_img.unsqueeze(0).expand(B, -1, -1, -1)
        canvas_exp = canvas_img.unsqueeze(0).expand(B, -1, -1, -1)
        alpha_exp = target_alpha.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        T_target = F.grid_sample(target_exp, grid, align_corners=False, padding_mode='zeros')
        T_canvas = F.grid_sample(canvas_exp, grid, align_corners=False, padding_mode='zeros')
        T_alpha = F.grid_sample(alpha_exp, grid, align_corners=False, padding_mode='zeros')

        local_grids = (grid + 1.0) / 2.0

        return T_target, T_canvas, T_alpha, local_grids

    @staticmethod
    @torch.compile(fullgraph=True,  mode="reduce-overhead")
    def _evolution_step(elites: torch.Tensor, progress: torch.Tensor, resolution: int,
                        min_size: Tensor, max_size: Tensor,
                        T_target_k: torch.Tensor, T_canvas_k: torch.Tensor,
                        T_alpha_k: torch.Tensor, local_grids_k: torch.Tensor) -> torch.Tensor:

        n_elites = elites.shape[0]
        n_mutants = 32
        p_val = progress[0]

        shift_amplitude = (20.0 / resolution) * p_val
        scale_amplitude = 0.10 * p_val
        rot_amplitude = math.radians(30.0) * p_val

        # Elite ist jetzt (N, 7)!
        base_params = elites.unsqueeze(1).expand(n_elites, n_mutants, 7).reshape(-1, 7)
        noise = torch.rand((n_elites * n_mutants, 6), device=elites.device) * 2.0 - 1.0

        cx_cy = (base_params[:, 0:2] + noise[:, 0:2] * shift_amplitude).clamp(0.0, 1.0)
        rw_rh_raw = base_params[:, 2:4] + noise[:, 2:4] * scale_amplitude
        rw_rh = torch.minimum(torch.maximum(rw_rh_raw, min_size), max_size)
        angle = base_params[:, 4:5] + noise[:, 4:5] * rot_amplitude
        alpha = base_params[:, 5:6]

        # WICHTIG: Die Mutanten ERBEN die "Spezies" (Rechteck/Ellipse) von ihrem Elite-Elternteil!
        stype = base_params[:, 6:7]

        mutant_params = torch.cat([cx_cy, rw_rh, angle, alpha, stype], dim=1)

        tile_size = T_target_k.shape[2]
        total_m = n_elites * n_mutants

        T_target_m = T_target_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, 3, tile_size, tile_size)
        T_canvas_m = T_canvas_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, 3, tile_size, tile_size)
        T_alpha_m = T_alpha_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, 1, tile_size, tile_size)
        local_grids_m = local_grids_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, tile_size, tile_size, 2)

        # Auch hier der Multiplexer für die Mutanten
        geom_only = mutant_params[:, :6]
        shape_types = mutant_params[:, 6].view(-1, 1, 1)
        sdfs_e = GPUShapes.sdf_ellipse(local_grids_m, geom_only)
        sdfs_r = GPUShapes.sdf_rectangle(local_grids_m, geom_only)
        sdfs_t = GPUShapes.sdf_triangle(local_grids_m, geom_only)

        sdfs = torch.where(shape_types == 0, sdfs_e, torch.where(shape_types == 1, sdfs_r, sdfs_t))

        masks = (sdfs <= 0.0).float()
        alphas = mutant_params[:, 5]

        colors = GPUColorAndLoss.compute_optimal_color(T_target_m, T_canvas_m, masks, alphas, T_alpha_m)
        blended = GPUColorAndLoss.blend_shape(T_canvas_m, colors, masks, alphas)
        mutant_scores = GPUColorAndLoss.compute_score(blended, T_target_m, T_alpha_m, T_canvas_m, masks, alphas, mutant_params)

        score_matrix = mutant_scores.view(n_elites, n_mutants)
        best_mutant_indices = torch.argmin(score_matrix, dim=1)

        row_offsets = torch.arange(n_elites, device=elites.device) * n_mutants
        flat_best_indices = row_offsets + best_mutant_indices

        return mutant_params[flat_best_indices]