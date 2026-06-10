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
    def find_best_shape(target_img: torch.Tensor, canvas_img: torch.Tensor, target_alpha: torch.Tensor,
                        n_samples: int = 2000, top_k: int = 64,
                        n_mutate: int = 40, tile_size: int = 112, chunk_size: int = 512,
                        min_size: float = 0.02, max_size: float = 0.8) -> tuple:

        device = target_img.device
        # ==========================================
        # PHASE 1: DIE SCHROTFLINTE (No Grad)
        # ==========================================
        with torch.no_grad():
            size_range = max_size - min_size
            params = torch.rand((n_samples, 7), device=device)  # Position
            params[:, 2:4] = params[:, 2:4] * size_range + min_size  # Größe
            params[:, 4] = params[:, 4] * math.pi  # Rotation
            params[:, 5] = params[:, 5] * 0.5 + 0.5  # Alpha zwischen 0.5 und 1.0
            params[:, 6] = torch.floor(torch.rand(n_samples, device=device) * 3.0)

            all_scores = []

            for i in range(0, n_samples, chunk_size):
                chunk_params = params[i: i + chunk_size]

                T_target, T_canvas, T_alpha, local_grids = OptimizerEngine._extract_tiles(
                    chunk_params, target_img, canvas_img, target_alpha, tile_size
                )

                scores = OptimizerEngine.shotgun_score(
                    chunk_params, T_target, T_canvas, T_alpha, local_grids
                )

                all_scores.append(scores)
            final_scores_phase1 = torch.cat(all_scores, dim=0)

        # ==========================================
        # PHASE 2: DER FILTER (Top-K)
        # ==========================================
        best_scores, best_indices = torch.topk(final_scores_phase1, top_k, largest=False)
        best_params = params[best_indices]


        T_target_k, T_canvas_k, T_alpha_k, local_grids_k = OptimizerEngine._extract_tiles(
            best_params, target_img, canvas_img, target_alpha, tile_size
        )

        # ------------------------------------------
        # DUMB MODE (Evolutionäre Mutation)
        # ------------------------------------------
        elites = best_params.clone()

        n_generations = n_mutate  # Wir nutzen n_mutate als Anzahl der Generationen

        resolution = target_img.shape[2]

        with torch.no_grad():
            for gen in range(1, n_generations + 1):
                progress = 1.0 - (gen - 1) / n_generations
                progress_tensor = torch.tensor([progress], device=device)
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
                )

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
            final_masks = torch.sigmoid(-final_sdfs * 1000.0)
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
    @torch.compile(fullgraph=True)
    def shotgun_score(chunk_params: torch.Tensor,
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
        masks = torch.sigmoid(-sdfs * 1000.0)
        alphas = chunk_params[:, 5]

        # Das schwere Heben (Farbe & Loss) passiert nur 1x pro Shape!
        colors = GPUColorAndLoss.compute_optimal_color(T_target, T_canvas, masks, alphas, T_alpha)
        blended = GPUColorAndLoss.blend_shape(T_canvas, colors, masks, alphas)
        scores = GPUColorAndLoss.compute_score(blended, T_target, T_alpha, T_canvas, masks, alphas, chunk_params)

        return scores

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
    def _evolution_step(elites: torch.Tensor, progress: torch.Tensor, resolution: int,
                        min_size: float, max_size: float,
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
        rw_rh = (base_params[:, 2:4] + noise[:, 2:4] * scale_amplitude).clamp(min_size, max_size)
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

        masks = torch.sigmoid(-sdfs * 1000.0)
        alphas = mutant_params[:, 5]

        colors = GPUColorAndLoss.compute_optimal_color(T_target_m, T_canvas_m, masks, alphas, T_alpha_m)
        blended = GPUColorAndLoss.blend_shape(T_canvas_m, colors, masks, alphas)
        mutant_scores = GPUColorAndLoss.compute_score(blended, T_target_m, T_alpha_m, T_canvas_m, masks, alphas, mutant_params)

        score_matrix = mutant_scores.view(n_elites, n_mutants)
        best_mutant_indices = torch.argmin(score_matrix, dim=1)

        row_offsets = torch.arange(n_elites, device=elites.device) * n_mutants
        flat_best_indices = row_offsets + best_mutant_indices

        return mutant_params[flat_best_indices]