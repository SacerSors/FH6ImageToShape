from typing import Any, Callable

import torch
import torch.nn.functional as F
import math
from sympy.testing.pytest import skip
from torch import Tensor

from GPUColorAndLoss import GPUColorAndLoss
from GPUShapes import GPUShapes


CX, CY, W, H, ROT, SKEW, ALPHA = 0, 1, 2, 3, 4, 5, 6
R, G, B, SCORE, STYPE = 7, 8, 9, 10, 11
class OptimizerEngine:
    """
    Modul 3: Das Herzstück. Sucht, filtert und optimiert Formen im Round-Robin-Verfahren.
    Beherrscht den "smart" (Adam) und "dumb" (Evolutionär) Mode.


    Master Tensor:
    Geometrie & Alpha (0-6):

    0: cx (Center X)
    1: cy (Center Y)
    2: w (Width)
    3: h (Height)
    4: rot (Rotation)
    5: skew (Neu!)
    6: alpha (Deckkraft)

Farbe (7-9):

    7: r, 8: g, 9: b

Typ & Meta (10-11):


    10: score (Loss-Wert)
    11: s_type (0=Ellipse, 1=Rechteck, 2=Dreieck)
    """

    @staticmethod
    def find_best_shape(target_img: torch.Tensor,
                        canvas_img: torch.Tensor,
                        random_samples: torch.Tensor,
                        target_alpha: torch.Tensor,
                        min_size: Tensor,
                        max_size: Tensor,
                        patch_fov_px: Tensor,
                        top_k: int = 64,
                        n_mutate: int = 40,
                        tile_size: int = 112,
                        chunk_size: int = 512,
                        resolution=1024,
                        alpha_base=0.5) -> Tensor:

        with torch.no_grad():
            device = target_img.device
            # ==========================================
            # PHASE 1: Init Shapes and score
            # ==========================================

            N = random_samples.shape[0]
            master_tensor = torch.zeros((N, 12), device=device)
            master_tensor[:, CX:CY+1] = random_samples[:, 0:2]  #Position x,y


            master_tensor[:, W:W+1] = torch.rand((N, 1), device=device) * (max_size[0, 0] - min_size[0, 0]) + min_size[0, 0]  # width
            master_tensor[:, H:H+1] = torch.rand((N, 1), device=device) * (max_size[0, 1] - min_size[0, 1]) + min_size[0, 1]  # height
            master_tensor[:, ROT:ROT+1] = torch.rand((N, 1), device=device) * math.pi  # rotation
            # im moment 0 da nicht implementiert
            master_tensor[:, SKEW:SKEW+1] = 0.0  # skew
            master_tensor[:, ALPHA:ALPHA+1] = torch.rand((N, 1), device=device) * alpha_base + 0.5  # alpha 5
            master_tensor[:, STYPE:STYPE+1] = torch.floor(torch.rand((N, 1), device=device) * 3.0)  # shape 6


            for i in range(0, N, chunk_size):

                T_target, T_canvas, T_alpha, local_grids = OptimizerEngine._extract_tiles(
                    master_tensor[i: i + chunk_size], target_img, canvas_img, target_alpha, tile_size, patch_fov_px
                )

                master_tensor[i: i + chunk_size,R:SCORE+1] = OptimizerEngine.shotgun_score_color(
                    master_tensor[i: i + chunk_size],min_size,max_size, T_target, T_canvas, T_alpha, local_grids
                )

            # ==========================================
            # PHASE 2: DER FILTER (Top-K)
            # ==========================================

            _, best_indices = torch.topk(master_tensor[:, SCORE], top_k, largest=False)
            elites = master_tensor[best_indices]


            T_target_k, T_canvas_k, T_alpha_k, local_grids_k = OptimizerEngine._extract_tiles(
                elites, target_img, canvas_img, target_alpha, tile_size,patch_fov_px
            )

            # ------------------------------------------
            # PHASE 3: Evolutionäre Mutation
            # ------------------------------------------


            n_generations = n_mutate  # Wir nutzen n_mutate als Anzahl der Generationen
            progress_tensor = torch.zeros(1, device=device)
            resolution = target_img.shape[2]


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




            return elites

    @staticmethod
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def shotgun_score_color(chunk_params: torch.Tensor,
                            min_size: Tensor, max_size: Tensor,
                            T_target: torch.Tensor, T_canvas: torch.Tensor,
                            T_alpha: torch.Tensor, local_grids: torch.Tensor) -> torch.Tensor:


        shape_types = chunk_params[:, STYPE].view(-1, 1, 1)

        geom_only = chunk_params[:, :SKEW+1]
        # 1. Wir berechnen alle 3 Mathematiken gleichzeitig (extrem billig auf der GPU)
        sdfs_e = GPUShapes.sdf_ellipse(local_grids, geom_only)
        sdfs_r = GPUShapes.sdf_rectangle(local_grids, geom_only)
        sdfs_t = GPUShapes.sdf_triangle(local_grids, geom_only)

        # 2. Der Multiplexer: Wählt den exakten SDF-Wert basierend auf der Form
        sdfs = torch.where(shape_types == 0, sdfs_e,
               torch.where(shape_types == 1, sdfs_r, sdfs_t))

        masks = (sdfs <= 0.0).float()
        alphas = chunk_params[:, ALPHA]

        colors = GPUColorAndLoss.compute_optimal_color(T_target, T_canvas, masks, alphas, T_alpha)
        blended = GPUColorAndLoss.blend_shape(T_canvas, colors, masks, alphas)
        scores = GPUColorAndLoss.compute_score(blended, T_target, T_alpha, T_canvas, masks, alphas)

        return torch.cat([colors, scores.unsqueeze(1)], dim=1)

    @staticmethod
    @torch.compile(fullgraph=True)
    def _extract_tiles(shapes: torch.Tensor, target_img, canvas_img, target_alpha, tile_size,patch_fov_px):
        B = shapes.shape[0]
        H, W = target_img.shape[1], target_img.shape[2]
        device = shapes.device

        cx = shapes[:, CX]
        cy = shapes[:, CY]

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
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def _evolution_step(elites: torch.Tensor, progress: torch.Tensor, resolution: int,
                        min_size: Tensor, max_size: Tensor,
                        T_target_k: torch.Tensor, T_canvas_k: torch.Tensor,
                        T_alpha_k: torch.Tensor, local_grids_k: torch.Tensor) -> torch.Tensor:

        n_elites = elites.shape[0]
        n_mutants = 32  # Eventuell als Parameter nach oben ziehen
        p_val = progress[0]

        # 2. Amplituden für das Rauschen
        shift_amplitude = (20.0 / resolution) * p_val
        scale_amplitude = 0.10 * p_val
        rot_amplitude = math.radians(30.0) * p_val
        skew_amplitude = 0.10 * p_val
        alpha_amplitude = 0.10 * p_val  # NEU: Alpha wird jetzt auch mutiert!

        # 3. Geometrie (0-6) und Typ (11) der Eltern extrahieren und für Mutanten vervielfältigen
        # base_geom Shape: (n_elites * n_mutants, 7)
        base_geom = elites[:, :ALPHA + 1].unsqueeze(1).expand(n_elites, n_mutants, 7).reshape(-1, 7)
        base_stype = elites[:, STYPE:STYPE + 1].unsqueeze(1).expand(n_elites, n_mutants, 1).reshape(-1, 1)

        # 4. Rauschen erzeugen und anwenden
        noise = torch.rand((n_elites * n_mutants, 7), device=elites.device) * 2.0 - 1.0

        cx_cy = (base_geom[:, CX:CY + 1] + noise[:, 0:2] * shift_amplitude).clamp(0.0, 1.0)
        rw_rh = torch.minimum(torch.maximum(base_geom[:, W:H + 1] + noise[:, 2:4] * scale_amplitude, min_size),
                              max_size)
        angle = base_geom[:, ROT:ROT + 1] + noise[:, 4:5] * rot_amplitude
        skew = base_geom[:, SKEW:SKEW + 1] + noise[:, 5:6] * skew_amplitude
        # Alpha clampen, damit Shapes nicht komplett unsichtbar werden (z.B. min 0.1)
        alpha = base_geom[:, ALPHA:ALPHA + 1]

        # Geometrie zusammenbauen: (K*M, 7)
        mutant_geom = torch.cat([cx_cy, rw_rh, angle, skew, alpha], dim=1)

        # 5. Kacheln (Tiles) für die GPU-Berechnung vervielfältigen
        total_m = n_elites * n_mutants
        tile_size = T_target_k.shape[2]

        T_target_m = T_target_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, 3, tile_size,
                                                                                             tile_size)
        T_canvas_m = T_canvas_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, 3, tile_size,
                                                                                             tile_size)
        T_alpha_m = T_alpha_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, 1, tile_size,
                                                                                           tile_size)
        local_grids_m = local_grids_k.unsqueeze(1).expand(n_elites, n_mutants, -1, -1, -1).reshape(total_m, tile_size,
                                                                                                   tile_size, 2)

        # 6. SDFs berechnen (Mutanten Mathe)
        math_only = mutant_geom[:, :SKEW + 1]  # Nur cx, cy, w, h, rot, skew
        stype_view = base_stype.view(-1, 1, 1)

        sdfs_e = GPUShapes.sdf_ellipse(local_grids_m, math_only)
        sdfs_r = GPUShapes.sdf_rectangle(local_grids_m, math_only)
        sdfs_t = GPUShapes.sdf_triangle(local_grids_m, math_only)

        sdfs = torch.where(stype_view == 0, sdfs_e, torch.where(stype_view == 1, sdfs_r, sdfs_t))
        masks = (sdfs <= 0.0).float()
        mutant_alphas = mutant_geom[:, ALPHA]

        # 7. Farbe und Score berechnen
        mutant_colors = GPUColorAndLoss.compute_optimal_color(T_target_m, T_canvas_m, masks, mutant_alphas, T_alpha_m)
        blended = GPUColorAndLoss.blend_shape(T_canvas_m, mutant_colors, masks, mutant_alphas)

        # Wir bauen kurz einen Dummy-(K*M, 12)-Tensor für die Loss-Funktion,
        # falls compute_score auf bestimmte Spalten zugreifen muss!
        dummy_scores = torch.zeros((total_m, 1), device=elites.device)
        mutant_master = torch.cat([mutant_geom, mutant_colors, dummy_scores, base_stype], dim=1)

        mutant_scores = GPUColorAndLoss.compute_score(blended, T_target_m, T_alpha_m, T_canvas_m, masks, mutant_alphas
                                                      )

        # Echte Scores in Spalte 10 nachtragen
        mutant_master[:, SCORE] = mutant_scores

        # =================================================================
        # 8. DER ELITISMUS-POOL (Jeder Vater tritt gegen seine 32 Kinder an)
        # =================================================================
        # Mutanten von (K*M, 12) zu (K, M, 12) umformen
        mutants_pool = mutant_master.view(n_elites, n_mutants, 12)

        # Eltern von (K, 12) zu (K, 1, 12) umformen
        parents_pool = elites.unsqueeze(1)

        # Zusammenkleben: Jede Zeile K hat jetzt (M + 1) Kandidaten
        competition_pool = torch.cat([mutants_pool, parents_pool], dim=1)

        # Finde den Index des kleinsten Scores in der SCORE-Spalte (10)
        pool_scores = competition_pool[:, :, SCORE]
        best_indices_in_pool = torch.argmin(pool_scores, dim=1)  # Shape: (K,)

        # Die Gewinner-Tensoren extrahieren
        batch_indices = torch.arange(n_elites, device=elites.device)
        winners = competition_pool[batch_indices, best_indices_in_pool, :]  # Shape: (K, 12)

        return winners