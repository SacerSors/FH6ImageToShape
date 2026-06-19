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


    @:param random_samples tensor mit (N,5)
            0,1 x,y
            2 bucketID
            3, 4 bucketCenter
    """

    @staticmethod
    def find_best_shape(target_img: torch.Tensor,
                        canvas_img: torch.Tensor,
                        random_samples: torch.Tensor,
                        target_alpha: torch.Tensor,
                        min_size: Tensor,
                        max_size: Tensor,
                        patch_fov_px: Tensor,
                        gpu_package:Tensor,
                        top_k: int = 64,
                        n_mutate: int = 40,
                        tile_size: int = 112,
                        chunk_size: int = 512,
                        resolution=1024,
                        alpha_base=0.5,
                        ) -> Tensor:

        with torch.no_grad():
            device = target_img.device
            # ==========================================
            # PHASE 1: Shotgun Init
            # ==========================================

            N = random_samples.shape[0]
            master_tensor = torch.zeros((N, 12), device=device)
            master_tensor[:, CX:CY+1] = random_samples[:, 0:2]  #Position x,y

            bucket_ids = random_samples[:, 2]
            bucket_centers = random_samples[:, 3:5]

            # Spalten: 5=xmin, 6=xmax, 7=ymin, 8=ymax
            bucket_bounds = random_samples[:, 5:9]

            master_tensor[:, W:W+1] = torch.rand((N, 1), device=device) * (max_size[0, 0] - min_size[0, 0]) + min_size[0, 0]  # width
            master_tensor[:, H:H+1] = torch.rand((N, 1), device=device) * (max_size[0, 1] - min_size[0, 1]) + min_size[0, 1]  # height
            master_tensor[:, ROT:ROT+1] = torch.rand((N, 1), device=device) * math.pi  # rotation
            # im moment 0 da nicht implementiert
            master_tensor[:, SKEW:SKEW+1] = 0.0  # skew
            master_tensor[:, ALPHA:ALPHA+1] = torch.rand((N, 1), device=device) * alpha_base + 0.5  # alpha 5
            master_tensor[:, STYPE:STYPE+1] = torch.floor(torch.rand((N, 1), device=device) * 3.0)  # shape 6




            # ==========================================
            # PHASE 1 & 2 MERGED: Ein Tile pro Bucket!
            # ==========================================
            unique_buckets, inverse_indices = torch.unique(bucket_ids, return_inverse=True)
            num_unique = unique_buckets.shape[0]

            elites_list = []
            bounds_list = []
            centers_list = []

            # 1. Wir holen uns die Zentren für JEDES eindeutige Bucket genau einmal
            # Da die Daten sortiert sind, reicht es, wenn wir das Zentrum des jeweils ersten Samples nehmen
            first_occurrence_indices = torch.zeros(num_unique, dtype=torch.long, device=device)
            first_occurrence_indices.scatter_(0, inverse_indices, torch.arange(N, device=device))
            unique_centers = bucket_centers[first_occurrence_indices]
            unique_bounds = bucket_bounds[first_occurrence_indices]

            # 2. Wir schneiden exakt B Tiles aus (z.B. max 128 Stück auf einmal!)
            # Das ist EIN einziger blitzschneller Aufruf für das gesamte Bild.
            T_target_B, T_canvas_B, T_alpha_B, local_grids_B = OptimizerEngine._extract_tiles(
                master_tensor[:num_unique], target_img, canvas_img, target_alpha, tile_size, patch_fov_px,
                bucket_centers=unique_centers
            )

            T_target_B = T_target_B.clone()
            T_canvas_B = T_canvas_B.clone()
            T_alpha_B = T_alpha_B.clone()
            local_grids_B = local_grids_B.clone()

            # ==========================================
            # DER ULTIMATIVE MERGE: 100% VEKTORISIERT
            # ==========================================
            # 3. Wir weisen die B Kacheln mit inverse_indices allen N Samples zu
            # (Das ist ein einziger Vektor-Befehl, keine for-Schleife!)
            T_target_N = T_target_B[inverse_indices]
            T_canvas_N = T_canvas_B[inverse_indices]
            T_alpha_N = T_alpha_B[inverse_indices]
            local_grids_N = local_grids_B[inverse_indices]

            # 4. EIN EINZIGER GPU AUFRUF für alle (z.B. 2048) Samples!
            scored_shapes = OptimizerEngine.shotgun_score_color(
                master_tensor, min_size, max_size, T_target_N, T_canvas_N, T_alpha_N, local_grids_N
            )
            master_tensor[:, R:SCORE + 1] = scored_shapes

            # 5. Gewinner pro Bucket finden (Einziger winziger CPU-Loop)
            elites_list = []

            for i in range(num_unique):
                # inverse_indices ist bereits 0 bis num_unique-1
                mask = (inverse_indices == i)
                bucket_shapes = master_tensor[mask]

                best_idx = torch.argmin(bucket_shapes[:, SCORE])
                elites_list.append(bucket_shapes[best_idx])

            # 6. Wieder zusammenbauen
            elites = torch.stack(elites_list)

            # WICHTIG: Da elites_list exakt in der Reihenfolge von unique_buckets
            # aufgebaut wurde, können wir die unique-Tensoren direkt übernehmen!
            elite_bounds = unique_bounds
            elite_centers = unique_centers

            # --- Top-K Reduktion für Phase 3 (VRAM Schutz) ---
            if elites.shape[0] > top_k:
                _, global_best_indices = torch.topk(elites[:, SCORE], top_k, largest=False)
                elites = elites[global_best_indices]
                elite_bounds = elite_bounds[global_best_indices]
                elite_centers = elite_centers[global_best_indices]


            # ==========================================
            # SORTING & PADDING (Der VRAM & Compile Schutz)
            # ==========================================
            current_count = elites.shape[0]

            # 1. IMMER global nach Score sortieren (Bester oben)
            _, sorted_idx = torch.sort(elites[:, SCORE])
            elites = elites[sorted_idx]
            elite_bounds = elite_bounds[sorted_idx]
            elite_centers = elite_centers[sorted_idx]

            # 2. Wenn wir zu viele haben -> Abschneiden
            if current_count > top_k:
                elites = elites[:top_k]
                elite_bounds = elite_bounds[:top_k]
                elite_centers = elite_centers[:top_k]

            # 3. Wenn wir zu wenige haben -> Mit den Besten auffüllen (Padding!)
            elif current_count < top_k:
                padding_needed = top_k - current_count

                # Wir nehmen die Indizes von vorne (die Besten) und fangen wieder
                # bei 0 an, falls wir mehr Padding brauchen als wir Elemente haben.
                pad_indices = torch.arange(padding_needed, device=device) % current_count

                # Klone einfach unten drankleben
                elites = torch.cat([elites, elites[pad_indices]], dim=0)
                elite_bounds = torch.cat([elite_bounds, elite_bounds[pad_indices]], dim=0)
                elite_centers = torch.cat([elite_centers, elite_centers[pad_indices]], dim=0)

            T_target_k, T_canvas_k, T_alpha_k, local_grids_k = OptimizerEngine._extract_tiles(
                elites, target_img, canvas_img, target_alpha, tile_size, patch_fov_px, bucket_centers=elite_centers
            )

            T_target_k = T_target_k.clone()
            T_canvas_k = T_canvas_k.clone()
            T_alpha_k = T_alpha_k.clone()
            local_grids_k = local_grids_k.clone()

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
                    local_grids_k=local_grids_k,
                    bounds=elite_bounds,
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
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def _extract_tiles(shapes: torch.Tensor, target_img, canvas_img,
                       target_alpha, tile_size,patch_fov_px,
                       bucket_centers):
        B = shapes.shape[0]
        H, W = target_img.shape[1], target_img.shape[2]
        device = shapes.device

        # grid_sample erwartet Koordinaten von -1.0 bis +1.0
        tx = bucket_centers[:, 0]
        ty = bucket_centers[:, 1]

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
                        T_alpha_k: torch.Tensor, local_grids_k: torch.Tensor,
                        bounds:Tensor) -> torch.Tensor:

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

        raw_cx_cy = base_geom[:, CX:CY + 1] + noise[:, 0:2] * shift_amplitude
        exp_bounds = bounds.unsqueeze(1).expand(n_elites, n_mutants, 4).reshape(-1, 4)
        # ... und sofort ins Gefängnis gesperrt (bounds[:, 0] ist xmin, bounds[:, 1] ist xmax etc.)
        cx = torch.clamp(raw_cx_cy[:, 0], min=exp_bounds[:, 0], max=exp_bounds[:, 1])
        cy = torch.clamp(raw_cx_cy[:, 1], min=exp_bounds[:, 2], max=exp_bounds[:, 3])
        cx_cy = torch.stack([cx, cy], dim=1)

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

        mutant_scores = GPUColorAndLoss.compute_score(blended, T_target_m, T_alpha_m, T_canvas_m, masks, mutant_alphas
                                                      )


        mutant_master = torch.cat([mutant_geom, mutant_colors, mutant_scores.unsqueeze(1), base_stype], dim=1)

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