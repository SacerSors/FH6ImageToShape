import torch


class GPUColorAndLoss:
    """
    Modul B & C: Berechnet die perfekte Farbe und bewertet den Fehler (Loss).
    Arbeitet mit Kacheln (Tiles) im Format (B, C, H, W).
    """

    @staticmethod
    @torch.compile(fullgraph=True)
    def compute_optimal_color(target_tile: torch.Tensor, canvas_tile: torch.Tensor,
                              mask: torch.Tensor, alpha: torch.Tensor,
                              target_alpha_tile: torch.Tensor) -> torch.Tensor:
        B = mask.shape[0]

        eff = mask * alpha.view(B, 1, 1)
        eff_c = eff.unsqueeze(1)

        # Ränder außerhalb des echten Bildes ignorieren
        eff_c = eff_c * target_alpha_tile

        residual = target_tile - (1.0 - eff_c) * canvas_tile

        num = torch.sum(eff_c * residual, dim=(2, 3))
        den = torch.sum(eff_c ** 2, dim=(2, 3))

        # FIX: Clamp statt Addition! Verfälscht die Farben kleiner Formen nicht mehr ins Schwarze.
        den = torch.clamp(den, min=1e-7)

        color = torch.clamp(num / den, 0.0, 1.0)

        return color

    @staticmethod
    @torch.compile(fullgraph=True)
    def blend_shape(canvas_tile: torch.Tensor, color: torch.Tensor,
                    mask: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Mischt die Form mit der gefundenen Farbe auf die aktuelle Leinwand.
        """
        B = mask.shape[0]
        eff_c = (mask * alpha.view(B, 1, 1)).unsqueeze(1)  # (B, 1, H, W)
        color_c = color.view(B, 3, 1, 1)  # (B, 3, 1, 1) für Broadcasting

        return eff_c * color_c + (1.0 - eff_c) * canvas_tile

    @staticmethod
    @torch.compile(fullgraph=True)
    def compute_score(blended_tile, target_tile, target_alpha_tile, canvas_tile, mask, alpha, params):
        B = mask.shape[0]

        # ==========================================================
        # 1. KANTEN-LOSS (Dein bisheriger L2/MSE-Fehler)
        # Super für messerscharfe Kanten und exakte Linien.
        # ==========================================================
        new_mse = torch.mean((blended_tile - target_tile) ** 2, dim=1, keepdim=True)
        old_mse = torch.mean((canvas_tile - target_tile) ** 2, dim=1, keepdim=True)
        mse_delta = new_mse - old_mse
        edge_loss = torch.sum(mse_delta * target_alpha_tile, dim=(1, 2, 3))  # (B,)

        # ==========================================================
        # 2. NEU: FARB-LOSS (L1/MAE-Fehler - Absolut, nicht quadriert!)
        # Reagiert extrem empfindlich auf falsche Farbtöne in großen Flächen.
        # OPTIMIZATION: Avoid wrapping existing tensors with torch.tensor() (e.g. torch.tensor(torch.abs(...)))
        # This prevents breaking the computation graph, detaching gradients, and graph breaks in torch.compile.
        # ==========================================================
        new_l1 = torch.mean(torch.abs(blended_tile - target_tile), dim=1, keepdim=True)
        old_l1 = torch.mean(torch.abs(canvas_tile - target_tile), dim=1, keepdim=True)
        l1_delta = new_l1 - old_l1
        color_loss = torch.sum(l1_delta * target_alpha_tile, dim=(1, 2, 3))  # (B,)

        # ==========================================================
        # 3. STRAFZONE (Rausmalen in den Hintergrund)
        # ==========================================================
        forbidden_zone = 1.0 - target_alpha_tile
        eff_c = (mask * alpha.view(B, 1, 1)).unsqueeze(1)
        spill = eff_c * forbidden_zone
        penalty = torch.sum(spill ** 2, dim=(1, 2, 3)) * 10.0  # (B,)

        # ==========================================================
        # 4. DIE BALANCE (Der Tuning-Regler)
        # Mit dem Faktor vor color_loss (hier 4.0) steuerst du, wie wichtig
        # dem System die richtige Farbe gegenüber den Kanten ist!
        # ==========================================================
        COLOR_WEIGHT = 5.0
        EDGE_WEIGHT = 2

        return (edge_loss * EDGE_WEIGHT) + (color_loss * COLOR_WEIGHT) + penalty