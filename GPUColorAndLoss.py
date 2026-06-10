import torch


class GPUColorAndLoss:
    """
    Modul B & C: Berechnet die perfekte Farbe und bewertet den Fehler (Loss).
    Arbeitet mit Kacheln (Tiles) im Format (B, C, H, W).
    """

    @staticmethod
    def compute_optimal_color(target_tile: torch.Tensor, canvas_tile: torch.Tensor,
                              mask: torch.Tensor, alpha: torch.Tensor
                              ,target_alpha_tile: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die analytisch perfekte (R,G,B) Farbe für B Formen gleichzeitig.
        target_tile / canvas_tile Shape: (B, 3, H, W)
        mask Shape: (B, H, W)
        alpha Shape: (B,)
        """
        B = mask.shape[0]

        # 1. Effektive Deckkraft (SDF-Maske * Transparenz-Parameter)
        eff = mask * alpha.view(B, 1, 1)  # Shape: (B, H, W)

        # 2. Die Kanal-Dimension hinzufügen, um mit RGB-Bildern zu rechnen
        eff_c = eff.unsqueeze(1)  # Shape: (B, 1, H, W)
        eff_c = eff_c * target_alpha_tile
        # 3. Least Squares: Was fehlt noch auf der Leinwand, um das Ziel zu erreichen?
        residual = target_tile - (1.0 - eff_c) * canvas_tile

        # 4. Zähler und Nenner über die Pixel (H, W) aufsummieren
        num = torch.sum(eff_c * residual, dim=(2, 3))  # Shape: (B, 3)
        den = torch.sum(eff_c ** 2, dim=(2, 3)) + 1e-5  # Shape: (B, 1)

        # 5. Farbe berechnen und auf gültige Werte [0.0, 1.0] begrenzen
        color = torch.clamp(num / den, 0.0, 1.0)  # Shape: (B, 3)

        return color

    @staticmethod
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
    def compute_score(blended_tile, target_tile, target_alpha_tile, canvas_tile, mask, alpha,params):
        B = mask.shape[0]

        # 1. Der Fehler der NEUEN Leinwand (mit Form)
        new_mse = torch.mean((blended_tile - target_tile) ** 2, dim=1, keepdim=True)

        # 2. Der Fehler der ALTEN Leinwand (ohne Form)
        old_mse = torch.mean((canvas_tile - target_tile) ** 2, dim=1, keepdim=True)

        # 3. Das DELTA (Neu minus Alt). Negative Werte bedeuten Verbesserung!
        mse = new_mse - old_mse

        # 2. Fehler nur dort werten, wo das Zielbild existiert (Ziel-Alpha > 0)
        # sum(dim=(1,2,3)) fasst das ganze Kachel-Bild zu EINER Zahl pro Form zusammen
        valid_loss = torch.sum(mse * target_alpha_tile, dim=(1, 2, 3))  # (B,)

        # 3. SMOOTH PENALTY (Strafe fürs Rausmalen in den transparenten Hintergrund)
        forbidden_zone = 1.0 - target_alpha_tile  # 1.0 im Hintergrund, 0.0 im Motiv
        eff_c = (mask * alpha.view(B, 1, 1)).unsqueeze(1)  # (B, 1, H, W)

        # Wie viel Farbe ist im verbotenen Bereich gelandet?
        spill = eff_c * forbidden_zone

        # Strafe: Quadriert (kleine Fehler = okay, große = massiv bestraft) * Gewichtung
        penalty = torch.sum(spill ** 2, dim=(1, 2, 3)) * 30.0  # (B,)

        # Der finale Score für den Adam Optimizer
        total_score = valid_loss + penalty


        return total_score