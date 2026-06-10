import torch


class GPUShapes:
    """
    Reine GPU-Mathe-Klasse für Signed Distance Fields (SDFs).
    Alle Eingaben und Berechnungen laufen strikt im relativen Raum (0.0 bis 1.0).
    """

    @staticmethod
    def create_relative_grid(res_x: int, res_y: int, device: torch.device) -> torch.Tensor:
        """
        Erstellt ein X/Y Koordinatengitter von 0.0 bis 1.0.
        Rückgabe-Shape: (res_y, res_x, 2)
        """
        x = torch.linspace(0.0, 1.0, res_x, device=device)
        y = torch.linspace(0.0, 1.0, res_y, device=device)

        # 'ij' Indexing ist wichtig, damit Y die Zeilen und X die Spalten sind
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        # Wir stapeln X und Y in den letzten Kanal -> (H, W, 2)
        return torch.stack([grid_x, grid_y], dim=-1)

    @staticmethod
    def _transform_space(grid: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, angle_rad: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Verschiebt und rotiert den Raum.
        Akzeptiert globale Grids (H, W, 2) und gechunkte lokale Grids (B, H, W, 2).
        """
        B = cx.shape[0]

        cos_a = torch.cos(angle_rad).view(B, 1, 1)
        sin_a = torch.sin(angle_rad).view(B, 1, 1)

        # NEU: Wir prüfen, ob das Grid 4 Dimensionen hat (gechunkte Kacheln)
        if grid.dim() == 4:
            grid_x = grid[..., 0]  # Shape: (B, H, W)
            grid_y = grid[..., 1]
        else:
            # Fallback für das globale Grid aus unseren ersten Tests
            grid_x = grid[None, ..., 0]  # Shape: (1, H, W)
            grid_y = grid[None, ..., 1]

        # Verschiebung
        dx = grid_x - cx.view(B, 1, 1)
        dy = grid_y - cy.view(B, 1, 1)

        # 2D Rotationsmatrix
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a

        return x_rot, y_rot

    @staticmethod
    def sdf_ellipse(grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Algebraische SDF für eine Ellipse.
        params Shape: (B, 6) -> [cx, cy, rx, ry, angle, alpha]
        Rückgabe Shape: (B, H, W)
        """
        B = params.shape[0]

        # unbind trennt den (B, 6) Tensor in 6 einzelne (B,) Tensoren auf
        cx, cy, rx, ry, angle, _ = params.unbind(dim=-1)

        x_rot, y_rot = GPUShapes._transform_space(grid, cx, cy, angle)

        # Division durch Null verhindern! (Kantenlänge darf minimal 1e-5 sein)
        rx_v = torch.clamp(rx.view(B, 1, 1), min=1e-5)
        ry_v = torch.clamp(ry.view(B, 1, 1), min=1e-5)

        return (x_rot / rx_v) ** 2 + (y_rot / ry_v) ** 2 - 1.0

    @staticmethod
    def sdf_rectangle(grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Exakte euklidische SDF für ein Rechteck (Inigo Quilez Methode).
        params Shape: (B, 6) -> [cx, cy, rx, ry, angle, alpha]
        """
        B = params.shape[0]
        cx, cy, rx, ry, angle, _ = params.unbind(dim=-1)

        x_rot, y_rot = GPUShapes._transform_space(grid, cx, cy, angle)

        rx_v = rx.view(B, 1, 1)
        ry_v = ry.view(B, 1, 1)

        # Absoluter Abstand zum Zentrum minus Halbe Breite/Höhe
        qx = torch.abs(x_rot) - rx_v
        qy = torch.abs(y_rot) - ry_v

        # max(q, 0) für die Distanz im Außenbereich
        max_qx = torch.maximum(qx, torch.zeros_like(qx))
        max_qy = torch.maximum(qy, torch.zeros_like(qy))
        outside_dist = torch.sqrt(
            torch.clamp(qx, min=0.0) ** 2 +
            torch.clamp(qy, min=0.0) ** 2 +
            1e-8
        )

        # min(max(qx, qy), 0) für die Distanz im Innenbereich
        dist_in = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))

        return (outside_dist + dist_in)

    @staticmethod
    def sdf_triangle(grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Normierte SDF für ein gleichschenkliges Dreieck.
        params Shape: (B, 6) -> [cx, cy, base_w, height, angle, alpha]
        """
        B = params.shape[0]
        cx, cy, rw, rh, angle, _ = params.unbind(dim=-1)

        x_rot, y_rot = GPUShapes._transform_space(grid, cx, cy, angle)

        # Auch hier: Null-Divisionen verhindern
        w_v = torch.clamp(rw.view(B, 1, 1), min=1e-5)
        h_v = torch.clamp(rh.view(B, 1, 1), min=1e-5)

        # Normierung: y=0 ist die Spitze, y=1 ist die Bodenkante
        y_norm = (y_rot / h_v) + 0.5

        # x=0 ist die Mitte, x=1 sind die schrägen Außenkanten
        x_norm = torch.abs(x_rot) / (w_v / 2.0)

        # Schnittmenge aus den schrägen Seiten und dem flachen Boden
        d_sides = x_norm - y_norm
        d_bottom = y_norm - 1.0

        return torch.maximum(d_sides, d_bottom)