import torch
import math
import matplotlib.pyplot as plt

from GPUColorAndLoss import GPUColorAndLoss
from GPUShapes import GPUShapes


# (Stelle sicher, dass GPUShapes und GPUColorAndLoss hier im Code verfügbar sind)

class TestGPUColorAndLoss:
    """
    Test-Klasse für die Farbberechnung und den Fehler-Score (Loss).
    Beweist die Funktion der Least-Squares-Farbe und der Alpha-Strafe.
    """

    @staticmethod
    def run_visual_test_1(res: int = 112):
        device = torch.device('cpu')  # CPU reicht für diesen Test völlig
        print(f"Starte Farb-Test auf: {device} | Kachelgröße: {res}x{res}")

        # 1. Das Gitter erzeugen
        grid = GPUShapes.create_relative_grid(res, res, device)

        # 2. Wir basteln uns ein Zielbild: Ein roter Kreis auf transparentem Grund
        # Abstand zum Zentrum berechnen
        dx, dy = grid[..., 0] - 0.5, grid[..., 1] - 0.5
        dist = torch.sqrt(dx ** 2 + dy ** 2)

        # Alpha-Kanal: 1.0 (Motiv) wenn Radius < 0.25, sonst 0.0 (Hintergrund)
        target_alpha = (dist < 0.25).float()


        # RGB-Bild: Komplett Rot [1.0, 0.0, 0.0]
        target_rgb = torch.zeros((3, res, res), device=device)
        target_rgb[0, :, :] = 1.0

        # Tensoren auf Batch-Size B=2 aufblasen (Wir testen 2 Formen gleichzeitig)
        target_tile = target_rgb.unsqueeze(0).repeat(2, 1, 1, 1)  # (2, 3, 112, 112)
        target_alpha_tile = target_alpha.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)  # (2, 1, 112, 112)
        canvas_tile = torch.zeros_like(target_tile)  # Eine komplett schwarze, leere Leinwand

        # 3. Wir testen zwei Formen:
        # Form 0 (Die Brave): Eine kleine Ellipse, die perfekt im roten Kreis liegt.
        # Form 1 (Die Böse): Ein riesiges Rechteck, das weit in den transparenten Hintergrund ragt.
        params = torch.tensor([
            [0.5, 0.5, 0.15, 0.15, 0.0, 1.0],  # Ellipse
            [0.5, 0.5, 0.40, 0.40, 0.0, 1.0]  # Riesen-Rechteck
        ], device=device)
        alphas = params[:, 5]

        # Masken generieren
        sdf_0 = GPUShapes.sdf_ellipse(grid, params[0:1])
        sdf_1 = GPUShapes.sdf_rectangle(grid, params[1:2])
        sdfs = torch.cat([sdf_0, sdf_1], dim=0)  # Zusammenfügen zu (2, 112, 112)
        masks = torch.sigmoid(-sdfs * 50.0)  # Schärfen

        # -------------------------------------------------------------
        # DIE MAGIE TESTEN
        # -------------------------------------------------------------

        # A) Finde die perfekten Farben!
        colors = GPUColorAndLoss.compute_optimal_color(target_tile, canvas_tile, masks, alphas)

        # B) Zeichne die Formen auf die Leinwand
        blended = GPUColorAndLoss.blend_shape(canvas_tile, colors, masks, alphas)

        # C) Bewerte die Formen!
        scores = GPUColorAndLoss.compute_score(blended, target_tile, target_alpha_tile, masks, alphas)

        # Konsolen-Ausgabe der rohen Zahlen
        print("\n--- ERGEBNISSE DER MATHEMATIK ---")
        print(f"Berechnete Farbe Form 0 (Brave Ellipse):  RGB {colors[0].tolist()}")
        print(f"Berechnete Farbe Form 1 (Böses Rechteck): RGB {colors[1].tolist()}")
        print(f"Score Form 0 (Brave Ellipse):  {scores[0].item():.4f} (Perfekt, fast 0)")
        print(f"Score Form 1 (Böses Rechteck): {scores[1].item():.4f} (Riesig durch Strafe!)")

        # Plotten
        TestGPUColorAndLoss._plot(target_rgb, target_alpha, blended, masks)

    @staticmethod
    def run_visual_test_2(res: int = 112):
        device = torch.device('cpu')
        print(f"Starte Farb-Test auf: {device} | Kachelgröße: {res}x{res}")

        # 1. Das Gitter erzeugen
        grid = GPUShapes.create_relative_grid(res, res, device)

        # 2. NEUES ZIELBILD: Ein roter Kreis auf GRÜNEM Grund
        target_rgb = torch.zeros((3, res, res), device=device)
        target_rgb[1, :, :] = 1.0 # Das ganze Bild ist erst einmal komplett GRÜN

        # Abstand zum Zentrum berechnen für den Kreis
        dx, dy = grid[..., 0] - 0.5, grid[..., 1] - 0.5
        dist = torch.sqrt(dx**2 + dy**2)
        circle_mask = dist < 0.25

        # Kreis rot färben (Rot an, Grün aus)
        target_rgb[0, circle_mask] = 1.0
        target_rgb[1, circle_mask] = 0.0

        # WICHTIG: Alpha-Kanal ist jetzt ÜBERALL 1.0, da es keine Transparenz mehr gibt!
        target_alpha = torch.ones((res, res), device=device)

        # Tensoren auf Batch-Size B=2 aufblasen (Wir testen 2 Formen gleichzeitig)
        target_tile = target_rgb.unsqueeze(0).repeat(2, 1, 1, 1)
        target_alpha_tile = target_alpha.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
        canvas_tile = torch.zeros_like(target_tile) # Schwarze Start-Leinwand

        # 3. Wir testen zwei Formen:
        params = torch.tensor([
            [0.5, 0.5, 0.15, 0.15, 0.0, 1.0], # Form 0: Ellipse (passt perfekt in den Kreis)
            [0.5, 0.5, 0.20, 0.40, 0.0, 1.0]  # Form 1: Rechteck (überlappt Rot und Grün)
        ], device=device)
        alphas = params[:, 5]

        # Masken generieren
        sdf_0 = GPUShapes.sdf_ellipse(grid, params[0:1])
        sdf_1 = GPUShapes.sdf_rectangle(grid, params[1:2])
        sdfs = torch.cat([sdf_0, sdf_1], dim=0)
        masks = torch.sigmoid(-sdfs * 50.0)

        # --- DIE MAGIE TESTEN ---
        colors = GPUColorAndLoss.compute_optimal_color(target_tile, canvas_tile, masks, alphas)
        blended = GPUColorAndLoss.blend_shape(canvas_tile, colors, masks, alphas)
        scores = GPUColorAndLoss.compute_score(blended, target_tile, target_alpha_tile, masks, alphas)

        # Konsolen-Ausgabe
        print("\n--- ERGEBNISSE MIT GRÜNEM HINTERGRUND ---")
        print(f"Farbe Form 0 (Brave Ellipse):  RGB {[round(c, 2) for c in colors[0].tolist()]}")
        print(f"Farbe Form 1 (Böses Rechteck): RGB {[round(c, 2) for c in colors[1].tolist()]}")
        print(f"Score Form 0 (Brave Ellipse):  {scores[0].item():.4f} (Perfekt)")
        print(f"Score Form 1 (Böses Rechteck): {scores[1].item():.4f} (Schlecht, reiner Farbfehler!)")

        # Plotten
        TestGPUColorAndLoss._plot(target_rgb,1, blended, masks)

    @staticmethod
    def _plot(target_rgb, target_alpha, blended, masks):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle("Modul 2 Test: Farbe & Smooth Penalty", fontsize=16)

        # 1. Zielbild visualisieren (wir mischen etwas Grau als Hintergrund dazu für Alpha)
        bg = torch.ones_like(target_rgb) * 0.2
        target_vis = target_rgb * target_alpha + bg * (1.0 - target_alpha)

        axes[0, 0].imshow(target_vis.permute(1, 2, 0).cpu().numpy(), origin='lower')
        axes[0, 0].set_title("Zielbild (Rot) & Alpha-Zone (Grau)")

        # 2. Form 0 (Die Brave)
        axes[0, 1].imshow(masks[0].cpu().numpy(), cmap='gray', origin='lower')
        axes[0, 1].set_title("Maske 0: Brave Ellipse")

        axes[0, 2].imshow(blended[0].permute(1, 2, 0).cpu().numpy(), origin='lower')
        axes[0, 2].set_title("Resultat 0 (Sehr guter Score)")

        # 3. Form 1 (Die Böse)
        axes[1, 0].axis('off')  # Platzhalter

        axes[1, 1].imshow(masks[1].cpu().numpy(), cmap='gray', origin='lower')
        axes[1, 1].set_title("Maske 1: Böses Riesen-Rechteck")

        axes[1, 2].imshow(blended[1].permute(1, 2, 0).cpu().numpy(), origin='lower')
        axes[1, 2].set_title("Resultat 1 (Zerstörter Score!)")

        for ax in axes.flatten():
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    TestGPUColorAndLoss.run_visual_test_1()
    TestGPUColorAndLoss.run_visual_test_2()