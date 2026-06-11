import torch
import math
import matplotlib.pyplot as plt

from GPUShapes import GPUShapes


# (Hier kopierst du deine GPUShapes Klasse von oben hinein)

class TestGPUShapes:
    """
    Test-Klasse zur visuellen Überprüfung der SDF-Mathematik.
    Generiert Plots für Ellipse, Rechteck und Dreieck.
    """

    @staticmethod
    def run_visual_test(resolution: int = 256, sharpness: float = 50.0):
        # Für den Test reicht die CPU, so brauchst du kein CUDA-Setup
        device = torch.device('cpu')
        print(f"Starte visuellen Test auf: {device} | Auflösung: {resolution}x{resolution}")

        # 1. Gitter erzeugen
        grid = GPUShapes.create_relative_grid(resolution, resolution, device)

        # 2. Test-Parameter definieren: [cx, cy, rx, ry, angle, alpha]
        # Wir drehen die Formen ein wenig, um die Rotationsmatrix zu testen
        params_ellipse = torch.tensor([[0.5, 0.5, 0.3, 0.15, math.radians(70), 1.0]], device=device)
        params_rect = torch.tensor([[0.5, 0.5, 0.25, 0.15, math.radians(-35), 1.0]], device=device)
        params_tri = torch.tensor([[0.5, 0.5, 0.4, 0.4, math.radians(45), 1.0]], device=device)

        # 3. Rohe SDFs berechnen
        sdf_ellipse = GPUShapes.sdf_ellipse(grid, params_ellipse)[0]  # [0] holt den ersten Batch
        sdf_rect = GPUShapes.sdf_rectangle(grid, params_rect)[0]
        sdf_tri = GPUShapes.sdf_triangle(grid, params_tri)[0]

        # 4. Masken via SDF generieren (mit negativer SDF, damit innen = 1.0)
        mask_ellipse = (sdf_ellipse <= 0.0).float()
        mask_rect = (sdf_rect <= 0.0).float()
        mask_tri = (sdf_tri <= 0.0).float()

        # 5. Plotten mit Matplotlib
        TestGPUShapes._plot_results(
            [sdf_ellipse, sdf_rect, sdf_tri],
            [mask_ellipse, mask_rect, mask_tri],
            ["Ellipse", "Rechteck", "Dreieck"],
            sharpness
        )

    @staticmethod
    def _plot_results(sdfs, masks, titles,sharpness):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("GPUShapes: SDF vs. Sigmoid Maske", fontsize=16)

        for i in range(3):
            # Obere Reihe: Rohes SDF (Heatmap)
            # Wir nutzen cmap='RdBu', damit negative Werte (innen) blau und positive (außen) rot sind
            ax_sdf = axes[0, i]
            im_sdf = ax_sdf.imshow(sdfs[i].cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5, origin='lower')
            ax_sdf.set_title(f"{titles[i]} - Rohes SDF")
            ax_sdf.axis('off')
            fig.colorbar(im_sdf, ax=ax_sdf, fraction=0.046, pad=0.04)

            # Untere Reihe: Sigmoid Maske (Schwarz/Weiß)
            ax_mask = axes[1, i]
            im_mask = ax_mask.imshow(masks[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
            ax_mask.set_title(f"{titles[i]} - Maske (Schärfe={sharpness})")
            ax_mask.axis('off')
            fig.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    TestGPUShapes.run_visual_test(512, 100)