import torch
import time



# Importiere deine Klassen (Pfade evtl. anpassen)
from GPUShapes import GPUShapes
from GPUColorAndLoss import GPUColorAndLoss
from OptimizerEngine import OptimizerEngine


class TestOptimizerEngine:
    """
    Test- und Benchmark-Klasse für Modul 3.
    Prüft alle Formen und misst die Ausführungszeit für Compiler-Tests.
    """

    @staticmethod
    def run_benchmark(res: int = 128):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Starte Optimizer-Benchmark auf: {device} | Auflösung: {res}x{res}\n")

        # 1. Zielbild generieren: Ein roter Kreis in der Mitte
        grid = GPUShapes.create_relative_grid(res, res, device)
        dx, dy = grid[..., 0] - 0.5, grid[..., 1] - 0.5
        dist = torch.sqrt(dx**2 + dy**2)



        w, h = 0.6, 0.4

        # Prüfe, ob die absolute X- und Y-Distanz innerhalb der halben Kantenlänge liegt
        mask_x = torch.abs(dx) < (w / 2.0)
        mask_y = torch.abs(dy) < (h / 2.0)

        # Beide Bedingungen müssen erfüllt sein (logisches UND)
        target_alpha = torch.logical_and(mask_x, mask_y).float()
        #target_alpha = (dist < 0.3).float()
        target_img = torch.zeros((3, res, res), device=device)
        target_img[0, target_alpha > 0] = 1.0  # Rot einfärben

        canvas_img = torch.zeros_like(target_img)  # Schwarze Leinwand

        # 2. Setup für den Durchlauf
        shapes_names = ["Ellipse", "Rechteck", "Dreieck"]
        results = []

        # 3. Das Turnier: Jede Form bekommt einen Versuch!
        for shape_type in range(3):
            print(f"--- Optimiere Form {shape_type}: {shapes_names[shape_type]} ---")

            # Zeitmessung Start
            start_t = time.time()

            # Engine aufrufen (Wir setzen tile_size=res, da wir hier noch keine Riesenbilder haben)
            params, color, score = OptimizerEngine.find_best_shape(
                target_img=target_img,
                canvas_img=canvas_img,
                target_alpha=target_alpha,
                shape_type=shape_type,
                n_samples=2000,
                top_k=64,
                n_mutate=40,
                tile_size=res,
                chunk_size=500
            )

            # Zeitmessung Ende
            end_t = time.time()
            duration = end_t - start_t

            print(f"Fertig! Dauer: {duration:.3f}s | Delta-Score: {score.item():.4f}")
            print(f"Gefundene Farbe: {[round(c, 2) for c in color.tolist()]}\n")

            # 4. Den Sieger für den Plot auf die Leinwand malen
            with torch.no_grad():
                # Tensoren auf Batch-Size 1 aufblasen für die Funktionen
                p_b = params.unsqueeze(0)
                c_b = color.unsqueeze(0)
                grid_b = grid.unsqueeze(0)

                if shape_type == 0:
                    sdf = GPUShapes.sdf_ellipse(grid_b, p_b)
                elif shape_type == 1:
                    sdf = GPUShapes.sdf_rectangle(grid_b, p_b)
                elif shape_type == 2:
                    sdf = GPUShapes.sdf_triangle(grid_b, p_b)

                mask = torch.sigmoid(-sdf * 100.0)  # Absolute Schärfe
                alpha = p_b[:, 5]

                blended = GPUColorAndLoss.blend_shape(canvas_img.unsqueeze(0), c_b, mask, alpha)
                results.append((blended[0], score.item(), duration))

        # 5. Plotten
        TestOptimizerEngine._plot(target_img, shapes_names, results)

    @staticmethod
    def _plot(target_img, names, results):

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle("OptimizerEngine: Form-Wettbewerb auf Rotes Ziel", fontsize=16)

        axes[0].imshow(target_img.permute(1, 2, 0).cpu().numpy(), origin='lower')
        axes[0].set_title("Zielbild (Target)")
        axes[0].axis('off')

        for i in range(3):
            img, score, duration = results[i]
            ax = axes[i + 1]
            ax.imshow(img.permute(1, 2, 0).cpu().numpy(), origin='lower')
            ax.set_title(f"{names[i]}\nScore: {score:.2f} | {duration:.2f}s")
            ax.axis('off')

        import matplotlib.pyplot as plt  # Fallback für plt.show()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    TestOptimizerEngine.run_benchmark()
    TestOptimizerEngine.run_benchmark()