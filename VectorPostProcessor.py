import torch
import math
import json
import cv2
import numpy as np
from GPUShapes import GPUShapes


class VectorPostProcessor:
    """
    Modul 4: Der Vektor-Gärtner (Standalone).
    Lädt ein JSON, löscht unsichtbare Formen und zeigt das Ergebnis.
    """

    @staticmethod
    def prune_shapes(vector_data: list, resolution: int, threshold_percent: float = 0.05, device: str = "cuda") -> list:
        if not vector_data:
            return []

        torch_device = torch.device(device)
        print(f"\n✂️  Starte Post-Processing für {len(vector_data)} Formen...")

        visibility_map = torch.ones((1, resolution, resolution), device=torch_device)
        grid = GPUShapes.create_relative_grid(resolution, resolution, torch_device).unsqueeze(0)

        pruned_vector_data = []
        total_pixels = resolution * resolution

        for shape in reversed(vector_data):
            params = torch.tensor([
                shape["cx"],
                shape["cy"],
                shape["rw"],
                shape["rh"],
                shape["angle"] / (180.0 / math.pi),
                shape["alpha"]
            ], device=torch_device).unsqueeze(0)

            shape_type = 0 if shape["type"] == "ellipse" else 2

            with torch.no_grad():
                if shape_type == 0:
                    sdfs = GPUShapes.sdf_ellipse(grid, params)
                else:
                    sdfs = GPUShapes.sdf_triangle(grid, params)

                mask = torch.sigmoid(-sdfs * 100.0)
                eff_opacity = mask * shape["alpha"]

                visible_contribution = eff_opacity * visibility_map
                visible_pixel_sum = torch.sum(visible_contribution).item()
                influence_percent = (visible_pixel_sum / total_pixels) * 100.0

                if influence_percent >= threshold_percent:
                    pruned_vector_data.insert(0, shape)
                    visibility_map *= (1.0 - eff_opacity)

        saved_shapes = len(pruned_vector_data)
        deleted_shapes = len(vector_data) - saved_shapes
        print(f"✅ Post-Processing beendet!")
        print(f"   - Formen vorher: {len(vector_data)}")
        print(f"   - Formen gelöscht: {deleted_shapes} ({(deleted_shapes / len(vector_data)) * 100:.1f}%)")
        print(f"   - Formen übrig: {saved_shapes}\n")

        return pruned_vector_data

    @staticmethod
    def visualize_json(vector_data: list, resolution: int = 1024, preview_interval: int = 50, device: str = "cuda"):
        """
        Zeichnet eine Liste von Vektor-Formen auf eine leere Leinwand und zeigt den Fortschritt.
        """
        torch_device = torch.device(device)
        print(f"🎨 Starte Vorschau-Renderer für {len(vector_data)} Formen...")

        # Start-Leinwand (Schwarz)
        canvas_img = torch.zeros((3, resolution, resolution), device=torch_device)
        grid = GPUShapes.create_relative_grid(resolution, resolution, torch_device).unsqueeze(0)

        for i, shape in enumerate(vector_data):
            params = torch.tensor([
                shape["cx"],
                shape["cy"],
                shape["rw"],
                shape["rh"],
                shape["angle"] / (180.0 / math.pi),
                shape["alpha"]
            ], device=torch_device).unsqueeze(0)

            color = torch.tensor(shape["color"], device=torch_device).view(3, 1, 1) / 255.0
            shape_type = 0 if shape["type"] == "ellipse" else 2

            # Form auf die Leinwand brennen
            with torch.no_grad():
                if shape_type == 0:
                    sdfs = GPUShapes.sdf_ellipse(grid, params)
                else:
                    sdfs = GPUShapes.sdf_triangle(grid, params)

                mask = torch.sigmoid(-sdfs * 100.0).squeeze(0)
                effective_alpha = mask * shape["alpha"]

                canvas_img = (color * effective_alpha) + (canvas_img * (1.0 - effective_alpha))

            # Zwischenschritte anzeigen
            if (i + 1) % preview_interval == 0 or i == len(vector_data) - 1:
                np_img = (canvas_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                display_img = cv2.resize(bgr_img, (512, 512), interpolation=cv2.INTER_AREA)

                cv2.imshow("Vector Pruning - Live Preview", display_img)
                cv2.waitKey(1)  # Kurzer Stop, damit das Fenster updatet

        print("🎬 Vorschau beendet. Drücke eine beliebige Taste im Bildfenster, um es zu schließen.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # === EINSTELLUNGEN ===
    INPUT_JSON = "bilder/frieren_vektor_2.json"  # Deine rohe Render-Datei
    OUTPUT_JSON = "frieren_pruned.json"  # Die neue, gesäuberte Datei
    THRESHOLD = 0.0  # %-Einfluss (Höher = mehr Formen werden gelöscht)
    PREVIEW_INTERVAL = 25  # Alle X Formen das Bild updaten
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Rohes JSON laden
    try:
        with open(INPUT_JSON, 'r') as f:
            raw_shapes = json.load(f)
    except FileNotFoundError:
        print(f"❌ Fehler: Datei '{INPUT_JSON}' nicht gefunden!")
        exit()

    # 2. Pruning (Der Gärtner-Lauf)
    cleaned_shapes = VectorPostProcessor.prune_shapes(
        vector_data=raw_shapes,
        resolution=1024,
        threshold_percent=THRESHOLD,
        device=DEVICE
    )

    # 3. Das saubere JSON speichern
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(cleaned_shapes, f, indent=4)
    print(f"💾 Gesäuberte Datei gespeichert unter: {OUTPUT_JSON}")

    # 4. Optional: Die gesäuberten Formen visuell abspielen
    # (Kommentiere diese Zeile aus, wenn du keine Vorschau willst)
    VectorPostProcessor.visualize_json(
        vector_data=cleaned_shapes,
        resolution=1024,
        preview_interval=PREVIEW_INTERVAL,
        device=DEVICE
    )