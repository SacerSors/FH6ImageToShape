import torch
import torchvision.transforms.functional as TF
from PIL import Image
import lpips
import argparse


class ImageEvaluator:
    def __init__(self, device=None):
        # Wir nutzen AlexNet. Das ist der Standard für LPIPS und extrem schnell.
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Lade LPIPS Modell (AlexNet) auf {self.device}...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        # LPIPS Netzwerk auf Evaluierungs-Modus setzen (kein Training)
        self.lpips_fn.eval()

    def _prepare_image(self, image_path, target_resolution=None):
        """Lädt ein Bild und formatiert es exakt so, wie LPIPS es braucht (-1.0 bis 1.0)"""
        img = Image.open(image_path).convert('RGB')

        if target_resolution:
            img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)

        # TF.to_tensor macht daraus [0.0, 1.0] in der Form (C, H, W)
        img_tensor = TF.to_tensor(img).to(self.device)

        # LPIPS braucht zwingend [-1.0, 1.0] und einen Batch-Dimension (B, C, H, W)
        img_tensor = (img_tensor * 2.0) - 1.0
        return img_tensor.unsqueeze(0)

    def evaluate(self, original_path, generated_path, resolution=None):
        """Berechnet LPIPS und MSE zwischen zwei Bildern."""
        with torch.no_grad():
            img_orig = self._prepare_image(original_path, resolution)
            img_gen = self._prepare_image(generated_path, resolution)

            # 1. LPIPS Score berechnen
            # normalize=True stellt sicher, dass die Tensoren nochmal intern sauber skaliert werden
            lpips_score = self.lpips_fn(img_orig, img_gen, normalize=True).item()

            # 2. MSE (Mean Squared Error) als klassischen Vergleichswert berechnen
            mse_score = torch.nn.functional.mse_loss(img_orig, img_gen).item()

            return lpips_score, mse_score


if __name__ == "__main__":
    # --- Manueller Aufruf über das Terminal ---
    # Beispiel: python evaluate.py original.jpg render.png

    #parser = argparse.ArgumentParser(description="Vergleicht zwei Bilder mit LPIPS und MSE.")
    #parser.add_argument("original", type=str, help="Pfad zum Originalbild (z.B. bilder/frierenHeart.jpg)")
    #parser.add_argument("render", type=str, help="Pfad zum gerenderten Bild (z.B. frierenHeart_vektor_UltraFast.png)")
    #parser.add_argument("--res", type=int, default=1024, help="Auflösung für den Vergleich (Standard: 1024)")

    #args = parser.parse_args()

    IMAGE_PATH = "bilder/frierenHeart.jpg"
    evaluator = ImageEvaluator()
    #lpips_val, mse_val = evaluator.evaluate(args.original, args.render, args.res)
    lpips_score, mse_score = evaluator.evaluate(IMAGE_PATH, "frierenHeart_vektor_UltraFast.png", 512)


    print("\n" + "=" * 40)
    print("🏆 BENCHMARK ERGEBNISSE 🏆")
    print("=" * 40)

    # Bei LPIPS und MSE gilt: Je KLeiner, desto BESSER (0.0 = identisch)
    print(f"LPIPS Score: {lpips_score:.5f}  (Kleiner = Besser)")
    print(f"MSE Score:   {mse_score:.5f}  (Kleiner = Besser)")
    print("=" * 40 + "\n")

    """
    normal

    LPIPS Score: 0.35026  (Kleiner = Besser)

    MSE Score:   0.01753  (Kleiner = Besser)

    

    multishape

    LPIPS Score: 0.33303  (Kleiner = Besser)

    MSE Score:   0.01598  (Kleiner = Besser

    

    bucketBased

    LPIPS Score: 0.36045  (Kleiner = Besser)

    MSE Score:   0.01856  (Kleiner = Besser) 
    """