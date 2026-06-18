import numpy as np
import torch


def filter_top_shapes(elites_tensor, max_return_shapes=3, filter_hard_limit=None, overlap_margin=0.95):
    """
    Nimmt den (N, 12) Master-Tensor, filtert nach EMA-Score und sortiert
    überlappende Formen mittels Non-Maximum Suppression (NMS) aus.

    overlap_margin: 1.0 bedeutet strikte Berührung. 0.8 erlaubt 20% Überlappung.
    """
    # 1. Sicherstellen, dass wir auf der CPU in Numpy arbeiten
    if isinstance(elites_tensor, torch.Tensor):
        data = elites_tensor.detach().cpu().numpy()
    else:
        data = elites_tensor

    # Spalten-Indizes (zur Sicherheit hart kodiert, entsprechend Master-Tensor)
    CX, CY, W, H = 0, 1, 2, 3
    SCORE = 10

    # 2. Nach Score sortieren (kleinster = bester zuerst)
    sort_idx = np.argsort(data[:, SCORE])
    data = data[sort_idx]

    # 3. Filtern nach Hard-Limit (EMA)
    if filter_hard_limit is not None:
        # Wir behalten nur Formen, deren Score KLEINER oder GLEICH dem Limit ist (kleiner ist besser!)
        valid_mask = data[:, SCORE] <= filter_hard_limit
        data = data[valid_mask]

    # Wenn nach dem EMA-Filter nichts mehr übrig ist, brechen wir direkt ab
    if len(data) == 0:
        return []

    # 4. Non-Maximum Suppression (NMS)
    accepted_shapes = []

    for candidate in data:
        if len(accepted_shapes) >= max_return_shapes:
            break  # Wir haben unser Limit an gleichzeitigen Formen erreicht!

        c_x, c_y = candidate[CX], candidate[CY]
        # Radius näherungsweise aus der größten Ausdehnung bestimmen (sicherste Kollision)
        c_radius = max(candidate[W], candidate[H]) / 2.0

        overlaps = False
        for accepted in accepted_shapes:
            a_x, a_y = accepted[CX], accepted[CY]
            a_radius = max(accepted[W], accepted[H]) / 2.0

            # Euklidische Distanz der Mittelpunkte
            dist = np.sqrt((c_x - a_x) ** 2 + (c_y - a_y) ** 2)

            # Kollisions-Check
            if dist < (c_radius + a_radius) * overlap_margin:
                overlaps = True
                break

        # Wenn sie mit keiner bisher akzeptierten Form kollidiert, nehmen wir sie auf!
        if not overlaps:
            accepted_shapes.append(candidate)

    return accepted_shapes
