import json
import math
import matplotlib.pyplot as plt


def analyze_render(json_file="telemetry.json", resolution=2048, spaghetti_threshold=0.2):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Fehler: {json_file} nicht gefunden. Lass erst den Renderer laufen!")
        return

    accepted = data.get("accepted", [])
    rejected = data.get("rejected", [])

    if not accepted:
        print("Keine akzeptierten Formen in der Telemetrie gefunden.")
        return

    # --- Daten aufbereiten (Akzeptierte Formen) ---
    ids = list(range(1, len(accepted) + 1))

    # Fläche aus Breite und Höhe der Form berechnen (w * h * pi)
    flaechen = [float(d["geometry"][2] * d["geometry"][3] * math.pi) for d in accepted]
    pinsel_max = [d["pinsel_max"] for d in accepted]

    # -----------------------------------------------------------------
    # NEU: Die Gabelung der Limits (Normal vs. Spaghetti)
    # -----------------------------------------------------------------
    max_areas = []
    min_areas_normal = []
    min_areas_spaghetti = []

    for p in pinsel_max:
        # 1. Maximal erlaubte Fläche (w=p, h=p)
        max_areas.append(p * p * math.pi)

        # Basis-Werte der Engine
        chunky = p * 0.33
        gpu_safe = 2.0 / resolution

        # 2. Minimal-Fläche für NORMALE Formen (Beide Seiten chunky)
        w_norm = min(chunky, p * 0.5)
        h_norm = min(chunky, p * 0.5)
        min_areas_normal.append(w_norm * h_norm * math.pi)

        # 3. Minimal-Fläche für SPAGHETTI Formen (Eine Seite chunky, eine Seite gpu_safe)
        if p > spaghetti_threshold:
            w_spag = min(chunky, p * 0.5)
            h_spag = min(chunky, p * 0.5)
        else:
            w_spag = min(chunky, p * 0.5)
            h_spag = min(gpu_safe, p * 0.5)

        min_areas_spaghetti.append(w_spag * h_spag * math.pi)

    # Spaghetti-Faktor: Aspektverhältnis (Max Länge / Min Dicke)
    spaghetti = []
    for d in accepted:
        w, h = d["geometry"][2], d["geometry"][3]
        ratio = max(w, h) / max(min(w, h), 1e-5)
        spaghetti.append(ratio)

    scores = [d["score"] for d in accepted]
    emas = [d["ema"] for d in accepted]

    # --- Daten aufbereiten (Abgelehnte Formen) ---
    rej_scores = [r[0] for r in rejected]
    rej_emas = [r[1] if r[1] is not None else 0 for r in rejected]
    rej_ids = list(range(1, len(rejected) + 1))

    # --- Dashboard Setup ---
    fig, axs = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle('🧬 Vector Renderer - Deep Research Dashboard', fontsize=18, fontweight='bold')

    # =================================================================
    # Graph 1: Flächeninhalt & Pinselgröße (Jetzt mit Gabelung!)
    # =================================================================
    axs[0].plot(ids, max_areas, color='crimson', linestyle='-', linewidth=2, label='Maximal erlaubte Fläche')

    # Die Gabelung zeichnen!
    axs[0].plot(ids, min_areas_normal, color='dodgerblue', linestyle='--', linewidth=2,
                label='Min Fläche (Normale Form)')
    axs[0].plot(ids, min_areas_spaghetti, color='cyan', linestyle='-', linewidth=2,
                label='Min Fläche (Spaghetti Limit)')

    sc = axs[0].scatter(ids, flaechen, c=pinsel_max, cmap='viridis', alpha=0.8, s=15, zorder=3,
                        label='Gezeichnete Form')
    axs[0].set_yscale('log')
    axs[0].set_title('1. Flächeninhalt der Formen im Pinsel-Trichter')
    axs[0].set_ylabel('Fläche (Log)')
    axs[0].grid(True, which="both", ls="--", alpha=0.3)
    axs[0].legend(loc="lower left")
    plt.colorbar(sc, ax=axs[0], label='Pinsel-Limit (Max_S)')

    # =================================================================
    # Graph 2: Der Spaghetti-Faktor (Asymmetrie)
    # =================================================================
    spaghetti_ratio_threshold = 3.0
    is_spaghetti = sum(1 for s in spaghetti if s > spaghetti_ratio_threshold)
    spaghetti_ratio = (is_spaghetti / len(spaghetti)) * 100

    axs[1].scatter(ids, spaghetti, color='darkorange', alpha=0.6, s=12)
    axs[1].axhline(y=spaghetti_ratio_threshold, color='red', linestyle='-', alpha=0.5,
                   label='Spaghetti-Grenze (Ratio > 3)')
    axs[1].set_title(f'2. Spaghetti-Faktor ({spaghetti_ratio:.1f}% des Bildes sind asymmetrische Linien)')
    axs[1].set_ylabel('Länge / Dicke')
    axs[1].set_ylim(0, min(max(spaghetti) + 2, 50))
    axs[1].legend()
    axs[1].grid(True, ls="--", alpha=0.5)

    # =================================================================
    # Graph 3: Erfolgreiche Scores & EMA
    # =================================================================
    axs[2].plot(ids, scores, label='Echter Score', color='royalblue', alpha=0.6, linewidth=1.5)
    axs[2].plot(ids, emas, label='EMA Filter', color='crimson', linewidth=2.5)
    axs[2].set_title('3. Score & EMA (Der Erfolgsweg)')
    axs[2].set_ylabel('L1 Fehler Score')
    axs[2].invert_yaxis()
    axs[2].legend()
    axs[2].grid(True, ls="--", alpha=0.5)

    # =================================================================
    # Graph 4: Müllverbrennung (Die abgelehnten Scores)
    # =================================================================
    if len(rejected) > 0:
        axs[3].scatter(rej_ids, rej_scores, color='gray', alpha=0.3, s=5, label='Abgelehnt')
        axs[3].plot(rej_ids, rej_emas, color='red', linewidth=1.5, alpha=0.8, label='EMA (Die Wand)')
        axs[3].set_title(f'4. Die {len(rejected)} Fehlversuche (Wann rannte die Engine gegen die Wand?)')
        axs[3].set_xlabel('Fehlversuche (Global)')
        axs[3].set_ylabel('L1 Fehler Score')
        axs[3].invert_yaxis()
        axs[3].legend()
        axs[3].grid(True, ls="--", alpha=0.5)
    else:
        axs[3].set_title('4. Keine Fehlversuche aufgezeichnet.')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


if __name__ == "__main__":
    analyze_render("telemetry.json")