from enum import Enum

class RenderPreset(Enum):
    # --- LEVEL 1: ULTRA FAST (Proof of Concept) ---
    # Reine Funktionsprüfung, extrem ungenau aber rasant
    ULTRA_FAST = {
        "patience_factor": 0.50,
        "MAX_BAD_SCORES": 10,
        "ema_positiv_reaction": 0.30,
        "ema_negativ_reaction": 0.05,
        "pinsel_steilheit": 0.40,      # Pinsel schrumpft sofort für extremen Speed
        "n_mutate": 8,
        "top_k": 4,
        "sample_multi": 1,
        "batch_multi": 1,
        "resolution": 512
    }

    # --- LEVEL 2: FAST ---
    # Bereits nutzbar, neigt aber zu leichten Flecken/Mängeln
    FAST = {
        "patience_factor": 0.60,
        "MAX_BAD_SCORES": 15,
        "ema_positiv_reaction": 0.25,
        "ema_negativ_reaction": 0.02,
        "pinsel_steilheit": 0.50,
        "n_mutate": 16,
        "top_k": 12,
        "sample_multi": 2,
        "batch_multi": 1,
        "resolution": 1024
    }

    # --- LEVEL 3: NORMAL (Der Sweet-Spot) ---
    # Optimal für den täglichen Gebrauch im Auto-Modus. Schnell & brillant.
    NORMAL = {
        "patience_factor": 0.65,
        "MAX_BAD_SCORES": 20,
        "ema_positiv_reaction": 0.20,
        "ema_negativ_reaction": 0.01,
        "pinsel_steilheit": 0.60,      # Deine gesunde Standard-Kurve
        "n_mutate": 32,
        "top_k": 24,
        "sample_multi": 4,
        "batch_multi": 2,
        "resolution": 2048
    }

    # --- LEVEL 4: HOCH ---
    # Für anspruchsvolle Bilder (wie Frierens Haare) und große Displays
    HOCH = {
        "patience_factor": 0.70,
        "MAX_BAD_SCORES": 30,
        "ema_positiv_reaction": 0.15,
        "ema_negativ_reaction": 0.01,
        "pinsel_steilheit": 0.70,      # Mehr Freiheit nach oben im Mittelbau
        "n_mutate": 64,
        "top_k": 48,
        "sample_multi": 8,
        "batch_multi": 4,
        "resolution": 2048
    }

    # --- LEVEL 5: ULTRA (Qualität >> Zeit) ---
    # Der ultimative Run. Der EMA zwingt die Engine zur absoluten Perfektion.
    ULTRA = {
        "patience_factor": 0.75,
        "MAX_BAD_SCORES": 50,          # Hohe Ausdauer für die harten Details
        "ema_positiv_reaction": 0.10,  # Sehr sturer EMA, verlangt konstante Höchstleistung
        "ema_negativ_reaction": 0.005,
        "pinsel_steilheit": 0.80,      # Maximale Freiheit nach oben ("Luft nach oben")
        "n_mutate": 80,
        "top_k": 64,
        "sample_multi": 10,
        "batch_multi": 4,
        "resolution": 2048
    }

    # --- LEVEL 6: I_HATE_MY_GPU (Was ist Zeit) ---
    # Der letzte Run. Der EMA zwingt die Engine zur absoluten Perfektion.
    I_HATE_MY_GPU = {
        "patience_factor": 0.75,
        "MAX_BAD_SCORES": 75,          # Hohe Ausdauer für die harten Details
        "ema_positiv_reaction": 0.10,  # Sehr sturer EMA, verlangt konstante Höchstleistung
        "ema_negativ_reaction": 0.005,
        "pinsel_steilheit": 0.80,      # Maximale Freiheit nach oben ("Luft nach oben")
        "n_mutate": 80,
        "top_k": 64,
        "sample_multi": 15,
        "batch_multi": 3,
        "resolution": 4096
    }