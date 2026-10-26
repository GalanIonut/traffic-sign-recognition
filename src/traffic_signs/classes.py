"""Numele claselor GTSRB, în limba română."""
from traffic_signs.config import NUM_CLASSES

CLASS_NAMES: dict[int, str] = {
    0: "Limită de viteză 20 km/h",
    1: "Limită de viteză 30 km/h",
    2: "Limită de viteză 50 km/h",
    3: "Limită de viteză 60 km/h",
    4: "Limită de viteză 70 km/h",
    5: "Limită de viteză 80 km/h",
    6: "Sfârșitul limitei de viteză 80 km/h",
    7: "Limită de viteză 100 km/h",
    8: "Limită de viteză 120 km/h",
    9: "Depășirea interzisă",
    10: "Depășirea interzisă pentru vehicule peste 3,5 t",
    11: "Prioritate la următoarea intersecție",
    12: "Drum cu prioritate",
    13: "Cedează trecerea",
    14: "Stop",
    15: "Circulația interzisă în ambele sensuri",
    16: "Accesul interzis vehiculelor peste 3,5 t",
    17: "Accesul interzis",
    18: "Atenție, alte pericole",
    19: "Curbă la stânga",
    20: "Curbă la dreapta",
    21: "Curbă dublă",
    22: "Drum denivelat",
    23: "Drum alunecos",
    24: "Drum îngustat pe partea dreaptă",
    25: "Lucrări",
    26: "Semafor",
    27: "Trecere pentru pietoni",
    28: "Copii",
    29: "Bicicliști",
    30: "Polei sau zăpadă",
    31: "Animale sălbatice",
    32: "Sfârșitul tuturor restricțiilor",
    33: "Obligatoriu la dreapta",
    34: "Obligatoriu la stânga",
    35: "Obligatoriu înainte",
    36: "Înainte sau la dreapta",
    37: "Înainte sau la stânga",
    38: "Ocolire prin dreapta",
    39: "Ocolire prin stânga",
    40: "Sens giratoriu",
    41: "Sfârșitul interzicerii depășirii",
    42: "Sfârșitul interzicerii depășirii pentru vehicule peste 3,5 t",
}


def class_name(class_id: int) -> str:
    """Întoarce numele românesc al clasei; aruncă ValueError pentru ID invalid."""
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"ID de clasă invalid: {class_id} (valid: 0…{NUM_CLASSES - 1})")
    return CLASS_NAMES[class_id]
