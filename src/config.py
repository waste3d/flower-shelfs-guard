from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class ZoneConfig:
    name: str
    box: Tuple[int, int, int, int] # (x1, y1, x2, y2)

@dataclass
class Settings:
    bot_token: str

IMAGE_ZONES: Dict[str, List[ZoneConfig]] = {
    'shelf_full_1.jpg': [
        ZoneConfig(name="full_zone", box=(0, 0, 183, 275)),
    ],
    'shelf_full_2.jpg': [
        ZoneConfig(name="full_zone", box=(0, 0, 200, 253)),
    ],
    'shelf_empty_1.jpg': [
        ZoneConfig(name="empty_zone", box=(0, 0, 275, 183)),
    ],
    'shelf_empty_2.jpg': [
        ZoneConfig(name="empty_zone", box=(0, 0, 740, 493)),
    ],
}

PAVILIONS = {
    1: "Карусель",
    2: "Пятёрочка",
    3: "Таллин",
    4: "Якорь",
}
