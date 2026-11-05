"""FOD-A v2.1 sınıf -> Türkçe isim, risk seviyesi ve pist bölgesi eşlemesi.

YÜKSEK: Uçak motoru emilimi (ingestion) veya lastik/iniş takımı hasarı riski yüksek, sert/keskin/metal cisimler
ORTA  : Hasar potansiyeli orta düzeyde olan parçalar
DÜŞÜK : Yumuşak veya hafif cisimler
"""

RISK_YUKSEK = "Yüksek"
RISK_ORTA = "Orta"
RISK_DUSUK = "Düşük"

CLASS_NAMES = [
    "AdjustableClamp",
    "AdjustableWrench",
    "Battery",
    "Bolt",
    "BoltNutSet",
    "BoltWasher",
    "ClampPart",
    "Cutter",
    "FuelCap",
    "Hammer",
    "Hose",
    "Label",
    "LuggagePart",
    "LuggageTag",
    "MetalPart",
    "MetalSheet",
    "Nail",
    "Nut",
    "PaintChip",
    "Pen",
    "PlasticPart",
    "Pliers",
    "Rock",
    "Screw",
    "Screwdriver",
    "SodaCan",
    "Tape",
    "Washer",
    "Wire",
    "Wood",
    "Wrench",
]

TURKISH_LABELS = {
    "AdjustableClamp": "Ayarlanabilir Kelepçe",
    "AdjustableWrench": "Kurbağacık Anahtar",
    "Battery": "Batarya / Pil",
    "Bolt": "Cıvata",
    "BoltNutSet": "Cıvata-Somun Seti",
    "BoltWasher": "Cıvata Pulu",
    "ClampPart": "Kelepçe Parçası",
    "Cutter": "Maket Bıçağı / Kesici",
    "FuelCap": "Yakıt Kapağı",
    "Hammer": "Çekiç",
    "Hose": "Hortum Parçası",
    "Label": "Etiket",
    "LuggagePart": "Bagaj Parçası",
    "LuggageTag": "Bagaj Etiketi",
    "MetalPart": "Metal Parça",
    "MetalSheet": "Metal Levha",
    "Nail": "Çivi",
    "Nut": "Somun",
    "PaintChip": "Boya Kalıntısı",
    "Pen": "Kalem",
    "PlasticPart": "Plastik Parça",
    "Pliers": "Pense",
    "Rock": "Taş / Kaya",
    "Screw": "Vida",
    "Screwdriver": "Tornavida",
    "SodaCan": "Teneke Kutu",
    "Tape": "Bant Parçası",
    "Washer": "Pul / Rondela",
    "Wire": "Tel / Kablo",
    "Wood": "Ahşap Parçası",
    "Wrench": "İngiliz Anahtarı",
}

RISK_MAP = {
    "AdjustableClamp": RISK_YUKSEK,
    "AdjustableWrench": RISK_YUKSEK,
    "Battery": RISK_ORTA,
    "Bolt": RISK_YUKSEK,
    "BoltNutSet": RISK_YUKSEK,
    "BoltWasher": RISK_YUKSEK,
    "ClampPart": RISK_YUKSEK,
    "Cutter": RISK_YUKSEK,
    "FuelCap": RISK_ORTA,
    "Hammer": RISK_YUKSEK,
    "Hose": RISK_ORTA,
    "Label": RISK_DUSUK,
    "LuggagePart": RISK_ORTA,
    "LuggageTag": RISK_DUSUK,
    "MetalPart": RISK_YUKSEK,
    "MetalSheet": RISK_YUKSEK,
    "Nail": RISK_YUKSEK,
    "Nut": RISK_YUKSEK,
    "PaintChip": RISK_DUSUK,
    "Pen": RISK_DUSUK,
    "PlasticPart": RISK_ORTA,
    "Pliers": RISK_YUKSEK,
    "Rock": RISK_YUKSEK,
    "Screw": RISK_YUKSEK,
    "Screwdriver": RISK_YUKSEK,
    "SodaCan": RISK_ORTA,
    "Tape": RISK_DUSUK,
    "Washer": RISK_YUKSEK,
    "Wire": RISK_YUKSEK,
    "Wood": RISK_ORTA,
    "Wrench": RISK_YUKSEK,
}

RISK_COLOR_BGR = {
    RISK_YUKSEK: (48, 59, 255),   # Kırmızı (Yüksek Risk)
    RISK_ORTA: (0, 165, 255),     # Turuncu/Amber (Orta Risk)
    RISK_DUSUK: (210, 112, 29),   # Mavi (Düşük Risk)
}


def risk_of(class_name: str) -> str:
    """Sınıf adına göre risk seviyesini döndürür."""
    return RISK_MAP.get(class_name, RISK_ORTA)


def tr_label_of(class_name: str) -> str:
    """Sınıf adına göre Türkçe karşılığını döndürür."""
    return TURKISH_LABELS.get(class_name, class_name)


def runway_sector_of(norm_x: float) -> str:
    """Kare üzerindeki yatay konuma göre pist sektör bilgisini döndürür."""
    return "Pist 12L" if norm_x < 0.5 else "Pist 12R"

