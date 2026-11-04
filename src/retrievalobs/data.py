"""Fictional orbital-station manuals; authored query templates are split before tuning."""

import numpy as np

from .core import Document

SYSTEMS = ["airlock", "coolant", "beacon", "greenhouse", "battery", "antenna"]
TOPICS = {
    "inspection": ("inspection interval", "inspect", "inspecion", "routine examination"),
    "restart": ("restart delay", "restart", "restat", "power cycle"),
    "pressure": ("pressure limit", "pressure", "presure", "maximum compression"),
    "reserve": ("reserve threshold", "reserve", "resrve", "minimum backup"),
}


def corpus():
    rng = np.random.default_rng(177)
    docs = []
    for station in ["Aster", "Boreal", "Cirrus", "Delta"]:
        for system in SYSTEMS:
            for topic, (field, _, _, _) in TOPICS.items():
                for version in [1, 2]:
                    value = int(rng.integers(5, 90))
                    unit = {
                        "inspection": "hours",
                        "restart": "seconds",
                        "pressure": "kPa",
                        "reserve": "percent",
                    }[topic]
                    text = (
                        f"{station} station {system}: the {field} is {value} {unit}. "
                        f"This {topic} procedure applies to the {system} module only."
                    )
                    key = f"{station.lower()}-{system}-{topic}-v{version}"
                    docs.append(
                        Document(
                            key,
                            station,
                            "2025-01-01" if version == 1 else "2026-01-01",
                            "2026-01-01" if version == 1 else None,
                            text,
                        )
                    )
    return docs


def queries(split):
    if split not in {"development", "test"}:
        raise ValueError("unknown split")
    items = []
    for station in ["Aster", "Boreal", "Cirrus", "Delta"]:
        for system in SYSTEMS:
            for topic, (field, literal, typo, paraphrase) in TOPICS.items():
                forms = (
                    {
                        "literal": f"{system} {field}",
                        "typo": f"{system} {typo}",
                        "paraphrase": f"{system} {paraphrase}",
                    }
                    if split == "test"
                    else {"development": f"What is the {literal} rule for {system}?"}
                )
                for kind, text in forms.items():
                    items.append(
                        {
                            "station": station,
                            "query": text,
                            "kind": kind,
                            "relevant": [f"{station.lower()}-{system}-{topic}-v2"],
                        }
                    )
            unsupported = (
                [
                    "How many crew members use this station?",
                    f"Who manufactured the {system} module?",
                ]
                if split == "test"
                else [f"What is the replacement price of the {system}?"]
            )
            for text in unsupported:
                items.append(
                    {"station": station, "query": text, "kind": "unsupported", "relevant": []}
                )
    return items
