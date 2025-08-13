from dataclasses import dataclass, field
from typing import List, Dict, Optional
from peft import TaskType, LoraConfig as PeftLoraConfig

from .lora import LoraConfig
from .training import NerTrainingConfig
from src.type_defs import (
    LabelNamesListType,
    LabelNames,
    NerLabelConfig,
    AggregationStrategyType,
)


@dataclass
class NERConfig:
    model_name: str = "Jean-Baptiste/roberta-large-ner-english"

    categories: List[str] = field(
        default_factory=lambda: [
            "ALL",
            "PER",
            "ORG",
            "MISC",
            "GPE",
            "FAC",
            "LOC",
            "PRODUCT",
            "EVENT",
            "QUANTITY",
            "DATE",
            "MONEY",
            "SHIP",
            "ARMOR",
            "WEAPON",
        ]
    )

    entity_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "MISC": "#c42c2c",
            "PER": "#caaa29",
            "ORG": "#3093a7",
            "GPE": "#af3050",
            "FAC": "#2e77af",
            "LOC": "#2eb488",
            "PRODUCT": "#9744d3",
            "EVENT": "#af9937",
            "QUANTITY": "#2ab68e",
            "DATE": "#d8913b",
            "MONEY": "#5441c2",
            "SHIP": "#a53090",
            "ARMOR": "#44a830",
            "WEAPON": "#9a9e30",
        }
    )

    category_info: Dict[str, str] = field(
        default_factory=lambda: {
            "PER": "Конкретные персонажи (игроки, NPC, известные лица)",
            "ORG": "Официальные организации, корпорации, фракции",
            "GPE": "Планеты, звездные системы с управлением",
            "FAC": "Станции, города, зоны с названием",
            "LOC": "Географические и космические локации без управления",
            "PRODUCT": "Конкретные предметы, техника, гаджеты",
            "SHIP": "Названия кораблей и их модификаций",
            "ARMOR": "Модели брони, костюмы",
            "WEAPON": "Конкретные виды оружия",
            "EVENT": "Названия игровых событий, миссий, кампаний",
            "QUANTITY": "Количественные значения с единицами",
            "DATE": "Годы, даты, игровые циклы",
            "MONEY": "Суммы игровой валюты",
            "MISC": "Прочие категории, не попадающие в другие",
        }
    )

    # Streamlit App Pagination settings
    items_per_page_default: int = 5
    items_per_page_max: int = 20

    # Extraction settings
    threshold_confidence: float = 0.7
    aggregation_strategy: AggregationStrategyType = "average"

    label_names: LabelNamesListType = field(default_factory=lambda: [LabelNames.LABELS])

    lora_config: PeftLoraConfig = field(
        default_factory=lambda: LoraConfig(task_type=TaskType.TOKEN_CLS)
    )
    training_config: Optional[NerTrainingConfig] = None

    def __post_init__(self):
        self.label_list = self._build_label_list()

        self.ner_label_config: NerLabelConfig = {
            "num_labels": len(self.label_list),
            "id2label": {i: label for i, label in enumerate(self.label_list)},
            "label2id": {label: i for i, label in enumerate(self.label_list)},
            "ignore_mismatched_sizes": True,
        }

    def _build_label_list(self) -> List[str]:
        labels = ["O"]

        for cat in self.categories:
            if cat.lower() == "все":
                continue

            labels.append(f"B-{cat}")
            labels.append(f"I-{cat}")

        return labels
