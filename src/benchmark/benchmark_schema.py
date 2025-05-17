from enum import Enum
from datetime import datetime


class BenchmarkInferenceSchema(Enum):
    TIMESTAMP = "timestamp"
    SYMBOL = "symbol"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    MODEL = "model"
    HORIZON = "horizon"

    @classmethod
    def required_fields(cls):
        return [
            cls.TIMESTAMP.value,
            cls.SYMBOL.value,
            cls.OPEN.value,
            cls.HIGH.value,
            cls.LOW.value,
            cls.CLOSE.value,
        ]

    @classmethod
    def optional_fields(cls):
        return [
            cls.MODEL.value,
            cls.HORIZON.value,
        ]

    @classmethod
    def all_fields(cls):
        return [field.value for field in cls]

    @classmethod
    def price_fields(cls):
        return [
            cls.OPEN.value,
            cls.HIGH.value,
            cls.LOW.value,
            cls.CLOSE.value
        ]

    @classmethod
    def field_types(cls) -> dict:
        return {
            cls.TIMESTAMP: datetime,
            cls.SYMBOL: str,
            cls.OPEN: float,
            cls.HIGH: float,
            cls.LOW: float,
            cls.CLOSE: float,
            cls.MODEL: str,
            cls.HORIZON: int,
        }


schema = BenchmarkInferenceSchema.TIMESTAMP
