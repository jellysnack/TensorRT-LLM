import dataclasses

from typing import Dict, Optional, Callable

@dataclasses.dataclass
class CalibrationConfig:
    dataset: str
    batch_size: int
    max_samples: int
    max_seq_length: int
    formatting_func: Optional[Callable] = None
    truncation: bool = False

    @staticmethod
    def from_dict(d: Dict) -> 'CalibrationConfig':
        return CalibrationConfig(
            dataset=d['dataset'],
            batch_size=d['batch_size'],
            max_samples=d['max_samples'],
            max_seq_length=d['max_seq_length'],
            formatting_func=d['formatting_func'],
            truncation=d['truncation']
        )
    
    def to_dict(self):
        result = dataclasses.asdict(self)
        if result['formatting_func'] is not None:
            result['formatting_func'] = '<cannot be printed>'
        return result
