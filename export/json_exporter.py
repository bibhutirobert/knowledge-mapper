"""JSON export: serialise the full compression result."""

import json
from modules.compression_engine import CompressionResult
from modules.mindmap_generator import generate_json


def export_to_json(result: CompressionResult, indent: int = 2) -> str:
    data = generate_json(result)
    return json.dumps(data, indent=indent, ensure_ascii=False)
