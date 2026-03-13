"""HTML export: wraps the pyvis HTML in a standalone page."""

from modules.compression_engine import CompressionResult
from modules.mindmap_generator import generate_pyvis_html


def export_to_html(result: CompressionResult, title: str = "Knowledge Map") -> str:
    inner = generate_pyvis_html(result)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #fafaf9; }}
    h1   {{ font-size: 1.4rem; font-weight: 500; padding: 1rem 1.5rem 0.25rem;
             color: #2C2C2A; margin: 0; }}
    .sub {{ font-size: 0.85rem; color: #888780; padding: 0 1.5rem 0.75rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="sub">Interactive knowledge map — drag nodes to explore</div>
  {inner}
</body>
</html>"""
