"""
Read a recorded .pptx deck into plain JSON, so the dashboard can draw the
slides in the browser (no PowerPoint, no conversion tool, no CDN).

Every shape keeps its position and size as a fraction of the slide, so the
page can place it the same way PowerPoint does. Text keeps its font size in
points; the page scales it to the slide's drawn width. Tables keep their
cells and fills, charts their categories and series values. Anything this
reader does not know is left out, never guessed: the .pptx download is the
complete file.
"""
import io

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

TITLE_PREFIX = "Slide title: "  # the Presentation Agent's name for each slide's title box


def _hex(color) -> str | None:
    """The RGB value of a pptx color, or None (theme or unset colors)."""
    try:
        return f"#{color.rgb}" if color is not None and color.type is not None and color.rgb is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _fill(fill) -> str | None:
    try:
        return _hex(fill.fore_color) if fill.type == 1 else None  # MSO_FILL.SOLID
    except (AttributeError, TypeError, ValueError):
        return None


def _paragraphs(frame) -> list[dict]:
    out = []
    for p in frame.paragraphs:
        runs = []
        for r in p.runs:
            if not r.text:
                continue
            font = r.font
            runs.append({"text": r.text, "bold": bool(font.bold), "italic": bool(font.italic),
                         "size": font.size.pt if font.size else None, "color": _hex(font.color)})
        out.append({"runs": runs, "level": p.level,
                    "align": {1: "left", 2: "center", 3: "right"}.get(int(p.alignment or 0), None)})
    return out


def _box(shape, width: int, height: int) -> dict:
    return {"x": (shape.left or 0) / width, "y": (shape.top or 0) / height,
            "w": (shape.width or 0) / width, "h": (shape.height or 0) / height}


def _table(shape) -> dict:
    table = shape.table
    cols = [c.width for c in table.columns]
    total = sum(cols) or 1
    rows = []
    for row in table.rows:
        rows.append([{"paragraphs": _paragraphs(cell.text_frame), "fill": _fill(cell.fill)} for cell in row.cells])
    return {"type": "table", "columns": [w / total for w in cols], "rows": rows}


def _chart(shape) -> dict:
    chart = shape.chart
    plot = chart.plots[0]
    series = []
    for s in plot.series:
        try:
            color = _fill(s.format.fill)
        except (AttributeError, TypeError, ValueError):
            color = None
        points = []
        for i in range(len(list(s.values))):
            try:
                points.append(_fill(s.points[i].format.fill))
            except (AttributeError, TypeError, ValueError, IndexError):
                points.append(None)
        series.append({"name": s.name, "values": list(s.values), "color": color, "point_colors": points})
    title = chart.chart_title.text_frame.text if chart.has_title and chart.chart_title.has_text_frame else ""
    return {"type": "chart", "chart_type": str(chart.chart_type).split(" ")[0].split(".")[-1],
            "title": title, "categories": [str(c) for c in plot.categories], "series": series,
            "horizontal": "BAR" in str(chart.chart_type)}


def _shape(shape, width: int, height: int) -> dict | None:
    item = None
    if getattr(shape, "has_table", False) and shape.has_table:
        item = _table(shape)
    elif getattr(shape, "has_chart", False) and shape.has_chart:
        item = _chart(shape)
    elif shape.shape_type in (MSO_SHAPE_TYPE.AUTO_SHAPE, MSO_SHAPE_TYPE.TEXT_BOX, MSO_SHAPE_TYPE.PLACEHOLDER):
        kind = "text"
        geometry = None
        if shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            kind = "shape"
            try:
                geometry = str(shape.auto_shape_type).split(" ")[0].split(".")[-1].lower()
            except (AttributeError, ValueError, NotImplementedError):
                geometry = "rectangle"
        item = {"type": kind, "geometry": geometry,
                "fill": _fill(shape.fill) if kind == "shape" else None,
                "paragraphs": _paragraphs(shape.text_frame) if shape.has_text_frame else [],
                "anchor": None}
        if shape.has_text_frame:
            try:
                item["anchor"] = {1: "top", 3: "middle", 4: "bottom"}.get(int(shape.text_frame.vertical_anchor or 0))
            except (TypeError, ValueError):
                pass
    if item is None:
        return None  # pictures, groups, media: not drawn here (the .pptx has them)
    item["name"] = shape.name
    item.update(_box(shape, width, height))
    return item


def read_deck(data: bytes) -> dict:
    """The deck as JSON: its aspect ratio and, per slide, its title, shapes
    (in drawing order) and speaker notes."""
    prs = Presentation(io.BytesIO(data))
    width, height = prs.slide_width, prs.slide_height
    slides = []
    for number, slide in enumerate(prs.slides, start=1):
        shapes = [s for s in (_shape(sh, width, height) for sh in slide.shapes) if s]
        title = next((sh.name[len(TITLE_PREFIX):] for sh in slide.shapes if sh.name.startswith(TITLE_PREFIX)), "")
        notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
        slides.append({"number": number, "title": title or f"Slide {number}", "shapes": shapes, "notes": notes})
    return {"width_pt": width / 12700, "height_pt": height / 12700, "slides": slides}
