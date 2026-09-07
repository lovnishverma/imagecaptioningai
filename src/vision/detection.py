from typing import Dict, List, Tuple

def format_object_detection(od_data: Dict) -> str:
    """Format object detection results into natural spatial language."""
    if not od_data or not od_data.get("labels"):
        return "No objects detected."

    labels = od_data.get("labels", [])
    bboxes = od_data.get("bboxes", [])

    if not labels:
        return "No objects detected."

    objects: List[Tuple[str, str]] = []
    for label, bbox in zip(labels, bboxes):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Florence uses 0-1000 coordinate space typically, or 0-999
        if cx < 333:
            pos = "on the left"
        elif cx < 666:
            pos = "in the center"
        else:
            pos = "on the right"
            
        objects.append((label.strip(), pos))

    # Deduplicate (keep first occurrence of each label type)
    seen: set = set()
    unique: List[Tuple[str, str]] = []
    for lbl, pos in objects:
        key = lbl.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((lbl, pos))

    if not unique:
        return "No objects detected."

    if len(unique) == 1:
        lbl, pos = unique[0]
        return f"I see a {lbl} {pos}."

    parts = [f"a {lbl} {pos}" for lbl, pos in unique]

    if len(parts) <= 5:
        return "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
    else:
        summary = ", ".join(parts[:5])
        return f"I see {len(unique)} objects including: {summary}, and {len(unique) - 5} more."
