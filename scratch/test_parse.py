import re
def _parse_lesson_sections(markdown_text: str) -> dict[str, str]:
    heading_map = {
        "standard":        "standards",
        "objective":       "objective",
        "essential":       "essential_q",
        "material":        "materials",
        "strateg":         "strategies",
        "outline":         "outline",
        "hook":            "outline",
        "direct instruct": "outline",
        "guided":          "guided",
        "independent":     "independent",
        "assessment":      "assessment",
        "exit ticket":     "assessment",
        "differentiat":    "differentiation",
        "accommodat":      "differentiation",
        "pacing":          "outline",
    }
    sections = {}
    current_key = None
    current_lines = []
    for line in markdown_text.split("\n"):
        clean_line = re.sub(r"^[\s#\*]+", "", line).strip()
        clean_line_no_colon = re.sub(r":$", "", clean_line).lower()
        heading_match_text = None
        if re.match(r"^\s*#{1,4}\s+(.+)", line):
            heading_match_text = clean_line_no_colon
        elif re.match(r"^\s*\*\*(.+?)\*\*:?\s*$", line):
            heading_match_text = clean_line_no_colon
        elif len(clean_line_no_colon) < 40 and not clean_line_no_colon.endswith("."):
            if not re.match(r"^\s*[-*+]\s", line) and not re.match(r"^\s*\d+\.\s", line):
                for keyword in heading_map.keys():
                    if keyword in clean_line_no_colon:
                        words = clean_line.split()
                        if len(words) <= 5: 
                            heading_match_text = clean_line_no_colon
                            break
        if heading_match_text:
            if current_key and current_lines:
                body = "\n".join(current_lines).strip()
                if current_key not in sections:
                    sections[current_key] = body
                else:
                    sections[current_key] += "\n\n" + body
            current_key = None
            for keyword, card_key in heading_map.items():
                if keyword in heading_match_text:
                    current_key = card_key
                    break
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    if current_key and current_lines:
        body = "\n".join(current_lines).strip()
        if current_key not in sections:
            sections[current_key] = body
        else:
            sections[current_key] += "\n\n" + body
    return sections

text = """Objective
Analyze how setting, events, conflict, and characterization contribute to the plot in a literary text.

Essential Question
How do the elements of setting, events, conflict, and characterization influence the development of a story's plot?

Instructional Materials
Excerpt from "Charlotte's Web" by E.B. White
Chart paper and markers

Teaching Strategies
Think-Pair-Share: Students discuss their ideas
"""

print(_parse_lesson_sections(text))
