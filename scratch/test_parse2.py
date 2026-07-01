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
        "assignment":      "assignments",
        "homework":        "homework",
        "reflection":      "reflection",
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

text = """Lesson Plan: ELA 5th Grade
Date: 2026-06-27 Grade: 5 | Subject: ELA | Duration: 45 minutes Standard: ELA.5.R.1.1 — Analyze how setting, events, conflict, and characterization contribute to the plot in a literary text. For informational texts, analyze the purpose and key details the author uses to support claims.

Objective
Analyze how setting, events, conflict, and characterization contribute to the plot in a literary text.

Essential Question
How do the elements of setting, events, conflict, and characterization influence the development of a story's plot?

Instructional Materials
Excerpt from "Charlotte's Web" by E.B. White
Chart paper and markers
Projector and screen
Graphic organizers for plot analysis
Whiteboard and markers
Teaching Strategies
Think-Pair-Share: Students discuss their ideas in pairs before sharing with the larger group.
Graphic Organizers: Students use organizers to visually structure their analysis of the text.
Interactive Read-Aloud: Teacher reads aloud with pauses for student engagement and discussion.
Hook (5 min)
Display an image from "Charlotte's Web" and ask students to predict the setting and main conflict of the story. Encourage them to think about how these elements might affect the plot.

Direct Instruction (10 min)
Use the Interactive Read-Aloud strategy to read a selected excerpt from "Charlotte's Web." Pause to point out examples of setting, events, conflict, and characterization. Model how each element contributes to the plot by filling in a graphic organizer on the whiteboard.

Guided Practice (10 min)
In pairs, students will use the Think-Pair-Share strategy to analyze how the setting, events, conflict, and characterization from the excerpt contribute to the progression of the plot. Each pair will explain their analysis by filling out a section of the graphic organizer and sharing their insights with the class, emphasizing the cause-and-effect relationship between these elements and the plot.

Independent Practice (10 min)
Students will independently analyze a new passage from the same text, focusing on how the setting, events, conflict, and characterization contribute to the plot's development. They will complete their own graphic organizer and write a brief analysis explaining these contributions, articulating the interconnections and effects of each element on the plot.

Assessment (5 min)
How does the setting in "Charlotte's Web" influence the development of the plot? Provide specific examples from the text to support your analysis.

Assignments
Completed graphic organizer analyzing the passage.
Written paragraph explaining the influence of setting and conflict on the plot.
Homework Notes
Reread the selected passage from "Charlotte's Web" at home and share with a family member what you learned about the story's plot elements.
Draw a picture representing your favorite scene and describe which plot elements are present.
Differentiation
For students needing support: Provide a partially filled graphic organizer and sentence starters for the paragraph.
For advanced students: Encourage them to find additional examples from the text that illustrate the plot elements.
Teacher Reflection
Did the students effectively use the graphic organizers to analyze the text?
How well did the Think-Pair-Share discussions promote deeper understanding of the plot elements?
What adjustments could be made to better support students who struggled with the analysis?"""

print(_parse_lesson_sections(text))
