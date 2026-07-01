# Lesson Plan Helper

**Track:** Agents for Good

Lesson Plan Helper is a standards-aligned, classroom-ready lesson planner designed for K-12 educators. It automates the search for state educational standards and builds a complete, aligned lesson plan from a teacher's brief topic description. 

To ensure strict alignment, the system uses a **multi-agent LangGraph workflow** that retrieves state-specific standards, supports a human-in-the-loop approval step, evaluates the generated lesson against a rigorous alignment rubric, and surgically rewrites any flagged sections in an autonomous revision loop.

---

## 1. System Architecture

The application is powered by a state-aware orchestration graph built with **LangGraph**. It consists of six processing nodes, three conditional routers, and a memory checkpointer that supports interrupts and resumes.

### Workflow Topology

```text
               START
                 │
                 ▼
          ┌─────────────┐
          │ intake_node │
          └──────┬──────┘
                 │
                 ├─[invalid inputs]──► END (workflow aborted)
                 │
                 ▼ [valid inputs]
        ┌────────────────┐
        │ discovery_node │
        └────────┬───────┘
                 │
                 ▼
      ┌────────────────────┐
      │ confirmation_node  │ ◄── [HUMAN-IN-THE-LOOP INTERRUPT]
      └──────────┬─────────┘     (Teacher reviews standard & approves/overrides)
                 │
                 ├─[rejected]────────► END (workflow stopped)
                 │
                 ▼ [approved]
        ┌───────────────┐
        │ planning_node │
        └────────┬──────┘
                 │
                 ▼
        ┌───────────────┐     ◄─────────────────────────┐
        │  review_node  │                               │
        └────────┬──────┘                               │
                 │                                      │ (Revision Loop)
                 ├─[failed checks & retry allowed]──────┼─► rewrite_node
                 │                                      │   (Surgical fix)
                 ├─[max retries exceeded]──► END        │
                 │                                      │
                 ▼ [all checks passed]                  │
                END (Approved plan delivered) ──────────┘
```

### Graph Nodes

1. **`intake_node`**: Validates teacher-provided inputs (subject, grade, topic, state, duration) to ensure the query has sufficient context.
2. **`discovery_node`**: Leverages the standards database to find the best-matching standards for the target grade, subject, and state based on keyword-overlap scoring.
3. **`confirmation_node`**: Triggers a LangGraph `interrupt` to pause execution, allowing the teacher to review the retrieved standard and either approve it or select an override.
4. **`planning_node`**: Invokes the **Planning Agent** to write a structured lesson plan in Markdown based on the approved standard.
5. **`review_node`**: Invokes the **Review Agent** to check the draft lesson against a strict 4-point rubric.
6. **`rewrite_node`**: Invokes the **Rewrite Agent** to address specific alignment failures flagged by the reviewer.

---

## 2. Multi-Agent Collaboration & Revision Loop

The system utilizes three specialized agents communicating through a shared **Workflow State**:

```
      ┌─────────────────────────────────────────────────────────────┐
      │                      LessonPlanState                        │
      │  Inputs | retrieved_standards | selected_standard | draft   │
      │  review_feedback | revision_count | final_lesson_plan       │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
 ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
 │Planning Agent │           │ Review Agent  │           │ Rewrite Agent │
 │ (Generates)   │           │ (Evaluates)   │           │   (Revises)   │
 └───────────────┘           └───────────────┘           └───────────────┘
```

### The Rubric Check

The **Review Agent** evaluates the draft lesson plan against four criteria:

*   **Standard Match**: Does the lesson objective use the exact cognitive verb and skill described in the standard?
*   **Objective-to-Assessment Match**: Does the assessment directly measure the target skill in the objective?
*   **Activity-to-Objective Match**: Do the practice activities directly build the skill outlined in the objective?
*   **Pacing Realism**: Do activity time estimates sum to the stated lesson duration?

### Targeted Revision

If any check fails, the Review Agent outputs a structured JSON report specifying the failed criteria and actionable feedback. The **Rewrite Agent** takes this report and the current draft, and rewrites **only** the sections that caused the failure (e.g., shrinking activity times to resolve pacing errors, or rewriting the assessment to match the objective), leaving the rest of the plan unchanged.

The revised plan is routed back to the Review Agent. This loop continues until all checks pass, or the maximum revision limit (default: 3) is reached.

---

## 3. Supported Educational Standards

The application includes an in-memory document store pre-populated with standards:

| Framework | State Key | Subject | Grade | Target Skills Covered |
|---|---|---|---|---|
| **Florida B.E.S.T.** | `florida` | ELA | Grade 5 | Central idea, theme, plot elements, figurative language |
| **Texas TEKS** | `texas` | ELA | Grade 5 | Inferences, central idea, text evidence, synthesizing texts |
| **Virginia SOL** | `virginia` | ELA | Grade 5 | Drawing conclusions, main idea, summarizing, organizational patterns |
| **Common Core ELA** | `common_core` | ELA | Grade 5 | Text evidence, main idea, summarizing, academic vocabulary |

*Fallback Routing*: If a teacher inputs a state not present in the database (e.g., "Alaska"), the system automatically falls back to the **Common Core** corpus for that grade and subject, and flags the standard as a fallback so the frontend can warn the teacher.

---

## 4. Setup & Running the Application

### Prerequisites

*   Python 3.10 or higher
*   An OpenAI API key (optional for simulation run, required for the Streamlit UI)

### Installation

1.  Clone this repository and navigate to the project directory.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  (Optional) Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your-actual-api-key-here
    ```

---

## 5. Running the Simulation Script

To verify the system's routing, confirmation interrupts, and multi-agent revision loops without using the UI, run the holistic integration script:

```bash
python simulate_run.py
```

### What the Simulation Does:
1.  **Mock Query**: Submits a Grade 5 Reading topic ("Explain central idea and details") for **Florida**.
2.  **Standards Discovery**: Successfully routes to the Florida BEST standards database, retrieving `ELA.5.R.1.2`.
3.  **Human Interruption**: Pauses at `confirmation_node` to simulate standard confirmation.
4.  **Forced Pacing Failure**: Mock-injects a draft lesson plan that schedules **125 minutes of activities** for a **45-minute lesson**.
5.  **Alignment Failure**: The **Review Agent** catches the pacing failure and routes the state to the **Rewrite Agent**.
6.  **Surgical Rewrite**: The Rewrite Agent shrinks activity durations (Hook 5m, Direct Instruction 10m, Guided Practice 15m, Independent Practice 10m, Assessment 5m) to sum to exactly 45 minutes.
7.  **Successful Loopback**: The Review Agent checks the revised plan, passes the pacing check, and completes the workflow.
8.  **Trace Logs**: Outputs the complete node-by-node update history and intermediate states to `evaluation_logs.json`.

---

## 6. Running the Streamlit App

To run the interactive teacher-facing web application:

```bash
streamlit run ui/app.py
```

*Note: The Streamlit app requires an active `OPENAI_API_KEY` configured in your environment or `.env` file to invoke the OpenAI agents.*
