"""Everything topic-specific lives here: record fields, prompts, and the example questions
shown in the GUI. Edit this file if your papers are not about membrane filtration."""
from __future__ import annotations

TOPIC = "membrane filtration"

# ---- Suggestions shown in the Streamlit GUI -------------------------------------------
LIBRARY_HINT = (
    "**What to enter:** ask a specific question about the papers in your library. "
    "Name the membrane or material, the substance being removed, and any limit **with units** "
    "(for example *below 3 bar*). Questions with a number or a condition get the best answers."
)
GLOBAL_HINT = (
    "**What to enter:** describe a topic like a search query - material + target substance + what you want "
    "(e.g. a review, recent results, typical values). Answers come from OpenAlex abstracts and the web, "
    "**not** from your own library."
)
LIBRARY_EXAMPLES = [
    "Which membrane has the highest rejection while operating below 3 bar?",
    "Compare the flux of the membranes tested for the same solute",
    "What feed concentration and temperature were used in the experiments?",
    "Summarize the main conclusions about membrane fouling",
]
GLOBAL_EXAMPLES = [
    "Recent advances in nanofiltration membranes for heavy metal removal",
    "Typical salt rejection of polyamide thin-film composite membranes",
    "Fouling mitigation strategies for ultrafiltration membranes",
    "Low-pressure membranes for dye removal from textile wastewater",
]
LIBRARY_PLACEHOLDER = "e.g. Which membrane gives the highest rejection below 3 bar?"
GLOBAL_PLACEHOLDER = "e.g. Recent advances in nanofiltration for heavy metal removal"

# ---- Extraction ------------------------------------------------------------------------
SECTION_SKIP = ("reference", "bibliograph", "acknowledg", "introduction", "author contribution",
                "conflict of interest", "funding")
SECTION_GOOD = ("result", "discussion", "experiment", "method", "material", "table", "performance")

EXTRACTION_SYSTEM = (
    "You extract experimental data records from scientific text. "
    "Output ONLY valid JSON. Never guess, never calculate, never convert units. "
    "Copy numbers exactly as written in the text."
)

EXTRACTION_INSTRUCTIONS = """Extract experimental result records from the excerpts below.

Return JSON of the form {"records": [ ... ]}. One record per distinct measured result
(one membrane + one solute + one set of operating conditions). Fields (use null when not stated):

- chunk_id: the id from the excerpt header where the value appears (copy exactly)
- membrane: membrane name / material / type as written
- solute: the substance being rejected or removed
- rejection_value: number as written; rejection_unit: "%" or "fraction"
- pressure_value: number as written; pressure_unit: as written (bar, kPa, MPa, psi...)
- flux_value: number as written; flux_unit: as written (e.g. "L/m2/h", "LMH")
- feed_conc_value: number as written; feed_conc_unit: as written (mg/L, g/L, ppm, mM...)
- temperature_value: number as written; temperature_unit: as written
- ph: number or null
- operation_mode: "cross-flow", "dead-end" or null
- evidence: a SHORT verbatim snippet (max 25 words) containing the main number

Rules:
- Only results measured in THIS paper. Ignore values quoted from other studies or literature comparison tables.
- Skip anything without at least a rejection value or a flux value.
- If nothing qualifies, return {"records": []}.
"""

# ---- Normalised record columns (SQLite) -------------------------------------------------
NUMERIC_COLUMNS = ["rejection_pct", "pressure_bar", "flux_lmh", "feed_conc_mg_l", "temperature_c", "ph"]
DISPLAY_COLUMNS = ["source", "page", "membrane", "solute", "rejection_pct", "pressure_bar", "flux_lmh",
                   "feed_conc_mg_l", "feed_conc_raw", "temperature_c", "ph", "operation_mode", "flags"]
