# Agent Workflow Design

## State schema

The LangGraph workflow uses a shared typed state in `src/agent/state.py`.

Core fields:

- `patient_input`: raw payload from the Streamlit form.
- `missing_fields`: required features absent from the payload.
- `normalized_patient_data`: validated subset of patient fields safe to use downstream.
- `risk_prediction`: model risk tier (`Low`, `Moderate`, `High`) or `Unavailable`.
- `risk_probability`: ML probability when a full score is available.
- `risk_factors`: top ML contributors, or rule-based fallback factors.
- `retrieved_chunks`: semantic-retrieval results from Chroma with chunk metadata.
- `retrieved_sources`: cleaned source labels shown in the UI and final report.
- `draft_summary`: deterministic risk summary used before final rendering.
- `recommendations`: deterministic, grounded action items.
- `final_report`: structured report object with markdown rendering.
- `disclaimer`: visible medical caution statement.
- `errors`: accumulated tool/model failures.
- `workflow_trace`: node-level execution log for demo/debug visibility.

Reducer-backed fields:

- `errors`
- `fallback_status`
- `workflow_trace`

These append status entries across nodes without exposing hidden chain-of-thought.

## Node responsibilities

`validate_input`

- Inspects the payload without raising.
- Records missing/invalid fields.
- Flags partial-output mode early when needed.

`normalize_input`

- Keeps only validated fields.
- Derives rule-based clinical signals from available data.

`score_risk`

- Calls the existing logistic-regression inference pipeline in `src/inference.py`.
- Preserves the original ML model and artifact-loading path.

`extract_risk_factors`

- Narrows the explainability payload to the highest-signal factors.

`retrieve_guidelines`

- Builds a retrieval query from risk tier, top factors, patient context, and optional user focus.
- Runs semantic similarity search over the persisted Chroma vector DB.

`generate_summary`

- Produces a deterministic summary grounded in visible workflow state.

`generate_recommendations`

- Builds deterministic recommendations first.
- Optionally asks the LLM to render the final structured report.
- Falls back cleanly when the LLM is unavailable or fails.

`validate_output`

- Enforces the output contract.
- Ensures the final report always has:
  - `Risk Summary`
  - `Key Factors`
  - `Recommendations`
  - `Follow-up Suggestions`
  - `Sources`
  - `Disclaimer`

`handle_fallback`

- Gracefully prepares partial state when scoring is impossible or a node fails upstream.

## Branching logic

Normal path:

`validate_input -> normalize_input -> score_risk -> extract_risk_factors -> retrieve_guidelines -> generate_summary -> generate_recommendations -> validate_output`

Partial-data path:

`validate_input -> normalize_input -> handle_fallback -> retrieve_guidelines -> generate_summary -> generate_recommendations -> validate_output`

Scoring-failure path:

`validate_input -> normalize_input -> score_risk -> handle_fallback -> retrieve_guidelines -> generate_summary -> generate_recommendations -> validate_output`

## Failure handling

- Missing required patient fields do not crash the app.
- Invalid values do not crash the app.
- ML scoring failure downgrades to a deterministic partial report.
- Empty retrieval sets a visible no-source message and continues safely.
- LLM failure falls back to deterministic report rendering.
- Unexpected graph/runtime failure returns a minimal safe report instead of an exception to the UI.

## Hallucination-reduction strategy

- The prompt explicitly forbids invented diagnoses and fabricated guideline claims.
- The LLM is instructed to ground all recommendations in:
  - patient input
  - model output
  - retrieved chunks
- Deterministic recommendations are created before optional LLM rendering.
- The final renderer checks for required markdown sections; malformed LLM output is discarded.
- Source attribution is always shown.
- The disclaimer is always visible.
- Uncertainty is stated explicitly when data is incomplete or retrieval returns nothing.
