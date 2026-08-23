### 01_Code
- volunteer_belonging_llm_model.py: exact simulation source archived with the
  reported DeepSeek/Ollama results.
- generate_volunteer_results_figures.py: script used to produce the principal
  descriptive/statistical outputs and figures.
- volunteer_belonging_llm_model_sensitivity.py: OAT sensitivity-analysis
  version using the same random seeds across values of each parameter.

### 02_Data
Exact outputs from the reported five-condition experiment:
- volunteer_agent_log.csv
- volunteer_condition_summary.csv
- volunteer_run_summary.csv
- volunteer_weekly_metrics.csv

The experiment used 30 agents, 50 activity cycles, and 30 runs per condition.

### 03_Main_Analysis
Contains the main statistical-result tables and manuscript figures derived from
the reported experiment.

### 04_Sensitivity_Analysis
Contains run-level, temporal, and summary results from the one-at-a-time
sensitivity analysis, together with 300-dpi figures.

### 05_LLM
Contains the exact prompt templates and the available LLM configuration details.
The formal model selects all actions; the LLM only generates language and
extracts social signals.

### 06_Configuration
Contains baseline model parameters, condition definitions, the sensitivity grid,
and Python package requirements.

## Main LLM configuration
- Serving platform: Ollama
- Model tag: deepseek-r1:8b
- Message-generation temperature: 0.2
- Action/signal-extraction temperature: 0.0
- Signal extraction requests JSON output

The archived experiment metadata does not contain the exact Ollama software
version or model digest/hash. If recoverable, these should be added before final
public archival.
