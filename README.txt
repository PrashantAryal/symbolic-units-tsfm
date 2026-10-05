symtsfm: symbolic words as pattern-level evidence for a frozen UniTS time-series foundation model

Layout
  src/symtsfm/
    data/          dataset loaders (UEA/UCR, ETT/Weather/Exchange/Electricity, anomaly series), routing in datasets.py
    symbolic/      SAX and SFA words, FastShapelets / BOSS-ST word selection, word features, rare-pattern score,
                   evidence read-outs; the other modules are variants that were tried and are not used in the paper
    models/        UniTS wrapper with the gated residual adapter
    fusion/        the gate, and word attributions for the evidence records
    evaluation/    metrics, anomaly protocol (Best-F1, VUS-PR), result reports
    experiments/   the training and evaluation runner
  scripts/         run_suite.py (runner with the data and metric patches used for the reported runs), run_symtsfm.py
                   (anomaly and evidence runs), aggregate_seeds.py, summarize_symtsfm.py, render_paper_figures.py
  configs/         run configurations (units_*: classification, forecasting, anomaly; evidence_heads/: anomaly and evidence runs)
  tests/           unit and smoke tests

Setup
  Python 3.12. See requirements.txt for the install commands.
  The UniTS code and its pretrained checkpoint are not included: put the UniTS repository at third_party/UniTS and
  give the checkpoint path in the configuration or with --units-checkpoint. Datasets are downloaded by the loaders
  into the folder given by --data-dir.

Run (from the repository root)
  python scripts/run_suite.py --help
  python scripts/run_symtsfm.py --help

Tests
  python -m pytest
  The model and pipeline tests need the UniTS repository at third_party/UniTS and are skipped without it.
  The anomaly tests need the optional vus package.
