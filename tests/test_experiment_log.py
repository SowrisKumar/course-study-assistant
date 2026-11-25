"""The experiment log records one row per (build, configuration) and never duplicates them."""

import pandas as pd

from study_assistant.config import PROJECT_ROOT
from study_assistant.experiment_log import COLUMNS, record

ARCHIVE = PROJECT_ROOT / "eval" / "archive" / "results_before_audit"
OLD_QUESTIONS = PROJECT_ROOT / "eval" / "archive" / "questions_before_audit.jsonl"


def test_record_logs_each_configuration_once(tmp_path):
    log = tmp_path / "log.csv"
    kw = dict(commit="abc1234", kb_documents=507, questions_path=OLD_QUESTIONS, log_path=log)
    first = record("build A", ARCHIVE, **kw)
    assert len(first) == 5 and list(pd.read_csv(log).columns) == COLUMNS
    record("build A", ARCHIVE, **kw)  # re-logging replaces, not appends
    record("build B", ARCHIVE, runs=["rag__ollama__llama3.1-8b__hybrid"], **kw)
    saved = pd.read_csv(log)
    assert len(saved) == 6 and saved.groupby(["build", "run"]).size().max() == 1


def test_logged_scores_match_the_step5_report(tmp_path):
    rows = record("old", ARCHIVE, commit="f1adc5a", kb_documents=507, questions_path=OLD_QUESTIONS,
                  log_path=tmp_path / "log.csv").set_index("run")
    hybrid = rows.loc["rag__ollama__llama3.1-8b__hybrid"]
    assert round(hybrid.correct_main, 2) == 0.97 and round(hybrid.not_covered_declined, 2) == 0.92
    assert round(rows.loc["no_rag__ollama__llama3.1-8b"].correct_main, 2) == 0.55
