"""Ensure portfolio exports represent the correct run and remain portable."""

import io
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from streamlit.testing.v1 import AppTest

from backend.schemas import ComparisonRequest, GenerationRequest, TrainingRequest
from backend.services.model_comparison import compare_models
from backend.services.modeling import train_linear_regression
from backend.services.synthetic_data import generate_dataset
from frontend.exports import build_report


def test_calibrated_export_round_trip_and_no_run_tokens():
    dataset = generate_dataset(GenerationRequest(n_samples=100)).model_dump()
    comparison = compare_models(ComparisonRequest(rows=dataset['rows'], models=['Linear Regression'],
                                                  include_diagnostics=True), persist=True).model_dump()
    evaluation = comparison['evaluation']
    explanation = dict(run_id=evaluation['run_id'], source='template', model=None,
                       text='Measured result summary.', notice='Local explanation.')
    archive_bytes, summary = build_report(dataset, evaluation, comparison, explanation)
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert len(archive.namelist()) == 7
        report_text = archive.read('results.json').decode('utf-8')
        assert evaluation['run_id'] not in report_text
        report = json.loads(report_text)
        assert report['evaluation']['test_metrics'] == evaluation['test_metrics']
        assert report['explanation']['source'] == 'template'
        exported_data = pd.read_csv(archive.open('dataset.csv'))
        pd.testing.assert_frame_equal(exported_data, pd.DataFrame(dataset['rows']))
        predictions = pd.read_csv(archive.open('test_predictions.csv'))
        assert len(predictions) == evaluation['test_count']
        assert predictions['row_index'].tolist() == [p['row_index'] for p in evaluation['predictions']]
        assert predictions['covered'].mean() == comparison['diagnostics']['empirical_coverage']
    assert 'Synthetic teaching data' in summary and 'Local explanation.' in summary


def test_uncalibrated_export_omits_stale_explanation():
    dataset = generate_dataset(GenerationRequest(n_samples=100)).model_dump()
    evaluation = train_linear_regression(TrainingRequest(rows=dataset['rows']), persist=True).model_dump()
    stale = dict(run_id='stale-run', source='openrouter', model='free', text='OLD EXPLANATION')
    blob, summary = build_report(dataset, evaluation, explanation=stale)
    with ZipFile(io.BytesIO(blob)) as archive:
        assert set(archive.namelist()) == {'README.md', 'results.json', 'dataset.csv', 'test_predictions.csv'}
        assert 'explanation' not in json.loads(archive.read('results.json'))
    assert 'OLD EXPLANATION' not in summary


def test_report_ui_makes_no_network_request():
    dataset = generate_dataset(GenerationRequest(n_samples=100)).model_dump()
    evaluation = train_linear_regression(TrainingRequest(rows=dataset['rows'])).model_dump()
    with patch('frontend.api_client.requests.post') as post:
        page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend' / 'app.py'), default_timeout=30).run()
        page.session_state['dataset'] = dataset
        page.session_state['training_result'] = evaluation
        page.radio[0].set_value('Linear Regression walkthrough').run()
        next(b for b in page.button if b.label == 'Prepare evaluation report').click().run()
        assert not page.exception
        assert len(page.get('download_button')) == 3  # dataset + bundle + summary
        post.assert_not_called()
