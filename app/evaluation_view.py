"""Read measured benchmark results and observed tool outcomes; never trigger paid runs."""
import json
from pathlib import Path
import streamlit as st
from src.config import PROJECT_ROOT


def render_model_evaluation(user_type, execution, report_path=None):
    if user_type not in ('doctor','attendant'):
        return
    with st.expander('Model evaluation and tool outcomes'):
        path = Path(report_path or PROJECT_ROOT/'reports/evaluation/latest.json')
        if not path.exists():
            st.info('No evaluation report yet. Run python -m scripts.evaluate_models.')
        else:
            try:
                report = json.loads(path.read_text())
                summary = report['summary']
                st.caption(f"Run: {report['run_at']} · Dataset: {report['dataset_version']}")
                st.write(report['scope'])
                st.write(report['grading'])
                st.json(report['models'])
                from app.performance_view import render_quality_charts
                render_quality_charts(report)
                st.write(f"Evaluated {summary['evaluated']}/{summary['total']}; failed: {summary['failed']}")
                st.dataframe([{'task':task,**metrics} for task,metrics in report['by_task'].items()],hide_index=True,use_container_width=True)
                st.write('Judge controls (known supported/unsupported answers)')
                st.json(report.get('judge_controls',[]))
                if report.get('judge_controls') and not all(c['passed'] for c in report['judge_controls']):
                    st.warning('Some judge controls failed. Review grading before relying on these scores.')
                for row in report['results']:
                    with st.expander(f"{row['id']} — {row['status']}"):
                        st.json(row)
            except (ValueError,KeyError,TypeError):
                st.warning('Evaluation report is invalid. Rerun the evaluation command.')
        st.caption('Observed application tool outcomes are separate from model-quality scores. Awaiting confirmation does not mean a booking succeeded.')
        with execution._connect() as c:
            rows = [dict(r) for r in c.execute('''SELECT tool_name,status,COUNT(*) AS count,
                       ROUND(AVG(duration_ms),2) AS mean_duration_ms
                       FROM agent_events WHERE event_type='goal_completed'
                       GROUP BY tool_name,status ORDER BY tool_name,status''')]
        if rows:
            st.dataframe(rows,hide_index=True,use_container_width=True)
        else:
            st.info('No completed tool events recorded yet.')
