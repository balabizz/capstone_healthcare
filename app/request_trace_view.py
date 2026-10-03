"""Explicit plans, execution outcomes and memory provenance, not model reasoning."""
import streamlit as st
from app.input_layout import REQUEST_INPUT_HEIGHT
from src.agents.scenario_testing import SCENARIOS, run_scenario


def render_breakdown(plan,steps):
    if plan.get('clarification'):
        st.info(plan['clarification'])
    outcomes = {r['id']:r for r in steps}
    rows = []
    for goal in plan['goals']:
        result = outcomes.get(goal['step'],{})
        rows.append({**goal,'status':result.get('status','planned'),
                     'duration_ms':result.get('duration_ms'), 'blocked_by':result.get('blocked_by',[])})
    if rows:
        st.dataframe(rows,hide_index=True,use_container_width=True)
        st.caption('Dependency edges: prerequisite → dependent subgoal')
        edges = [{'prerequisite':dep,'dependent':g['step']} for g in plan['goals'] for dep in g['depends_on']]
        if edges:
            st.dataframe(edges,hide_index=True,use_container_width=True)
        for row in rows:
            st.write(f"{row['step']} — {row['goal']} ({row['status']})")
            st.text(row['query'])
            st.json({k:row[k] for k in ('tool','depends_on','blocked_by','duration_ms','specialty','preferences')})


def render_request_traces(execution,requester,patient):
    with st.expander('Memory traces and planning breakdowns'):
        st.caption('Saved operational provenance: explicit planned tasks and context supplied to tools. This is not hidden model reasoning. “Supplied” does not prove the model relied on a particular turn.')
        try:
            traces = execution.request_traces.list(requester,patient)
            if not traces:
                st.info('No request traces saved for this account and selected patient yet.')
                return
            choice = st.selectbox('Inspect request',range(len(traces)),
                                  format_func=lambda i:f"{traces[i]['created_at']} · {traces[i]['request_id']}",key='inspect_request_trace')
            trace = traces[choice]
            render_breakdown(trace['plan'],trace['steps'])
            if not trace['memory']:
                st.info('No conversational memory was supplied for this request.')
            for index,lookup in enumerate(trace['memory']):
                st.write(f"{lookup['stage']}: {lookup['status']} · {lookup['count']} turns")
                st.caption(lookup.get('selection',''))
                if lookup['turns']:
                    st.dataframe(lookup['turns'],hide_index=True,use_container_width=True)
                    if st.checkbox('Show referenced conversation excerpts',key=f"trace_excerpts_{trace['request_id']}_{index}"):
                        turns = execution.conversations.get_turns(requester,patient,[t['turn_id'] for t in lookup['turns']])
                        for turn in turns:
                            st.text(f"{turn['turn_id']} · {turn['created_at']}\nQuestion: {turn['query']}\nAnswer: {turn['answer']}")
                        if len(turns)<len(lookup['turns']):
                            st.info('Some referenced turns were deleted or are no longer available.')
            if st.button('Delete saved traces for selected patient'):
                execution.request_traces.clear(requester,patient)
                st.rerun()
        except (ValueError,PermissionError) as error:
            st.warning(str(error))


def render_scenario_testing():
    with st.expander('Scenario testing sandbox'):
        st.caption('Uses an isolated temporary database and synthetic patients. Only planning uses OpenAI; medical answers and final summaries are fixtures. No real records or bookings are changed. Scenario text is sent to OpenAI; use fictional details.')
        name = st.selectbox('Scenario',list(SCENARIOS),key='scenario_name')
        query = st.text_area('Scenario request',value=SCENARIOS[name]['query'],key='scenario_query_'+name,height=REQUEST_INPUT_HEIGHT)
        if st.button('Run sandbox scenario'):
            try:
                with st.spinner('Running synthetic scenario…'):
                    result = run_scenario(name,query)
                st.session_state.scenario_result = result
            except (ValueError,PermissionError) as error:
                st.session_state.pop('scenario_result',None)
                st.error(f'Scenario did not complete: {error}')
            except Exception:
                st.session_state.pop('scenario_result',None)
                st.error('Scenario could not complete. Check the OpenAI configuration and retry.')
        result = st.session_state.get('scenario_result')
        if result:
            st.write(f"Result for: {result['scenario']}")
            st.caption(result['mode'])
            render_breakdown(result['plan'],result['steps'])
            st.write('Memory lookup provenance')
            st.json(result['memory'])
            st.json(result['checks'])
            st.write('Expected subgoals: '+', '.join(result['expected_goals']))
            if result['missing_expected_goals']:
                st.warning('Expected subgoals absent: '+', '.join(result['missing_expected_goals']))
            else:
                st.success('Expected subgoals present. This checks planning structure, not clinical correctness.')
            st.write(f"Synthetic appointments written: {result['synthetic_bookings_written']}")
