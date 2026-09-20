"""Staff performance charts; denominators and benchmark scope remain explicit."""
from datetime import datetime, timedelta, timezone
import streamlit as st
from src.repositories.performance_repository import PerformanceRepository


def percent(value):
    return 'N/A' if value is None else f'{value:.1%}'


def render_performance_dashboard(user_type, execution):
    if user_type not in ('doctor','attendant'):
        return
    with st.expander('Performance analysis dashboard',expanded=False):
        today = datetime.now(timezone.utc).date()
        start = st.date_input('Performance start date (UTC)',value=today-timedelta(days=29),key='performance_start')
        end = st.date_input('Performance end date (UTC)',value=today,key='performance_end')
        try:
            report = PerformanceRepository(execution).summarize(start,end)
        except ValueError as error:
            st.warning(str(error))
            return
        booking = report['booking']
        st.metric('Booking success rate',percent(booking['success_rate']))
        st.metric('Confirmed / attempted bookings',f"{booking['confirmed']} / {booking['attempts']}")
        st.metric('Booking latency p95',f"{booking['p95_ms']} ms" if booking['p95_ms'] is not None else 'N/A')
        st.caption('Rate = confirmed bookings / unique logged confirmation requests. Proposals are excluded. Successful requests count once; otherwise the latest outcome is used. Repeated confirmations are deduplicated; a retry can resolve an unknown outcome. Historical unlogged failures cannot be reconstructed.')
        st.write(f"Discovery proposal events (not booking attempts): {report['pending_proposals']} · replay events excluded: {booking['replays_excluded']}")
        if booking['outcomes']:
            chart([{'outcome':k,'count':v} for k,v in booking['outcomes'].items()], 'outcome','count')
            st.caption('Daily confirmed bookings and attempts (UTC)')
            chart([{'date':r['date'],'series':key,'count':r[key]} for r in report['booking_daily']
                   for key in ('attempts','confirmed')], 'date','count','series',mark='line')
            st.caption('Daily booking success rate (0–1)')
            chart(report['booking_daily'],'date','success_rate',mark='line',rate=True)
        else:
            st.info('No logged booking attempts in this period. No success rate can be calculated.')
        if report['tool_metrics']:
            st.caption('Tool success rate (success / all completed executions; pending, blocked and denied remain distinct outcomes)')
            chart(report['tool_metrics'],'tool','success_rate',rate=True)
            st.caption('Tool latency in milliseconds; missing durations excluded')
            chart([{'tool':r['tool'],'statistic':key,'ms':r[key]} for r in report['tool_metrics']
                   for key in ('mean_ms','p95_ms')], 'tool','ms','statistic')
            chart(report['tool_outcomes'],'tool','count','status')
            st.dataframe(report['tool_metrics'],use_container_width=True)
        else:
            st.info('No completed tool events in this period.')
        st.download_button('Download performance aggregates (JSON)',data=__import__('json').dumps(report,indent=2),
                           file_name='performance-aggregates.json',mime='application/json')


def render_quality_charts(report):
    """Benchmark scores must not be represented as ratings of production chat traffic."""
    summary = report['summary']
    st.caption('Latest synthetic benchmark only; independent of operational date filters. Scores are LLM-judged averages, not production clinical accuracy.')
    st.metric('Evaluation completion',f"{summary['evaluated']} / {summary['total']}")
    st.metric('Accuracy score (0–1)',str(round(summary['accuracy'],3)) if summary['accuracy'] is not None else 'N/A')
    st.metric('Answers with hallucination flagged',percent(summary['hallucination']))
    st.caption('Answer-quality scores (0–1; higher is better)')
    chart([{'task':task,'metric':metric,'score':values[metric]} for task,values in report['by_task'].items()
           for metric in ('accuracy','relevance','faithfulness')], 'task','score','metric',rate=True)
    st.caption('Hallucination rate by task (0–1; lower is better)')
    chart([{'task':task,'rate':v['hallucination']} for task,v in report['by_task'].items()], 'task','rate',rate=True)
    st.caption('Generation latency by task (ms; excludes judge time, includes deterministic abstentions)')
    chart([{'task':task,'statistic':key,'ms':v[key]} for task,v in report['by_task'].items()
           for key in ('mean_generation_latency_ms','p95_generation_latency_ms')], 'task','ms','statistic')
    chart([{'task':task,'outcome':key,'count':v[key]} for task,v in report['by_task'].items()
           for key in ('evaluated','failed')], 'task','count','outcome')


def chart(rows,x,y,color=None,mark='bar',rate=False):
    """Vega-Lite accepts JSON records directly and provides hover evidence."""
    encoding = {'x':{'field':x,'type':'temporal' if x=='date' else 'nominal'},
                'y':{'field':y,'type':'quantitative',**({'scale':{'domain':[0,1]}} if rate else {})},
                'tooltip':[{'field':key,'type':'quantitative' if key==y else 'nominal'} for key in (x,y,*([color] if color else []))]}
    if color:
        encoding['color'] = {'field':color,'type':'nominal'}
        if mark=='bar':
            encoding['xOffset'] = {'field':color}
    st.vega_lite_chart(spec={'data':{'values':rows},'mark':{'type':mark,**({'point':True} if mark=='line' else {})},
                             'encoding':encoding},use_container_width=True)
