"""Aggregate operational telemetry without exposing patients or clinical content."""
from collections import Counter, defaultdict
from datetime import date, timedelta
import math
import statistics


def latency(values):
    values = sorted(v for v in values if isinstance(v,(int,float)) and v >= 0)
    return {'mean_ms':round(statistics.mean(values),2) if values else None,
            'p95_ms':values[math.ceil(.95*len(values))-1] if values else None,
            'latency_samples':len(values)}


class PerformanceRepository:
    def __init__(self, execution):
        self.execution = execution

    def summarize(self, start, end):
        if not isinstance(start,date) or not isinstance(end,date) or start > end:
            raise ValueError('Choose a valid start/end date range.')
        bounds = (start.isoformat(),(end+timedelta(days=1)).isoformat())
        with self.execution._connect() as c:
            # Earliest confirmed success wins over repeated submissions/errors for the
            # same request; otherwise retain its latest outcome. Dedupe before filtering.
            bookings = [dict(r) for r in c.execute('''WITH ranked AS (
                SELECT status,duration_ms,created_at,COALESCE(json_extract(details_json,'$.replayed'),0) AS replayed,
                ROW_NUMBER() OVER (
                    PARTITION BY request_id,patient_id,COALESCE(json_extract(details_json,'$.requester_patient_id'),patient_id)
                    ORDER BY CASE WHEN status='success' THEN 0 ELSE 1 END,
                        CASE WHEN status='success' THEN created_at END ASC,
                        CASE WHEN status='success' THEN rowid END ASC,created_at DESC,rowid DESC
                ) AS rank FROM agent_events
                WHERE event_type='appointment_booking_completed')
                SELECT status,duration_ms,replayed,substr(created_at,1,10) AS day FROM ranked
                WHERE rank=1 AND created_at>=? AND created_at<?''',bounds)]
            tools = [dict(r) for r in c.execute('''SELECT tool_name,status,duration_ms,substr(created_at,1,10) AS day
                FROM agent_events WHERE event_type='goal_completed' AND created_at>=? AND created_at<?''',bounds)]
            replays = c.execute('''SELECT COUNT(*) FROM agent_events WHERE event_type='appointment_booking_completed'
                AND json_extract(details_json,'$.replayed')=1 AND created_at>=? AND created_at<?''',bounds).fetchone()[0]
        counts = Counter(r['status'] for r in bookings)
        daily = defaultdict(Counter)
        for row in bookings:
            daily[row['day']]['attempts'] += 1
            daily[row['day']]['confirmed'] += row['status']=='success'
        by_tool = defaultdict(list)
        for row in tools:
            by_tool[row['tool_name'] or 'unknown'].append(row)
        tool_rows = []
        statuses = []
        for tool, rows in sorted(by_tool.items()):
            status_counts = Counter(r['status'] for r in rows)
            tool_rows.append({'tool':tool,'executions':len(rows),'successes':status_counts['success'],
                             'success_rate':status_counts['success']/len(rows),
                             **latency([r['duration_ms'] for r in rows])})
            statuses.extend({'tool':tool,'status':status,'count':count} for status,count in sorted(status_counts.items()))
        return {'start':start.isoformat(),'end':end.isoformat(),'timezone':'UTC',
                'booking':{'attempts':len(bookings),'confirmed':counts['success'],
                           'success_rate':counts['success']/len(bookings) if bookings else None,
                           'outcomes':dict(counts),'replays_excluded':replays-sum(bool(r['replayed']) for r in bookings),
                           **latency([r['duration_ms'] for r in bookings])},
                'booking_daily':[{'date':day,'attempts':v['attempts'],'confirmed':v['confirmed'],
                                  'success_rate':v['confirmed']/v['attempts']} for day,v in sorted(daily.items())],
                'tool_metrics':tool_rows,'tool_outcomes':statuses,
                'pending_proposals':sum(r['status']=='awaiting_confirmation' for r in tools if r['tool_name']=='appointments.discover')}
