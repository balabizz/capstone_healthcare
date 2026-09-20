"""Observable context provenance; never model chain-of-thought or clinical content."""


def memory_trace(stage, turns, status='supplied'):
    return {'stage':stage,'status':status if turns else 'empty',
            'selection':'Recent continuity plus keyword-related older turns; scoped to requester and patient.',
            'count':len(turns),'turns':[{'turn_id':r['turn_id'],'created_at':r['created_at'],
                'selection_reason':r.get('selection_reason','selected'),
                'truncated':r.get('truncated',False),'question_characters':len(r['query']),
                'answer_characters':len(r['answer'])} for r in turns]}
