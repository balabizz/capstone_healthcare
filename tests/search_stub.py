"""Unavailable provider stub for workflow tests unrelated to networking."""
class OfflineMedicalSearch:
    def search(self, query):
        return {'status':'failed','query':query,'searched_at':'2026-09-20T00:00:00+00:00',
                'date_from':'2024-09-20','date_to':'2026-09-20','sources':[],
                'providers':[{'provider':'PubMed','status':'failed','returned':0},
                             {'provider':'WHO','status':'failed','returned':0}]}
