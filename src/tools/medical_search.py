"""Live PubMed/MEDLINE and WHO publication retrieval, with bounded evidence."""
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from threading import Lock
from time import monotonic, sleep
from urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
import requests

from src.config import NCBI_API_KEY, NCBI_EMAIL, MEDICAL_SEARCH_DAYS

_NCBI_LOCK = Lock()
_NCBI_LAST = 0.0


class ProviderError(RuntimeError):
    pass


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        if tag in ('p', 'div', 'li', 'br'):
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain(value):
    parser = PlainText()
    parser.feed(value or '')
    return re.sub(r'\s+', ' ', ''.join(parser.parts)).strip()


def safe_topic(query):
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 300:
        raise ValueError('Use a short, general disease or treatment topic for external search.')
    # Reject obvious identifiers/person-specific requests at the external boundary.
    # This is a conservative guard, not a general-purpose de-identification engine.
    if (re.search(r'[@:/\\]|\b(?:my|father|mother|spouse|wife|husband|daughter|son|patient[_ -]?id|dob|mr|mrs)\b', query, re.I)
        or re.search(r'\d{5,}', query) or '[' in query or ']' in query):
        raise ValueError('External search requires a general medical topic without personal details or identifiers.')
    topic = re.sub(r'\b(?:find|search|latest|current|recent|summarize|summarise|publications|publication|please|information|about|what|are|the|for)\b',
                   ' ', query, flags=re.I)
    topic = re.sub(r'\s+', ' ', topic).strip(' .?!')
    if len(topic) < 3:
        raise ValueError('Please specify a medical condition or intervention.')
    return topic


class MedicalSearch:
    def __init__(self, session=None, days=MEDICAL_SEARCH_DAYS, limit=5, now=None, throttle=True):
        if type(days) is not int or not 1 <= days <= 3650 or type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError('Invalid medical search window or result limit.')
        self.session = session or requests.Session()
        self.days, self.limit, self.throttle = days, limit, throttle
        self.now = now or (lambda: datetime.now(timezone.utc))

    def _get(self, url, params, ncbi=False):
        global _NCBI_LAST
        if ncbi and self.throttle:
            with _NCBI_LOCK:
                sleep(max(0, .35 - (monotonic() - _NCBI_LAST)))
                _NCBI_LAST = monotonic()
        # No arbitrary URLs, provider redirects or unbounded downloads.
        try:
            response = self.session.get(url, params=params, timeout=(5, 20), stream=True, allow_redirects=False)
            try:
                if response.status_code != 200:
                    raise ProviderError('Provider returned an unsuccessful response; retry later.')
                chunks, size = [], 0
                for chunk in response.iter_content(8192):
                    size += len(chunk)
                    if size > 2_000_000:
                        raise ProviderError('Provider response exceeded the size limit.')
                    chunks.append(chunk)
                return b''.join(chunks)
            finally:
                response.close()
        except requests.RequestException as error:
            raise ProviderError('Provider could not be reached; retry later.') from error

    def _pubmed(self, query, first, last):
        shared = {'db': 'pubmed', 'tool': 'capstone_healthcare'}
        if NCBI_EMAIL:
            shared['email'] = NCBI_EMAIL
        if NCBI_API_KEY:
            shared['api_key'] = NCBI_API_KEY
        search = json.loads(self._get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', {
            **shared, 'term': f'({query}) NOT ("Retracted Publication"[pt] OR "Retraction of Publication"[pt])',
            'retmode': 'json', 'retmax': self.limit, 'sort': 'pub_date', 'datetype': 'pdat',
            'mindate': first.strftime('%Y/%m/%d'), 'maxdate': last.strftime('%Y/%m/%d')}, ncbi=True))
        if 'esearchresult' not in search or search.get('error') or search['esearchresult'].get('errorlist'):
            raise ProviderError('PubMed could not interpret the search topic.')
        data = search['esearchresult']
        ids = data['idlist']
        if not isinstance(ids, list) or any(not isinstance(i, str) or not i.isdigit() for i in ids):
            raise ProviderError('PubMed returned invalid record identifiers.')
        ids = ids[:self.limit]
        if not ids:
            return [], int(data['count'])
        xml = self._get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
            {**shared, 'id': ','.join(ids), 'retmode': 'xml'}, ncbi=True)
        if b'<!ENTITY' in xml.upper():
            raise ProviderError('Unsupported XML entity declaration.')
        root = ET.fromstring(xml)
        if root.tag != 'PubmedArticleSet':
            raise ProviderError('Unexpected PubMed response.')
        sources = []
        def text(node):
            return ''.join(node.itertext()).strip() if node is not None else ''
        for article in root.findall('PubmedArticle'):
            pmid = article.findtext('./MedlineCitation/PMID')
            if pmid not in ids:
                continue
            publication_types = [text(n) for n in article.findall('.//PublicationType')]
            if set(publication_types) & {'Retracted Publication', 'Retraction of Publication'} or article.find(".//CommentsCorrections[@RefType='RetractionIn']") is not None:
                continue
            title = text(article.find('./MedlineCitation/Article/ArticleTitle'))
            abstract = '\n'.join((n.get('Label', '') + ': ' if n.get('Label') else '') + text(n)
                                 for n in article.findall('./MedlineCitation/Article/Abstract/AbstractText'))
            published = article.find('./MedlineCitation/Article/ArticleDate')
            if published is not None:
                stamp = '-'.join(published.findtext(p, '') for p in ('Year','Month','Day'))
            else:
                published = article.find('./MedlineCitation/Article/Journal/JournalIssue/PubDate')
                stamp = text(published.find('MedlineDate')) if published is not None else ''
                if not stamp and published is not None:
                    stamp = ' '.join(published.findtext(p, '') for p in ('Year','Month','Day')).strip()
            sources.append({'id': f'pubmed:{pmid}', 'provider': 'PubMed',
                'index_status': article.find('./MedlineCitation').get('Status', 'unknown'),
                'title': title[:1000], 'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                'published': stamp or None, 'publication_types': publication_types,
                'journal': text(article.find('./MedlineCitation/Article/Journal/Title'))[:500],
                'evidence_type': 'abstract' if abstract else 'metadata_only',
                'excerpt': abstract[:6000], 'truncated': len(abstract) > 6000})
        return sources, int(data['count'])

    def _who(self, query, first, last):
        # Search the disease terms in titles/overviews; omit generic request verbs.
        stop = {'latest','current','recent','treatment','treatments','methods','options','guidelines',
                'summarize','find','search','for','the','and','of','what','are','is','in','new','information',
                'publication','publications','evidence','research','literature','advances'}
        terms = [word.lower() for word in re.findall(r'[A-Za-z0-9-]+', query) if word.lower() not in stop][:12]
        if not terms:
            raise ValueError('Please specify a medical condition or topic.')
        clauses = [f"(contains(tolower(Title),'{term}') or contains(tolower(Overview),'{term}'))" for term in terms]
        dates = f'PublicationDateAndTime ge {first.isoformat()}T00:00:00Z and PublicationDateAndTime le {last.isoformat()}T23:59:59Z'
        data = json.loads(self._get('https://www.who.int/api/hubs/publications', {
            '$filter': ' and '.join(clauses + [dates]), '$orderby': 'PublicationDateAndTime desc',
            '$top': self.limit, '$select': 'Id,Title,UrlName,Overview,Summary,PublicationDateAndTime', '$count': 'true'}))
        if not isinstance(data.get('value'), list):
            raise ProviderError('Unexpected WHO response.')
        sources = []
        for row in data['value'][:self.limit]:
            if not row.get('Id') or not row.get('Title') or not row.get('UrlName'):
                continue
            excerpt = plain(row.get('Overview') or row.get('Summary'))
            sources.append({'id': 'who:' + str(row['Id']), 'provider': 'WHO',
                'title': plain(row['Title'])[:1000],
                'url': 'https://www.who.int/publications/i/item/' + quote(row['UrlName'], safe=''),
                'published': row.get('PublicationDateAndTime'), 'publication_types': ['WHO publication'],
                'evidence_type': 'publication_overview' if excerpt else 'metadata_only',
                'excerpt': excerpt[:6000], 'truncated': len(excerpt) > 6000})
        return sources, data.get('@odata.count')

    def search(self, query):
        query = safe_topic(query)
        now = self.now()
        last, first = now.date(), now.date() - timedelta(days=self.days)
        sources, providers = [], []
        for name, operation in (('PubMed', self._pubmed), ('WHO', self._who)):
            try:
                found, count = operation(query, first, last)
                sources.extend(found)
                providers.append({'provider': name, 'status': 'success' if found else 'no_results',
                                  'returned': len(found), 'matching_count': count})
            except (ProviderError, ValueError, KeyError, TypeError, ET.ParseError) as error:
                providers.append({'provider': name, 'status': 'failed', 'returned': 0,
                                  'error': str(error) if isinstance(error, ProviderError) else 'Provider returned malformed or unsupported data.'})
        unique = {s['id']: s for s in sources}
        failures = sum(p['status'] == 'failed' for p in providers)
        status = 'failed' if failures == len(providers) else 'partial' if failures else 'success' if unique else 'no_results'
        return {'status': status, 'query': query, 'searched_at': now.isoformat(),
                'date_from': first.isoformat(), 'date_to': last.isoformat(),
                'providers': providers, 'sources': list(unique.values())}
