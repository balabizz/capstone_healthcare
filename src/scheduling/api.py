"""Private Doctor Schedule API backed by the existing SQLite appointments table.

Run: python -m src.scheduling.api --port 8765
Bearer authentication identifies the trusted application, which supplies identities
from its authenticated session. This is not a public patient-facing API.
"""
import argparse
import hmac
import json
import sqlite3
from wsgiref.simple_server import make_server
from src.config import SQLITE_DB_PATH, SCHEDULE_API_TOKEN
from src.repositories.schedule_repository import ScheduleRepository


def create_app(database_path=SQLITE_DB_PATH, token=SCHEDULE_API_TOKEN):
    if len(token) < 24:
        raise ValueError('Configure a Schedule API service token of at least 24 characters.')
    repository = ScheduleRepository(database_path)
    routes = {'/specialists': repository.specialists, '/slots': repository.discover, '/bookings': repository.book}

    def app(environ, start_response):
        status, output = '200 OK', None
        if not hmac.compare_digest(environ.get('HTTP_AUTHORIZATION', ''), 'Bearer ' + token):
            status, output = '403 Forbidden', {'error': 'Access denied'}
        elif environ.get('REQUEST_METHOD') != 'POST' or environ.get('PATH_INFO') not in routes:
            status, output = '404 Not Found', {'error': 'Unknown endpoint'}
        else:
            try:
                length = int(environ.get('CONTENT_LENGTH') or 0)
                if not 0 < length <= 16384:
                    raise ValueError('Invalid request size')
                body = json.loads(environ['wsgi.input'].read(length))
                if not isinstance(body, dict):
                    raise ValueError('Object required')
                output = routes[environ['PATH_INFO']](**body)
            except PermissionError:
                status, output = '403 Forbidden', {'error': 'Access denied'}
            except (ValueError, TypeError, sqlite3.IntegrityError):
                status, output = '400 Bad Request', {'error': 'Invalid scheduling request'}
            except Exception:
                status, output = '503 Service Unavailable', {'error': 'Scheduling failed; retry with the same request key'}
        data = json.dumps(output).encode()
        start_response(status, [('Content-Type', 'application/json'), ('Content-Length', str(len(data)))])
        return [data]
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--database', default=SQLITE_DB_PATH)
    args = parser.parse_args()
    with make_server('127.0.0.1', args.port, create_app(args.database)) as server:
        print(f'Doctor Schedule API listening on http://127.0.0.1:{args.port}')
        server.serve_forever()
