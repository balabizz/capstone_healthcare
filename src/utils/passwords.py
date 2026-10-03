"""Password hashing for provisioned staff; legacy patient/doctor compatibility."""
import hashlib
import hmac
import secrets


def hash_password(password):
    if not isinstance(password, str) or len(password) < 12:
        raise ValueError('Use a password of at least 12 characters.')
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 600000).hex()
    return f'pbkdf2_sha256$600000${salt}${digest}'


def verify_password(password, stored, *, allow_legacy=False):
    if not stored.startswith('pbkdf2_sha256$'):
        return allow_legacy and hmac.compare_digest(password.encode(), stored.encode())
    try:
        _, rounds, salt, expected = stored.split('$')
        rounds = int(rounds)
        if not 100000 <= rounds <= 2000000:
            return False
        actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), rounds).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False
