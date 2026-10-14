"""Shared deterministic preprocessing. No network lookups during inference."""
import hashlib
import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit
import tldextract

# Version-pinned package snapshot, including private suffixes such as github.io.
PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
PREPROCESSING_VERSION = '2026-10-14-v1'

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def normalize_url(value):
    if not isinstance(value, str):
        raise ValueError('URL must be a string.')
    value = value.strip()
    if not value or len(value) > 8192 or re.search(r'[\s\x00-\x1f\x7f\\]', value):
        raise ValueError('Enter a valid HTTP(S) URL without spaces or control characters.')
    if re.search(r'%(?![0-9a-fA-F]{2})', value):
        raise ValueError('URL contains an invalid percent escape.')
    if not re.match(r'^https?://', value, re.I):
        if '://' in value or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', value):
            # A numeric port on a schemeless hostname is permitted.
            if not re.match(r'^[^/:]+\.([^/:]+):\d+(?:/|$)', value):
                raise ValueError('Only HTTP and HTTPS URLs are supported.')
        value = 'https://' + value
    try:
        p = urlsplit(value)
        if p.scheme.lower() not in ('http', 'https') or not p.hostname:
            raise ValueError('URL must have a hostname.')
        hostname = p.hostname.rstrip('.').lower()
        try:
            ip = ipaddress.ip_address(hostname)
            hostname = f'[{ip.compressed}]' if ip.version == 6 else ip.compressed
        except ValueError:
            hostname = hostname.encode('idna').decode('ascii')
            labels = hostname.split('.')
            if len(labels) < 2 or len(hostname) > 253 or not all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', x) for x in labels):
                raise ValueError('URL must contain a valid domain name or IP address.')
            if re.fullmatch(r'[0-9.]+', hostname):
                raise ValueError('Invalid IP address.')
        port = p.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError('Invalid port.')
        authority = hostname
        if port is not None and (p.scheme.lower(), port) not in [('https',443),('http',80)]:
            authority += f':{port}'
        # Preserve user information for threat signals, never request the target.
        if '@' in p.netloc:
            authority = p.netloc.rsplit('@',1)[0] + '@' + authority
        path = '' if p.path in ('','/') else p.path
        return urlunsplit((p.scheme.lower(), authority, path, p.query, p.fragment))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid HTTP(S) URL: {exc}') from exc

def url_group(value):
    host = urlsplit(normalize_url(value)).hostname
    ext = PSL(host)
    return ext.top_domain_under_public_suffix or host

def url_model_text(value):
    """Host-only lexical model: avoid path/source shortcut in the available corpus.

    Paths still undergo syntax validation and are shown as scope limitations.
    This cannot detect malicious pages hosted on an otherwise benign domain.
    """
    host = urlsplit(normalize_url(value)).hostname
    return host.removeprefix('www.')

def normalize_sms(value):
    if not isinstance(value, str):
        raise ValueError('Message must be a string.')
    value = ' '.join(unicodedata.normalize('NFKC', value).split())
    if not value or len(value) > 10000:
        raise ValueError('Message must contain 1 to 10000 characters.')
    return value

def sms_template(value):
    value = normalize_sms(value).casefold()
    value = re.sub(r'https?://\S+|www\.\S+', ' URL ', value)
    value = re.sub(r'[\w.+-]+@[\w.-]+\.[a-z]+', ' EMAIL ', value)
    value = re.sub(r'\d+(?:[.,:/-]\d+)*', ' NUMBER ', value)
    value = ''.join(c if c.isalnum() or unicodedata.category(c).startswith('M') else ' ' for c in value)
    return ' '.join(value.split())

def split_for(group):
    bucket = int(digest('threatguard-split-v1:'+group)[:8],16) % 100
    return 'train' if bucket < 70 else 'selection' if bucket < 80 else 'calibration' if bucket < 90 else 'test'
