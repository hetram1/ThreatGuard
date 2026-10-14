import pytest
from ml.reliable.preprocessing import (
    normalize_url,
    normalize_sms,
    url_group,
    sms_template,
    split_for,
)


@pytest.mark.parametrize(
    "value",
    [
        "not a url",
        "javascript:alert(1)",
        "https://",
        "hello world",
        "ftp://example.com",
        "https://a..com",
        "https://-bad.com",
        "https://example.com:99999",
        "https://example.com/%GG",
        "https://999.999.999.999",
        "https://example.com\\@evil.example",
        "https://example.com/\x00",
    ],
)
def test_invalid_urls(value):
    with pytest.raises(ValueError):
        normalize_url(value)


@pytest.mark.parametrize(
    "value",
    [
        "github.com",
        "github.com/",
        "https://github.com",
        "HTTPS://GITHUB.COM/",
        "https://github.com:443/",
    ],
)
def test_equivalent_roots(value):
    assert normalize_url(value) == "https://github.com"


def test_preserve_meaningful_components():
    assert (
        normalize_url("http://example.com:8080/a?x=1#frag")
        == "http://example.com:8080/a?x=1#frag"
    )
    assert normalize_url("https://example.com/a") != normalize_url(
        "https://example.com/a/"
    )
    assert normalize_url("https://user@example.com/a") == "https://user@example.com/a"
    assert normalize_url("https://[2001:db8::1]:443/") == "https://[2001:db8::1]"


def test_public_and_private_suffix_groups():
    assert (
        url_group("https://a.example.co.uk")
        == url_group("https://b.example.co.uk")
        == "example.co.uk"
    )
    assert url_group("https://alice.github.io") != url_group("https://bob.github.io")


def test_message_normalization():
    assert normalize_sms("  hello\nworld ") == "hello world"
    assert sms_template(
        "Your code is 1234. Visit https://one.example/a"
    ) == sms_template("Your code is 9876! Visit https://two.example/b")
    assert "आपका" in sms_template("आपका ओटीपी 1234 है।")
    with pytest.raises(ValueError):
        normalize_sms(" \n ")


def test_grouping_is_transitive():
    from ml.reliable.prepare import near_groups

    texts = [
        "your code is NUMBER do not share this code with anyone",
        "your code is NUMBER do not share this code with anyone",
        "lets meet for lunch tomorrow",
    ]
    groups, _ = near_groups(texts)
    assert groups[0] == groups[1] and groups[0] != groups[2]
    assert split_for(groups[0]) == split_for(groups[1])
