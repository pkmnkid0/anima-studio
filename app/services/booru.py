"""
Thin wrappers around each booru site's public JSON search API. These are
the same documented, public, unauthenticated endpoints each site's own
search page uses - this module just lets the app call them from server
side and normalize the results into one shape the rest of the app works
with, the same way tools like gallery-dl or Grabber do.

Nothing here bypasses a site's own content gating: the "rating" filter
is translated to each site's own rating tag syntax (the same tag you'd
type into that site's search box yourself), and defaults to the
safest/most restrictive option. Explicit/questionable content is only
ever fetched if the user explicitly widens the rating filter themselves.
"""
from __future__ import annotations

import re

import requests

from . import settings_store

USER_AGENT = "AnimaLoRAStudio/1.0 (dataset collection tool; +local use)"
REQUEST_TIMEOUT = 20
AUTOCOMPLETE_TIMEOUT = 8

# Gelbooru (Aug 2025) and Rule34.xxx (Aug 19 2025) both locked their public
# dapi/index.php search endpoints behind an account - every request now
# needs `&api_key=...&user_id=...` from the requester's own account
# settings page, or the site answers with a 401/"authentication required"
# error instead of results. Danbooru, Safebooru, e621, Konachan and
# yande.re are unaffected and still work fully anonymously.
SOURCES_REQUIRING_CREDENTIALS = {"gelbooru", "rule34"}
CREDENTIAL_SIGNUP_URL = {
    "gelbooru": "https://gelbooru.com/index.php?page=account&s=options",
    "rule34": "https://rule34.xxx/index.php?page=account&s=options",
}

# Danbooru/e621 share this numeric tag-category scheme.
_DANBOORU_CATEGORY = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}

# Normalized rating -> per-site rating tag(s)
_RATING_MAP = {
    "danbooru": {"safe": "rating:g", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
    "gelbooru": {"safe": "rating:safe", "sensitive": "rating:sensitive",
                 "questionable": "rating:questionable", "explicit": "rating:explicit"},
    "safebooru": {"safe": "rating:safe", "sensitive": "rating:safe",
                  "questionable": "rating:questionable", "explicit": "rating:explicit"},
    "e621": {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
    "rule34": {"safe": "rating:safe", "sensitive": "rating:safe",
               "questionable": "rating:questionable", "explicit": "rating:explicit"},
    # moebooru sites (konachan/yandere) only have three tiers - "sensitive"
    # maps to the same tag as "safe" since there's no separate tier for it.
    "konachan": {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
    "yandere": {"safe": "rating:s", "sensitive": "rating:s", "questionable": "rating:q", "explicit": "rating:e"},
}

SOURCES = ["danbooru", "gelbooru", "safebooru", "e621", "rule34", "konachan", "yandere"]


class BooruError(RuntimeError):
    pass


class CredentialsRequiredError(BooruError):
    """Raised instead of making a doomed request when a source requires
    an api_key/user_id we don't have configured yet."""

    def __init__(self, source: str):
        signup = CREDENTIAL_SIGNUP_URL.get(source, "")
        super().__init__(
            f"{source.capitalize()} now requires a free account's API key & user ID for every "
            f"search (a site-wide change, not something this app can work around). "
            f"Get yours from {signup} and add it under Collect \u2192 API keys."
        )
        self.source = source


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _credentials(source: str) -> dict:
    return settings_store.get_booru_credentials().get(source, {})


def _require_credentials(source: str) -> dict:
    creds = _credentials(source)
    if not creds.get("api_key") or not creds.get("user_id"):
        raise CredentialsRequiredError(source)
    return creds


def _split_tags(field: str) -> list[str]:
    return [t for t in (field or "").split() if t]


def _normalize_danbooru(post: dict) -> dict | None:
    file_url = post.get("file_url") or post.get("large_file_url")
    if not file_url:
        return None
    rating_code = post.get("rating", "g")
    rating = {"g": "safe", "s": "sensitive", "q": "questionable", "e": "explicit"}.get(rating_code, "unknown")
    return {
        "id": post["id"],
        "source": "danbooru",
        "post_url": f"https://danbooru.donmai.us/posts/{post['id']}",
        "file_url": file_url,
        "preview_url": post.get("preview_file_url") or file_url,
        "width": post.get("image_width"),
        "height": post.get("image_height"),
        "rating": rating,
        "score": post.get("score", 0),
        "extension": post.get("file_ext", ""),
        "tags": {
            "general": _split_tags(post.get("tag_string_general", "")),
            "character": _split_tags(post.get("tag_string_character", "")),
            "copyright": _split_tags(post.get("tag_string_copyright", "")),
            "artist": _split_tags(post.get("tag_string_artist", "")),
            "meta": _split_tags(post.get("tag_string_meta", "")),
        },
    }


_GELBOORU_LIKE_DOMAINS = {
    "gelbooru": "gelbooru.com",
    "safebooru": "safebooru.org",
    "rule34": "rule34.xxx",
}


def _normalize_gelbooru_like(post: dict, source: str) -> dict | None:
    file_url = post.get("file_url")
    if not file_url:
        return None
    rating_raw = str(post.get("rating", "safe")).lower()
    rating = {"general": "safe", "safe": "safe", "sensitive": "sensitive",
              "questionable": "questionable", "explicit": "explicit"}.get(rating_raw, "unknown")
    # Gelbooru/Safebooru/Rule34 don't split tags by category in the basic
    # API - everything comes back as one space-separated string.
    all_tags = _split_tags(post.get("tags", ""))
    domain = _GELBOORU_LIKE_DOMAINS.get(source, "gelbooru.com")
    return {
        "id": post.get("id"),
        "source": source,
        "post_url": f"https://{domain}/index.php?page=post&s=view&id={post.get('id')}",
        "file_url": file_url,
        "preview_url": post.get("preview_url") or file_url,
        "width": post.get("width"),
        "height": post.get("height"),
        "rating": rating,
        "score": post.get("score", 0),
        "extension": (file_url.rsplit(".", 1)[-1] if "." in file_url else ""),
        "tags": {"general": all_tags, "character": [], "copyright": [], "artist": [], "meta": []},
    }


def _normalize_moebooru(post: dict, source: str) -> dict | None:
    """Konachan/yande.re - same underlying "moebooru" software, tags come
    back as one space-separated string with no category split, and a
    3-tier rating (s/q/e - no separate "sensitive" tier)."""
    file_url = post.get("file_url") or post.get("jpeg_url")
    if not file_url:
        return None
    rating = {"s": "safe", "q": "questionable", "e": "explicit"}.get(post.get("rating"), "unknown")
    domain = "konachan.com" if source == "konachan" else "yande.re"
    return {
        "id": post.get("id"),
        "source": source,
        "post_url": f"https://{domain}/post/show/{post.get('id')}",
        "file_url": file_url,
        "preview_url": post.get("preview_url") or file_url,
        "width": post.get("width"),
        "height": post.get("height"),
        "rating": rating,
        "score": post.get("score", 0),
        "extension": (file_url.rsplit(".", 1)[-1] if "." in file_url else ""),
        "tags": {"general": _split_tags(post.get("tags", "")), "character": [], "copyright": [], "artist": [], "meta": []},
    }


def _normalize_e621(post: dict) -> dict | None:
    file_info = post.get("file", {})
    file_url = file_info.get("url")
    if not file_url:
        return None
    rating = {"s": "safe", "q": "questionable", "e": "explicit"}.get(post.get("rating"), "unknown")
    tags = post.get("tags", {})
    return {
        "id": post.get("id"),
        "source": "e621",
        "post_url": f"https://e621.net/posts/{post.get('id')}",
        "file_url": file_url,
        "preview_url": (post.get("preview", {}) or {}).get("url") or file_url,
        "width": file_info.get("width"),
        "height": file_info.get("height"),
        "rating": rating,
        "score": (post.get("score", {}) or {}).get("total", 0),
        "extension": file_info.get("ext", ""),
        "tags": {
            "general": tags.get("general", []),
            "character": tags.get("character", []),
            "copyright": tags.get("copyright", []),
            "artist": tags.get("artist", []),
            "meta": tags.get("meta", []) + tags.get("species", []) + tags.get("lore", []),
        },
    }


_DANBOORU_ANON_TAG_LIMIT = 2
_DANBOORU_EXTRA_TAG_MAX_PAGES = 5  # cap on extra pages fetched hunting for post-filter matches


def _post_matches_extra_tags(post: dict, extra_tags: list[str]) -> bool:
    """Whether a normalized Danbooru post's own tags satisfy every tag
    beyond the 2 Danbooru's anonymous API actually accepted - this is how
    more than 2 tags get supported at all: extra tags aren't sent to the
    API (it would just reject the request), they're checked against each
    post's own returned tag list instead, which Danbooru already gives us
    for free with every result. Supports Danbooru's "-tag" exclude syntax
    the same way the API itself does."""
    all_post_tags = {t.lower() for tags in post.get("tags", {}).values() for t in tags}
    for tag in extra_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag.startswith("-"):
            if tag[1:].lower() in all_post_tags:
                return False
        elif tag.lower() not in all_post_tags:
            return False
    return True


def search(source: str, tags: str, rating: str = "safe", limit: int = 40, page: int = 1) -> list[dict]:
    """Search a booru source. `rating` is one of safe/sensitive/questionable/
    explicit/any. Returns a list of normalized post dicts.

    Danbooru specifically only accepts 2 tags per search for anonymous
    (no API key) requests - more than that here doesn't fail, it sends
    the rating tag plus your first tag to the API (keeping rating
    enforcement server-side, since that's safety-relevant) and applies
    any further tags as a post-filter against each result's own tags,
    fetching a few extra pages as needed to still fill up to `limit`
    matches - which is why a search with more tags can take a bit longer."""
    if source not in SOURCES:
        raise BooruError(f"Unknown source: {source}")
    limit = max(1, min(int(limit), 200))
    page = max(1, int(page))

    user_tags = (tags or "").strip().split()
    rating_tag = _RATING_MAP[source].get(rating) if rating != "any" else None
    query = " ".join(([rating_tag] if rating_tag else []) + user_tags)

    session = _session()
    try:
        if source == "danbooru":
            api_tags = ([rating_tag] if rating_tag else []) + user_tags
            extra_tags = []
            if len(api_tags) > _DANBOORU_ANON_TAG_LIMIT:
                extra_tags = api_tags[_DANBOORU_ANON_TAG_LIMIT:]
                api_tags = api_tags[:_DANBOORU_ANON_TAG_LIMIT]
            api_query = " ".join(api_tags)

            if not extra_tags:
                resp = session.get(
                    "https://danbooru.donmai.us/posts.json",
                    params={"tags": api_query, "limit": limit, "page": page},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                posts = resp.json()
                if isinstance(posts, dict) and "success" in posts and not posts.get("success", True):
                    raise BooruError(posts.get("message", "Danbooru search failed"))
                return [p for p in (_normalize_danbooru(p) for p in posts) if p]

            # More than 2 tags: page through the 2-tag API query ourselves,
            # post-filtering each batch, until `limit` matches are found or
            # we hit the page cap - `page` becomes "the Nth page of
            # already-filtered results" from the caller's perspective.
            matched: list[dict] = []
            api_page = 1
            skip = (page - 1) * limit
            for _ in range(_DANBOORU_EXTRA_TAG_MAX_PAGES):
                resp = session.get(
                    "https://danbooru.donmai.us/posts.json",
                    params={"tags": api_query, "limit": 200, "page": api_page},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                raw_posts = resp.json()
                if isinstance(raw_posts, dict) and "success" in raw_posts and not raw_posts.get("success", True):
                    raise BooruError(raw_posts.get("message", "Danbooru search failed"))
                if not raw_posts:
                    break
                for raw in raw_posts:
                    normalized = _normalize_danbooru(raw)
                    if normalized and _post_matches_extra_tags(normalized, extra_tags):
                        matched.append(normalized)
                if len(matched) >= skip + limit:
                    break
                api_page += 1
            return matched[skip:skip + limit]

        if source in ("gelbooru", "safebooru"):
            if source == "gelbooru":
                base = "https://gelbooru.com/index.php"
                params = {**_require_credentials("gelbooru")}
            else:
                base = "https://safebooru.org/index.php"
                params = {}
            resp = session.get(
                base,
                params={"page": "dapi", "s": "post", "q": "index", "json": 1,
                        "tags": query, "limit": limit, "pid": page - 1, **params},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 401:
                raise CredentialsRequiredError(source)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("@attributes", {}).get("count") is None and "post" not in data and source == "gelbooru":
                # Gelbooru answers auth failures with a 200 + {"error": "..."}
                # body rather than a proper HTTP error code in some cases.
                err = data.get("error") if isinstance(data.get("error"), str) else None
                if err:
                    raise BooruError(f"Gelbooru: {err}")
            posts = data.get("post", []) if isinstance(data, dict) else (data or [])
            if isinstance(posts, dict):
                posts = [posts]
            return [p for p in (_normalize_gelbooru_like(p, source) for p in posts) if p]

        if source == "rule34":
            creds = _require_credentials("rule34")
            resp = session.get(
                "https://api.rule34.xxx/index.php",
                params={"page": "dapi", "s": "post", "q": "index", "json": 1,
                        "tags": query, "limit": limit, "pid": page - 1, **creds},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 401:
                raise CredentialsRequiredError(source)
            resp.raise_for_status()
            data = resp.json()
            posts = data if isinstance(data, list) else (data.get("post", []) if isinstance(data, dict) else [])
            if isinstance(posts, dict):
                posts = [posts]
            return [p for p in (_normalize_gelbooru_like(p, "rule34") for p in posts) if p]

        if source == "e621":
            resp = session.get(
                "https://e621.net/posts.json",
                params={"tags": query, "limit": limit, "page": page},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("posts", [])
            return [p for p in (_normalize_e621(p) for p in posts) if p]

        if source in ("konachan", "yandere"):
            domain = "konachan.com" if source == "konachan" else "yande.re"
            resp = session.get(
                f"https://{domain}/post.json",
                params={"tags": query, "limit": limit, "page": page},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            posts = resp.json()
            return [p for p in (_normalize_moebooru(p, source) for p in posts) if p]
    except requests.exceptions.ConnectionError as exc:
        if "NameResolutionError" in str(exc) or "getaddrinfo" in str(exc):
            raise BooruError(
                f"Couldn't look up {source}'s address - this is a DNS problem, not something "
                f"wrong with Anima Studio itself. Usually either your internet is down, or "
                f"your ISP/network is blocking that specific site (common in some regions for "
                f"booru-style sites). Try switching your DNS to 8.8.8.8 (Google) or 1.1.1.1 "
                f"(Cloudflare), or open the site directly in a browser to check."
            ) from exc
        raise BooruError(
            f"Couldn't connect to {source} - check your internet connection, or a firewall/VPN "
            f"may be blocking it."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise BooruError(f"Request to {source} failed: {exc}") from exc
    except ValueError as exc:
        # The request succeeded (2xx) but the body wasn't valid JSON - most
        # often a Cloudflare/bot-check challenge page or a rate-limit
        # notice instead of the real API response.
        raise BooruError(
            f"{source} returned an unexpected (non-JSON) response - it may be rate-limiting "
            "or blocking automated requests right now. Try again in a bit, or a different source."
        ) from exc

    raise BooruError(f"Unhandled source: {source}")


def credentials_status() -> dict:
    """{"gelbooru": {"required": True, "configured": False, "signup_url": "..."}, ...}
    for every source that needs an api_key/user_id, for the settings panel."""
    configured = settings_store.get_booru_credentials()
    out = {}
    for source in SOURCES_REQUIRING_CREDENTIALS:
        creds = configured.get(source, {})
        out[source] = {
            "required": True,
            "configured": bool(creds.get("api_key") and creds.get("user_id")),
            "api_key": creds.get("api_key", ""),
            "user_id": creds.get("user_id", ""),
            "signup_url": CREDENTIAL_SIGNUP_URL.get(source, ""),
        }
    return out


def flatten_tags(tags: dict, order: tuple[str, ...] = ("artist", "copyright", "character", "general", "meta")) -> list[str]:
    """Flatten a category-split tag dict into a single ordered list,
    matching the conventional booru caption ordering (artist, series,
    character, then general descriptive tags, meta last)."""
    out: list[str] = []
    for cat in order:
        out.extend(tags.get(cat, []))
    return out


def autocomplete(source: str, query: str, limit: int = 10) -> list[dict]:
    """Live tag suggestions as the user types - the same feature each
    site's own search box offers. Returns normalized
    {name, category, post_count} entries. Any parsing surprise from a
    given site's response degrades to an empty list rather than raising,
    so a shape mismatch on one site never breaks typing."""
    query = (query or "").strip()
    if not query or source not in SOURCES:
        return []
    limit = max(1, min(int(limit), 20))
    session = _session()

    try:
        if source == "danbooru":
            resp = session.get(
                "https://danbooru.donmai.us/autocomplete.json",
                params={"search[query]": query, "search[type]": "tag_query", "limit": limit},
                timeout=AUTOCOMPLETE_TIMEOUT,
            )
            resp.raise_for_status()
            out = []
            for it in resp.json():
                if not isinstance(it, dict):
                    continue
                name = it.get("value") or it.get("label")
                if not name:
                    continue
                out.append({
                    "name": name,
                    "category": _DANBOORU_CATEGORY.get(it.get("category"), "general"),
                    "post_count": it.get("post_count"),
                })
            return out[:limit]

        if source == "e621":
            resp = session.get(
                "https://e621.net/tags/autocomplete.json",
                params={"search[name_matches]": query},
                timeout=AUTOCOMPLETE_TIMEOUT,
            )
            resp.raise_for_status()
            out = []
            for it in resp.json():
                if not isinstance(it, dict):
                    continue
                name = it.get("name") or it.get("value")
                if not name:
                    continue
                out.append({
                    "name": name,
                    "category": _DANBOORU_CATEGORY.get(it.get("category"), "general"),
                    "post_count": it.get("post_count"),
                })
            return out[:limit]

        if source in ("gelbooru", "safebooru", "rule34"):
            base = {
                "gelbooru": "https://gelbooru.com/index.php",
                "safebooru": "https://safebooru.org/index.php",
                "rule34": "https://api.rule34.xxx/index.php",
            }[source]
            extra = {}
            if source in SOURCES_REQUIRING_CREDENTIALS:
                creds = _credentials(source)
                if not creds.get("api_key") or not creds.get("user_id"):
                    return []  # autocomplete degrades silently - the search itself will explain why
                extra = creds
            resp = session.get(
                base,
                params={"page": "autocomplete2", "term": query, "type": "tag_query", "limit": limit, **extra},
                timeout=AUTOCOMPLETE_TIMEOUT,
            )
            resp.raise_for_status()
            out = []
            for it in resp.json():
                if not isinstance(it, dict):
                    continue
                name = it.get("value") or it.get("label")
                if not name:
                    continue
                count = it.get("post_count")
                if count is None:
                    m = re.search(r"\(([\d,]+)\)", it.get("label", ""))
                    if m:
                        try:
                            count = int(m.group(1).replace(",", ""))
                        except ValueError:
                            count = None
                out.append({"name": name, "category": "general", "post_count": count})
            return out[:limit]

        # konachan/yandere (moebooru) don't expose a comparable tag
        # autocomplete endpoint - fall through to the empty result below
        # rather than guessing at one.
    except (requests.exceptions.RequestException, ValueError):
        return []

    return []
