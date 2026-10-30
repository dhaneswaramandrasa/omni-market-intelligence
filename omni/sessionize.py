"""From raw host events to app sessions.

The network does not see "a Food session". It sees a stream of hostnames per
subscriber: the app's API, image CDNs with random-looking names, telemetry,
maps, ads, push services shared by every app, and plain web browsing.

Version 1 (rules): a dictionary of known hosts per app. Simple, but new and
obfuscated hosts go unattributed and sessions fall apart.

Version 2 (embedding): hosts that pop up together belong together. Word2Vec on
each subscriber's host stream puts an app's unknown hosts next to its known
ones, so the dictionary grows itself. Shared hosts sit between apps and are
attached to whichever app session they fall inside.
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

SHARED = [f"{h}.example" for h in ("cdn-global", "push-svc", "analytics-a", "analytics-b", "ads-net", "maps-tiles",
                                   "fonts-cdn", "crash-report", "auth-sso", "payments-gw")]
BROWSE = [f"site-{k:02d}.example" for k in range(40)]


def host_catalog(brands, seed=7):
    """Per brand: a few known hosts (in the starting dictionary) and more hidden ones (random-looking names)."""
    rng = np.random.default_rng(seed)
    cat = {}
    for b in brands:
        known = [f"api.{b}.example", f"{b}-app.example"]
        hidden = ["".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz0123456789"), 8)) + ".cdn.example" for _ in range(8)]
        cat[b] = dict(known=known, hidden=hidden)
    return cat


def raw_events(sessions, catalog, n_people=15000, seed=7):
    """Expand app sessions of a sample of people into timestamped host events, plus background browsing."""
    rng = np.random.default_rng(seed)
    people = rng.choice(sessions.person.unique(), n_people, replace=False)
    s = sessions[sessions.person.isin(people)].reset_index(drop=True).copy()
    s["t0"] = s.hour * 3600 + rng.integers(0, 3300, len(s))
    s["true_session"] = np.arange(len(s))
    rows = []
    for r in s.itertuples():
        n = rng.integers(3, 15)
        c = catalog[r.brand]
        kind = rng.choice(3, n, p=[.1, .55, .35])                    # known / hidden / shared
        hosts = np.where(kind == 0, rng.choice(c["known"], n), np.where(kind == 1, rng.choice(c["hidden"], n), rng.choice(SHARED, n)))
        ts = r.t0 + np.sort(rng.integers(0, int(rng.uniform(60, 900)), n))
        rows.append(pd.DataFrame(dict(person=r.person, t=ts, host=hosts, brand=r.brand, true_session=r.true_session)))
    ev = pd.concat(rows, ignore_index=True)
    # background browsing between app sessions: no app, no session
    nb = len(ev) // 2
    bg = pd.DataFrame(dict(person=rng.choice(people, nb), t=rng.integers(0, 86400, nb),
                           host=rng.choice(BROWSE + SHARED[:4], nb), brand="none", true_session=-1))
    ev = pd.concat([ev, bg], ignore_index=True).sort_values(["person", "t"]).reset_index(drop=True)
    return ev, s


def host_sentences(ev, gap=600):
    """One 'sentence' per burst of activity: a new one starts after `gap` seconds of silence."""
    new = (ev.person != ev.person.shift()) | (ev.t.diff() > gap)
    burst = new.cumsum()
    return ev.groupby(burst).host.apply(list).tolist()


def attribute_hosts(ev, catalog, min_sim=.45, margin=.08, seed=7):
    """Rules vs embedding: which app does each host belong to?"""
    rules = {h: b for b, c in catalog.items() for h in c["known"]}
    w2v = Word2Vec(host_sentences(ev), vector_size=32, window=5, min_count=3, sg=1, epochs=8, workers=1, seed=seed)
    cent = {b: np.mean([w2v.wv[h] for h in c["known"] if h in w2v.wv], 0) for b, c in catalog.items()}
    names = list(cent)
    C = np.array([cent[b] / np.linalg.norm(cent[b]) for b in names])
    emb = dict(rules)
    sims = {}
    for h in w2v.wv.index_to_key:
        if h in rules:
            continue
        v = w2v.wv[h] / np.linalg.norm(w2v.wv[h])
        s = C @ v
        o = np.argsort(-s)
        sims[h] = s
        if s[o[0]] >= min_sim and s[o[0]] - s[o[1]] >= margin:
            emb[h] = names[o[0]]
    return rules, emb, w2v, sims


def sessionize(ev, host_app, gap=300, attach=120):
    """Group a person's attributed events into app sessions.

    An event joins the running session of its app if it comes within `gap` seconds.
    Unattributed events (shared hosts) join a session if one is open within `attach` seconds.
    """
    app = ev.host.map(host_app).fillna("")
    out = np.full(len(ev), -1)
    sid = -1
    open_ = {}                                               # app -> (session id, last t)
    last_person = None
    for i, (p, t, a) in enumerate(zip(ev.person.values, ev.t.values, app.values)):
        if p != last_person:
            open_ = {}
            last_person = p
        if a:
            cur = open_.get(a)
            if cur is None or t - cur[1] > gap:
                sid += 1
                cur = (sid, t)
            open_[a] = (cur[0], t)
            out[i] = cur[0]
        else:
            best = max(((v[1], v[0]) for v in open_.values() if t - v[1] <= attach), default=None)
            if best is not None:
                out[i] = best[1]
    return out, app.values


def evaluate(ev, pred_session, pred_app):
    """Event coverage, and how many true sessions come back as exactly one predicted session of the right app."""
    app_ev = ev.true_session >= 0
    cov = (pred_app[app_ev.values] != "").mean()
    d = pd.DataFrame(dict(true=ev.true_session, pred=pred_session, brand=ev.brand, app=pred_app))[app_ev.values]
    per_true = d[d.pred >= 0].groupby("true").pred.nunique()
    n_true = ev.true_session[app_ev].nunique()
    exact = (per_true == 1).sum() / n_true
    found = len(per_true) / n_true
    wrong = (d.app != "") & (d.app != d.brand)
    return dict(event_coverage=float(cov), sessions_found=float(found), sessions_exact=float(exact),
                fragments_per_session=float(d[d.pred >= 0].groupby("true").pred.nunique().mean()),
                wrong_app_events=float(wrong.mean()))
