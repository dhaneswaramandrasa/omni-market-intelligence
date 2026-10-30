"""The four-step funnel: app user → driver or customer → service → transacted.

Step 1 (which app) is a host lookup and is treated as given. Steps 2-4 are
gradient-boosted classifiers trained on a labelled panel (people who agreed to
share their app receipts) and applied to everyone else.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score

SESSION_X = ["hour", "host_pay", "host_partner", "kb", "duration_s", "move_km_60m"]


def user_features(sessions):
    """Daily features per person × brand, for the driver-vs-customer model."""
    g = sessions.groupby(["person", "brand"])
    f = pd.DataFrame(dict(
        sessions=g.size(), cells=g.cell.nunique(), hours=g.hour.nunique(),
        partner_share=g.host_partner.mean(), move_km=g.move_km_60m.sum(), kb=g.kb.sum(),
    ))
    f["driver"] = g.driver.first()
    return f.reset_index()


def _session_matrix(s):
    X = s[SESSION_X].astype(float).copy()
    for v in sorted(s.host_service.unique()):
        X[f"host_{v}"] = (s.host_service == v).astype(float)
    return X


def split_panel(df, frac=.1, seed=7):
    rng = np.random.default_rng(seed)
    people = df.person.unique()
    panel = set(rng.choice(people, int(len(people) * frac), replace=False))
    return df.person.isin(panel).values


def fit_driver(users, panel):
    X = users[["sessions", "cells", "hours", "partner_share", "move_km", "kb"]]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=7).fit(X[panel], users.driver[panel])
    pred = clf.predict(X)
    test = ~panel
    rep = dict(precision=precision_score(users.driver[test], pred[test]), recall=recall_score(users.driver[test], pred[test]))
    return pred, rep


def fit_service(cust, panel):
    X = _session_matrix(cust)
    out = np.empty(len(cust), object)
    reports = {}
    for kind, idx in cust.groupby(cust.brand.isin(["green", "lime"])).groups.items():
        idx = np.asarray([cust.index.get_loc(i) for i in idx])
        p = panel[idx]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=7).fit(X.iloc[idx][p], cust.service.iloc[idx][p])
        pr = clf.predict(X.iloc[idx])
        out[idx] = pr
        reports["ride-hailing" if kind else "e-commerce"] = classification_report(
            cust.service.iloc[idx][~p], pr[~p], output_dict=True, zero_division=0)
    return out, reports


def fit_txn(cust, panel):
    X = _session_matrix(cust)
    X["svc_pred"] = pd.factorize(cust.service_pred)[0]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=7).fit(X[panel], cust.txn[panel])
    prob = clf.predict_proba(X)[:, 1]
    # pick the threshold where predicted transactions match the panel's rate: unbiased totals
    thr = np.quantile(prob[panel], 1 - cust.txn[panel].mean())
    pred = prob >= thr
    t = ~panel
    rep = dict(auc=roc_auc_score(cust.txn[t], prob[t]), precision=precision_score(cust.txn[t], pred[t]),
               recall=recall_score(cust.txn[t], pred[t]), true_rate=cust.txn[t].mean(), pred_rate=pred[t].mean())
    return pred, rep


def export_web(path, cells, people, sessions, brands, services, svc_hours, metrics, scale, trips):
    """Compact JSON for the map. Counts are simulated people; the page multiplies by `scale`."""
    n = len(cells)
    hours = np.arange(24)
    # everyone on the network, by cell and hour (home at night, work by day for workers)
    act = np.array([.25, .15, .1, .08, .1, .25, .55, .8, .9, .9, .9, .9, .95, .9, .9, .9, .9, .95, 1, 1, 1, .95, .8, .5])
    day = (hours >= 8) & (hours <= 17)
    allh = np.zeros((n, 24))
    for h in hours:
        loc = np.where(people.worker & day[h], people.work, people.home)
        allh[:, h] = np.bincount(loc, minlength=n) * act[h]
    out = dict(meta=dict(scale=scale, brands=brands, services=services, svc_hours=svc_hours, metrics=metrics),
               cells=[[round(a, 4), round(b, 4), c] for a, b, c in zip(cells.lon, cells.lat, cells.area)],
               all=np.round(allh).astype(int).ravel().tolist(), brand={}, trips=trips)
    for b in brands:
        s = sessions[sessions.brand == b]
        uh = s.drop_duplicates(["person", "hour"])
        bh = np.zeros((n, 24), int)
        np.add.at(bh, (uh.cell.values, uh.hour.values), 1)
        d = s[s.driver_pred]
        drv = np.bincount(d.drop_duplicates("person").cell, minlength=n)
        svc = {}
        c = s[~s.driver_pred]
        for v in services[b]:
            cv = c[c.service_pred == v]
            act_u = cv.drop_duplicates("person")
            txn_u = cv[cv.txn_pred].drop_duplicates("person")
            svc[v] = [np.bincount(act_u.cell, minlength=n).tolist(), np.bincount(txn_u.cell, minlength=n).tolist()]
        out["brand"][b] = dict(hour=bh.ravel().tolist(), drivers=drv.tolist(), svc=svc,
                               app=int(s.person.nunique()), n_drivers=int(d.person.nunique()),
                               ordering=int(c[c.txn_pred].person.nunique()))
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))


def driver_trips(cells, sessions, brand, n=250, seed=7):
    """Synthetic trip paths for a sample of predicted drivers: [[lon, lat, minute], ...] per driver."""
    rng = np.random.default_rng(seed)
    ds = sessions[(sessions.brand == brand) & sessions.driver_pred]
    first = ds.sort_values("hour").drop_duplicates("person").set_index("person").cell   # where each driver starts the day
    xy = cells[["lon", "lat"]].values
    paths = []
    for p in rng.choice(first.index.values, min(n, len(first)), replace=False):
        t = rng.uniform(300, 420)
        here = xy[first[p]] + rng.normal(0, .004, 2)
        path = [[*np.round(here, 4), int(t)]]
        while t < 1380:
            dest = here + rng.normal(0, .03, 2)
            km = np.hypot(*(dest - here)) * 111
            t += km / 22 * 60                                   # ~22 km/h in traffic
            mid = (here + dest) / 2 + rng.normal(0, .006, 2)
            path += [[*np.round(mid, 4), int(t - km / 44 * 60)], [*np.round(dest, 4), int(t)]]
            here = dest
            t += rng.uniform(5, 40)                             # waiting for the next order
            path.append([*np.round(here, 4), int(t)])
        paths.append(path)
    return paths
