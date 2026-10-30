"""Writes notebooks/omni_market_intelligence.ipynb. Run it, then execute the notebook."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Omni Market Intelligence: app markets from telco traffic

How many people in **BSD** ordered food on **Gas-pol** today, compared with **Ne-beng**?
The apps won't tell each other. A mobile network can estimate it for every brand at once, from traffic alone.

The funnel this notebook rebuilds:

```
telco DAU ─► app users (host lookup) ─► drivers vs customers ─► service (Ride/Car/Food/Mart…) ─► transacted?
```

> I built the production version at Eureka AI (2021–2024) for super-app, ride-hailing and e-commerce clients
> in Jakarta, Bangkok, Kuala Lumpur and Singapore, and it grew from 5% to 70% of company revenue.
> **Here everything is simulated and every brand is anonymised**; the method is the real one."""),
code("""import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from omni.sim import simulate, BRANDS, SERVICES, SVC_HOURS, bumps
from omni import model as m
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
SCALE = 150          # one simulated person = 150 real subscribers → about 30M daily users in Jabodetabek"""),
md("## 1. Simulate a day on the network\n200,000 people on a 2 km hex grid over Jabodetabek. Each one sleeps somewhere, 60% commute to work, and some use one or more of four apps."),
code("""cells, people, sessions = simulate()
print(f'{len(cells):,} hexes · {len(people):,} people · {len(sessions):,} app sessions')
sessions.head()"""),
md("Each session is what the network sees: hour, cell, a (noisy) **service host** hint, whether a **payment** or **driver-partner** endpoint was hit, bytes, duration, and how far the phone moved in the next hour."),
md("""## 1b. From host events to app sessions
The table above already has one row per session, but the network never sees sessions. It sees **hostnames**:
an app's API, image CDNs with random names, telemetry, plus shared hosts (push, ads, maps, payments) that every app uses.

- **Version 1, rules:** a dictionary of known hosts per app. New or obfuscated hosts are missed and sessions fall apart.
- **Version 2, embedding:** hosts that **pop up together** belong together. Word2Vec over each subscriber's host stream puts an app's unknown hosts next to its known ones, so the dictionary grows itself. Shared hosts sit between apps and attach to whichever session they fall inside."""),
code("""from omni import sessionize as z
catalog = z.host_catalog(BRANDS)
events, ev_sessions = z.raw_events(sessions, catalog, n_people=8000)
print(f'{len(events):,} host events from {events.person.nunique():,} people · {events.host.nunique()} distinct hosts')
events.head(8)"""),
code("""rules, emb, w2v_hosts, sims = z.attribute_hosts(events, catalog)
hidden = {h: b for b, c in catalog.items() for h in c['hidden']}
print(f"hidden hosts attributed correctly: {sum(emb.get(h) == b for h, b in hidden.items())}/{len(hidden)}")
print(f"shared hosts wrongly given to one app: {sum(h in emb for h in z.SHARED)}/{len(z.SHARED)}")
sess_eval = {}
for name, table in [('rules', rules), ('embedding', emb)]:
    pred, app = z.sessionize(events, table)
    sess_eval[name] = z.evaluate(events, pred, app)
pd.DataFrame(sess_eval).round(3)"""),
md("Rules alone recover about half of the real sessions. The embedding finds almost all of them, and most come back whole."),
code("""from sklearn.decomposition import PCA
hosts = [h for h in w2v_hosts.wv.index_to_key]
from sklearn.manifold import TSNE
xy = TSNE(2, perplexity=12, random_state=7, init='pca').fit_transform(w2v_hosts.wv[hosts])
kind = ['shared' if h in z.SHARED else 'browse' if h in z.BROWSE else emb.get(h, '?') for h in hosts]
known = {h for c in catalog.values() for h in c['known']}
fig, ax = plt.subplots(figsize=(6, 4.2))
for k in ['browse', 'shared'] + list(BRANDS):
    sel = np.array(kind) == k
    c = np.array(BRANDS[k]['color']) / 255 if k in BRANDS else ('#bbb' if k == 'browse' else '#555')
    ax.scatter(*xy[sel].T, s=18, color=c, label=BRANDS[k]['name'] if k in BRANDS else k)
for i, h in enumerate(hosts):
    if h in known: ax.annotate(h.split('.')[0] if h.startswith('api') else h.split('.')[0], xy[i], fontsize=6.5)
ax.set_axis_off(); ax.legend(frameon=False, fontsize=7, loc='best'); ax.set_title('Host embedding: unknown hosts land next to their app')
plt.tight_layout(); plt.savefig('../docs/host-embedding.png'); plt.show()"""),
md("## 2. Labelled panel\nModels are trained on a 10% **panel** of people whose app receipts are known (in production: a consented research panel). Everything is then scored on the other 90%."),
md("## 3. Drivers vs customers\nPer person and brand: sessions, distinct cells, active hours, share of partner-endpoint traffic, distance moved."),
code("""users = m.user_features(sessions)
panel = m.split_panel(users)
users['driver_pred'], driver_rep = m.fit_driver(users, panel)
s = sessions.merge(users[['person', 'brand', 'driver_pred']], on=['person', 'brand'])
ride_users = users[users.brand.isin(['gaspol', 'nebeng'])]
print(f"driver precision {driver_rep['precision']:.2f} · recall {driver_rep['recall']:.2f}")
print(f"drivers as share of ride-hailing app users: {ride_users.driver_pred.mean():.1%}")"""),
md("## 4. Which service?\nA session's host hint is wrong or ambiguous 22% of the time, so the model also uses bytes, duration and movement afterwards (a Ride moves you; Food doesn't)."),
code("""cust = s[~s.driver_pred].reset_index(drop=True)
cpanel = m.split_panel(cust)
cust['service_pred'], svc_rep = m.fit_service(cust, cpanel)
for k, r in svc_rep.items():
    print(f"{k:13s} accuracy {r['accuracy']:.2f} · " + ' · '.join(f"{c} F1 {r[c]['f1-score']:.2f}" for c in r if c not in ('accuracy', 'macro avg', 'weighted avg')))"""),
md("""## 4b. Ride or Car, when both use one host
Two-wheel **Ride** and four-wheel **Car** usually call the same host, so the model above can only say **Transport**.
The way in: **Ne-beng** calls a separate host for its car product. That gives free labels on Ne-beng's own traffic.
A supervised model learns what a car trip looks like on the network (longer, slower in traffic, more toll-road cells,
more stop-and-go) and then labels **Gas-pol**, whose host says nothing."""),
code("""cust['service_pred'], rc_rep = m.split_ride_car(cust, teacher='nebeng', student='gaspol')
print(f"Gas-pol ordered trips labelled correctly: {rc_rep['accuracy_ordered']:.0%} (all transport sessions: {rc_rep['accuracy_all']:.0%})")
print(f"Gas-pol car share of trips: true {rc_rep['car_share_true']:.0%} · estimated {rc_rep['car_share_pred']:.0%}")
print(f"teacher label noise (car sessions that missed their own host): {rc_rep['teacher_label_noise']:.0%}")"""),
md("Trips are easy; sessions that end without a trip are mostly guesswork, which is why the session-level number is lower. The car share comes out a little low because some Ne-beng car sessions miss their own host and teach the model they were rides. In production that bias is measured on a panel and corrected."),
md("## 5. Did the session end in an order?\nThe threshold is set so the predicted transaction rate matches the panel's rate. That keeps the market totals unbiased even though single sessions are uncertain."),
code("""cust['txn_pred'], txn_rep = m.fit_txn(cust, cpanel)
print(f"AUC {txn_rep['auc']:.3f} · precision {txn_rep['precision']:.2f} · recall {txn_rep['recall']:.2f}")
print(f"sessions with an order: true {txn_rep['true_rate']:.1%} · predicted {txn_rep['pred_rate']:.1%}")
dau = cust.groupby(['person', 'brand']).txn_pred.any()
print(f"customers who ordered at least once today: {dau.mean():.1%}")"""),
md("## 6. The funnel"),
code("""rows = []
for b, cfg in BRANDS.items():
    u = users[users.brand == b]; c = cust[cust.brand == b]
    rows.append(dict(brand=cfg['name'], app_dau=len(u) * SCALE, drivers=u.driver_pred.sum() * SCALE,
                     customers=(~u.driver_pred).sum() * SCALE,
                     ordering_customers=c[c.txn_pred].person.nunique() * SCALE))
funnel = pd.DataFrame(rows).set_index('brand')
print(f'telco DAU (Jabodetabek, simulated): {len(people) * SCALE / 1e6:.0f}M')
(funnel / 1e6).round(2).rename(columns=lambda x: x + ' (M)')"""),
md("## 7. The question clients pay for\nFood: daily users who ordered, by area, Gas-pol vs Ne-beng."),
code("""cust['area'] = cells.area.values[cust.cell]
food = cust[(cust.service_pred == 'Food') & cust.txn_pred & cust.brand.isin(['gaspol', 'nebeng'])]
tab = food.groupby(['area', 'brand']).person.nunique().unstack().mul(SCALE)
tab['lime_share'] = tab.nebeng / (tab.gaspol + tab.nebeng)
tab = tab.sort_values('lime_share', ascending=False)
display((tab[['gaspol', 'nebeng']] / 1e3).round(0).astype(int).rename(columns={'gaspol': 'Gas-pol (K)', 'nebeng': 'Ne-beng (K)'}).join(tab.lime_share.map('{:.0%}'.format)))

fig, ax = plt.subplots(figsize=(7, 3.6))
y = np.arange(len(tab))
ax.barh(y - .2, tab.gaspol / 1e3, .4, color=np.array(BRANDS['gaspol']['color']) / 255, label='Gas-pol')
ax.barh(y + .2, tab.nebeng / 1e3, .4, color=np.array(BRANDS['nebeng']['color']) / 255, label='Ne-beng')
ax.set_yticks(y, tab.index); ax.invert_yaxis(); ax.set_xlabel('daily users who ordered food (K)')
ax.legend(frameon=False); ax.set_title('Food orders by area: Ne-beng leads in BSD & Tangerang Selatan')
plt.tight_layout(); plt.savefig('../docs/food-by-area.png'); plt.show()"""),
md("## 8. Export for the 3D map"),
code("""s2 = s.merge(cust[['person', 'brand', 'hour', 'cell', 'service_pred', 'txn_pred']].drop_duplicates(['person', 'brand', 'hour', 'cell']),
              on=['person', 'brand', 'hour', 'cell'], how='left')
s2['service_pred'] = s2.service_pred.fillna('Driver'); s2['txn_pred'] = s2.txn_pred.fillna(False).astype(bool)
brands = {b: dict(name=c['name'], kind=c['kind'], color=c['color']) for b, c in BRANDS.items()}
services = {b: list(SERVICES[c['kind']]) for b, c in BRANDS.items()}
svc_hours = {v: np.round(bumps(h), 4).tolist() for v, h in SVC_HOURS.items()}
metrics = dict(driver=driver_rep, txn=txn_rep, ride_car=rc_rep, service={k: round(r['accuracy'], 3) for k, r in svc_rep.items()},
               txn_dau=float(dau.mean()))
trips = {b: m.driver_trips(cells, s2, b, n=220) for b in ('gaspol', 'nebeng')}
# sessionization demo: host map + one person's morning, for the web
from sklearn.manifold import TSNE
hx = TSNE(2, perplexity=12, random_state=7, init='pca').fit_transform(w2v_hosts.wv[hosts]); hx = (hx - hx.min(0)) / (hx.max(0) - hx.min(0))
host_map = [[round(float(a), 3), round(float(b), 3), k, int(h in known), h] for (a, b), k, h in zip(hx, kind, hosts)]
pred, app = z.sessionize(events, emb)
events['pred'] = pred; events['app'] = app
g = events[events.true_session >= 0].groupby('person')
span = g.t.max() - g.t.min()
busy = ((g.brand.nunique() * 100 + g.true_session.nunique() * 10) * (span < 4 * 3600)).idxmax()   # several apps and sessions within a few hours
tl = events[events.person == busy]
app_t = tl[tl.true_session >= 0].t
tl = tl[(tl.t >= app_t.min() - 1200) & (tl.t <= app_t.max() + 1200)]
timeline = [[int(t), h, a or ('shared' if h in z.SHARED else 'browse'), int(p)] for t, h, a, p in zip(tl.t, tl.host, tl.app, tl.pred)]
metrics['sessions'] = sess_eval
m.export_web('../web/data/omni.json', cells, people, s2, brands, services, svc_hours, metrics, SCALE, trips,
             extra=dict(host_map=host_map, timeline=timeline))
import os; print(f"{os.path.getsize('../web/data/omni.json') / 1e6:.1f} MB")"""),
md("""## Takeaways
- Four signals carry most of the weight: the host hint, payment endpoints, bytes/duration, and movement after the session.
- Single sessions are uncertain (precision around 0.8), but calibrated totals per area are what clients buy, and those are stable.
- In production: PySpark over clickstream and cell records, models tracked in MLflow, refreshed daily per city."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbf.write(nb, 'notebooks/omni_market_intelligence.ipynb')
