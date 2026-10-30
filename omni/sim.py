"""Simulated Jabodetabek telco network: places, people, app sessions.

Everything here is invented. The structure follows the production pipeline:
the network sees a subscriber's cell, the hosts their phone talks to, bytes,
duration and how far they move afterwards. It never sees the app's own data.

Brands are anonymised: two ride-hailing super-apps (Gas-pol, Ne-beng) and two
marketplaces (ShoppingYuk, Lapakku).
"""
import numpy as np
import pandas as pd

BRANDS = {
    "gaspol":  dict(name="Gas-pol",  kind="ride", dau=.17, driver=.07, color=(46, 230, 140)),
    "nebeng":   dict(name="Ne-beng",   kind="ride", dau=.13, driver=.08, color=(255, 214, 10)),
    "shoppingyuk": dict(name="ShoppingYuk",   kind="shop", dau=.15, driver=0,   color=(255, 92, 57)),
    "lapakku":   dict(name="Lapakku",     kind="shop", dau=.10, driver=0,   color=(56, 189, 248)),
}
SERVICES = {
    "ride": dict(Ride=.35, Car=.15, Food=.35, Mart=.15),
    "shop": dict(Fashion=.35, Electronics=.2, Groceries=.25, Home=.2),
}
# chance one session ends in an order (10-15% of sessions on average)
TXN_P = dict(Ride=.19, Car=.17, Food=.14, Mart=.11, Fashion=.09, Electronics=.06, Groceries=.13, Home=.08)
# when each service is used (peak, width, weight)
SVC_HOURS = dict(
    Ride=[(7, 1.2, 1.3), (17.5, 1.5, 1.2)], Car=[(8, 1.5, .8), (18, 2, 1), (22, 1.5, .5)],
    Food=[(12, 1.2, 1.3), (19, 1.5, 1.2), (23, 1.5, .4)], Mart=[(10, 2, .7), (19, 2, 1)],
    Fashion=[(12, 2, .6), (21, 2, 1.2)], Electronics=[(13, 3, .7), (21, 2, 1)],
    Groceries=[(9, 2, 1), (19, 2, .8)], Home=[(11, 3, .7), (20, 2, 1)],
)

# Area centres (lon, lat). A hex belongs to the nearest one.
AREAS = {
    "Jakarta Selatan": (106.810, -6.270), "Jakarta Pusat": (106.835, -6.180), "Jakarta Barat": (106.740, -6.165),
    "Jakarta Timur": (106.895, -6.235), "Jakarta Utara": (106.880, -6.130), "BSD & Tangerang Selatan": (106.680, -6.305),
    "Tangerang": (106.620, -6.180), "Bekasi": (107.000, -6.250), "Cikarang": (107.150, -6.290),
    "Depok": (106.815, -6.400), "Bogor": (106.800, -6.580),
}
# where people sleep (lon, lat, spread, weight) and where they spend the day
HOME = [(106.81, -6.27, .05, 3), (106.74, -6.17, .04, 2.2), (106.90, -6.23, .05, 2.6), (106.88, -6.13, .03, 1.4),
        (106.83, -6.18, .025, 1), (106.67, -6.30, .045, 1.8), (106.61, -6.19, .04, 1.6), (107.00, -6.26, .05, 2.4),
        (107.14, -6.29, .04, 1), (106.82, -6.40, .045, 1.9), (106.80, -6.57, .05, 1.4)]
WORK = [(106.822, -6.212, .018, 4), (106.830, -6.230, .015, 2), (106.79, -6.24, .025, 1.5), (106.655, -6.300, .02, 1.6),
        (106.910, -6.190, .02, 1.3), (107.150, -6.300, .03, 1.4), (106.740, -6.150, .03, 1.2), (106.630, -6.180, .03, 1),
        (106.83, -6.40, .03, .8), (106.80, -6.58, .03, .6), (106.99, -6.24, .03, 1)]
# brand strength by area (multiplies daily-active chance); makes the market map uneven
BRAND_AREA = {
    "gaspol": {"Jakarta Selatan": 1.2, "Jakarta Pusat": 1.2, "Depok": 1.1, "BSD & Tangerang Selatan": .85, "Tangerang": .9},
    "nebeng": {"BSD & Tangerang Selatan": 1.35, "Tangerang": 1.25, "Jakarta Barat": 1.15, "Jakarta Pusat": .9},
    "shoppingyuk": {"Bekasi": 1.2, "Cikarang": 1.2, "Bogor": 1.1},
    "lapakku": {"Jakarta Selatan": 1.15, "BSD & Tangerang Selatan": 1.2},
}
# Ne-beng's food business is unusually strong in BSD: the story the demo tells
FOOD_BOOST = {("nebeng", "BSD & Tangerang Selatan"): 1.5, ("gaspol", "BSD & Tangerang Selatan"): .85}

BBOX = (106.55, -6.66, 107.22, -6.09)


def bumps(spec, hours=np.arange(24)):
    v = np.full(len(hours), .03)
    for peak, width, w in spec:
        d = np.minimum(abs(hours - peak), 24 - abs(hours - peak))
        v = v + w * np.exp(-d ** 2 / (2 * width ** 2))
    return v / v.sum()


def hex_grid(step: float = .018):
    """Pointy hex centres over the bounding box, minus the sea north of the coast."""
    x0, y0, x1, y1 = BBOX
    pts, row, y = [], 0, y0
    while y <= y1:
        x = x0 + (step / 2 if row % 2 else 0)
        while x <= x1:
            coast = -6.105 + .02 * np.sin((x - 106.6) * 9)
            if y < coast:
                pts.append((x, y))
            x += step
        y += step * .866
        row += 1
    cells = pd.DataFrame(pts, columns=["lon", "lat"])
    names = list(AREAS)
    c = np.array(list(AREAS.values()))
    d = (cells.lon.values[:, None] - c[:, 0]) ** 2 + (cells.lat.values[:, None] - c[:, 1]) ** 2
    cells["area"] = [names[i] for i in d.argmin(1)]
    return cells


def _density(cells, spec):
    v = np.zeros(len(cells))
    for lon, lat, s, w in spec:
        v += w * np.exp(-((cells.lon - lon) ** 2 + (cells.lat - lat) ** 2) / (2 * s ** 2))
    return (v + .002) / (v + .002).sum()


def simulate(n_people: int = 200_000, seed: int = 7):
    """Return cells, people and sessions (one row per app session, with truth labels)."""
    rng = np.random.default_rng(seed)
    cells = hex_grid()
    home = rng.choice(len(cells), n_people, p=_density(cells, HOME))
    worker = rng.random(n_people) < .6
    work = np.where(worker, rng.choice(len(cells), n_people, p=_density(cells, WORK)), home)
    people = pd.DataFrame(dict(home=home, work=work, worker=worker))
    people["area"] = cells.area.values[home]

    rows = []
    for b, cfg in BRANDS.items():
        mult = people.area.map(BRAND_AREA[b]).fillna(1).values
        active = rng.random(n_people) < cfg["dau"] * mult
        idx = np.flatnonzero(active)
        driver = rng.random(len(idx)) < cfg["driver"]
        svc_names = list(SERVICES[cfg["kind"]])
        svc_p = np.array(list(SERVICES[cfg["kind"]].values()))
        # customers: 1 + Poisson sessions; drivers: many short partner-app sessions
        part = rng.random(len(idx)) < .4                                  # part-time drivers look like busy customers
        k = np.where(driver, np.where(part, rng.poisson(2, len(idx)) + 1, rng.poisson(6, len(idx)) + 2), 1 + rng.poisson(.8 if cfg["kind"] == "ride" else 1.5, len(idx)))
        pid = np.repeat(idx, k)
        drv = np.repeat(driver, k)
        m = len(pid)
        svc = rng.choice(len(svc_names), m, p=svc_p)
        hour = np.empty(m, int)
        for j, s in enumerate(svc_names):
            sel = svc == j
            hour[sel] = rng.choice(24, sel.sum(), p=bumps(SVC_HOURS[s]))
        hour[drv] = rng.choice(24, drv.sum(), p=bumps([(8, 3, 1), (13, 3, .8), (19, 3, 1)]))
        at_work = people.worker.values[pid] & (hour >= 8) & (hour <= 17)
        cell = np.where(at_work, people.work.values[pid], people.home.values[pid])
        roam = rng.choice(len(cells), drv.sum(), p=_density(cells, WORK + HOME))   # full-time drivers are wherever the orders are
        full = drv & ~np.repeat(part, k)
        cell[full] = roam[full[drv]]
        service = np.array(svc_names)[svc]
        p_txn = np.vectorize(TXN_P.get)(service)
        boost = np.array([FOOD_BOOST.get((b, a), 1) if s == "Food" else 1
                          for a, s in zip(people.area.values[pid], service)])
        txn = (~drv) & (rng.random(m) < p_txn * boost)
        rows.append(pd.DataFrame(dict(person=pid, brand=b, driver=drv, service=np.where(drv, "Driver", service),
                                      hour=hour, cell=cell, txn=txn)))
    sessions = pd.concat(rows, ignore_index=True)
    sessions = add_network_features(sessions, rng)
    return cells, people, sessions


def add_network_features(s, rng):
    """What the network sees for each session. Signals are noisy on purpose."""
    m = len(s)
    ride = s.service.isin(["Ride", "Car"]).values
    s["host_service"] = np.where(rng.random(m) < .78, s.service, rng.choice(s.service.unique(), m))  # SNI hint, 22% wrong/ambiguous
    s["host_pay"] = rng.random(m) < np.where(s.txn, .72, .07)          # payment endpoint seen
    s["host_partner"] = rng.random(m) < np.where(s.driver, .45, .05)    # driver-app endpoint
    base_kb = s.service.map(dict(Ride=300, Car=320, Food=900, Mart=1100, Fashion=2400, Electronics=1800,
                                 Groceries=1300, Home=2000, Driver=250)).values
    s["kb"] = rng.lognormal(np.log(base_kb * np.where(s.txn, 1.6, 1)), .5)
    s["duration_s"] = rng.lognormal(np.log(np.where(s.txn, 340, 150)), .6)
    move = np.where(s.txn & ride, rng.lognormal(np.log(6), .6, m), rng.exponential(.6, m))
    move = np.where(s.driver, rng.lognormal(np.log(4), .7, m), move)
    s["move_km_60m"] = move                                            # distance travelled in the next hour

    # Ride (2 wheels) and Car (4 wheels) usually share one host: the network only sees "Transport".
    # Ne-beng is the exception: its car product calls its own host, which gives free labels.
    car = (s.service == "Car").values
    shared = s.host_service.isin(["Ride", "Car"]).values
    own_car_host = (s.brand == "nebeng").values & car & (rng.random(m) < .9)
    s["host_service"] = np.where(shared, np.where(own_car_host, "Car", "Transport"), s.host_service)
    # trip shape after an order: cars go further, slower in traffic, and use toll roads; bikes weave through
    trip = s.txn.values & ride
    s["trip_km"] = np.where(trip, rng.lognormal(np.log(np.where(car, 10, 5)), .5), 0)
    s["speed_kmh"] = np.where(trip, np.clip(rng.normal(np.where(car, 18, 24), 5), 4, 60), 0)
    s["toll_share"] = np.where(trip, np.clip(rng.beta(np.where(car, 2, .3), np.where(car, 6, 12)), 0, 1), 0)
    s["stop_go"] = np.where(trip, rng.normal(np.where(car, .55, .4), .12), 0)   # share of time below 5 km/h
    return s
