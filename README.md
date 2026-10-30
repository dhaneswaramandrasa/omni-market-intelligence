# Omni Market Intelligence

**Who ordered what, where, today?** App markets read from telco traffic: app users → drivers vs customers → service → orders, for every brand and every area at once.

[**▶ Live 3D map**](https://dhaneswaramandrasa.github.io/omni-market-intelligence/) · [Notebook](notebooks/omni_market_intelligence.ipynb) · [Portfolio](https://dhaneswara-mandrasa.vercel.app)

![Gas-pol vs Ne-beng food orders in BSD](docs/demo-gaspol-vs-nebeng.png)

> **Simulated data, real method, anonymised brands.** I built the production version at Eureka AI (2021–2024)
> for super-app, ride-hailing and e-commerce clients in Jakarta, Bangkok, Kuala Lumpur and Singapore. It grew
> from 5% to 70% of company revenue. Clients, data and code stay confidential; this repo is a clean reimplementation.

## The question

*How many people in BSD ordered food on Gas-pol today, compared with Ne-beng?*
Neither app shares its numbers. A mobile network sees every app's traffic, so it can estimate this for all of them.

| Brand (anonymised) | Kind | Services |
|---|---|---|
| Ride-hailing **Gas-pol**, Ride-hailing **Ne-beng** | super-app | Ride · Car · Food · Mart |
| E-commerce **ShoppingYuk**, E-commerce **Lapakku** | marketplace | Fashion · Electronics · Groceries · Home |

## Step zero: from hostnames to sessions

The network never sees "a Food session". It sees a stream of **hostnames** per subscriber: an app's API, image
CDNs with random names, telemetry, and push, ads, maps and payment hosts that every app shares.

1. **Rules first:** a dictionary of known hosts per app. Simple, but most traffic goes to hosts nobody listed.
2. **Then embedding:** Word2Vec over each subscriber's host stream (a new "sentence" after 10 minutes of silence).
   Hosts that **pop up together** land together, so each app's unknown hosts sit next to its known ones and the
   dictionary grows itself. Shared hosts sit between apps and attach to whichever session they fall inside.
3. **Sessionize:** a run of one app's events with gaps under 5 minutes is one session.

| | Host dictionary | Embedding |
|---|---|---|
| App events attributed | 10% | 65% |
| Real sessions found | 57% | 99% |
| …recovered as exactly one session | 53% | 89% |
| Unknown hosts attributed correctly | n/a | 32 / 32 |

![Sessions from host events](docs/demo-sessions.png)

## The funnel

```
telco DAU (30M, Jabodetabek)
 └─ app users            host lookup on traffic                       Gas-pol ≈ 5.3M · Ne-beng ≈ 4.1M
     ├─ drivers          classifier: partner endpoint, cells, hours    ≈ 7% of ride-hailing app users
     └─ customers
         ├─ service      classifier: host hint, bytes, duration, movement afterwards
         └─ ordered?     classifier: payment endpoint, heavier sessions, a trip right after
                                                                       ≈ 12% of sessions · ≈ 24% of customers
```

Models are gradient-boosted trees trained on a 10% labelled **panel** and scored on the rest:

| Step | Held-out result |
|---|---|
| Driver vs customer | precision 0.94 · recall 0.89 |
| Service | accuracy 0.89 (ride-hailing) · 0.86 (e-commerce) |
| Ordered? | AUC 0.96 · precision 0.78 · recall 0.80 |

The order threshold is set so the predicted rate matches the panel's rate, which keeps **area totals unbiased**
even though single sessions are uncertain. Clients buy the totals.

![Food orders by area](docs/food-by-area.png)

## The demo

`web/index.html` (MapLibre + deck.gl, no build step) reads `web/data/omni.json`, written by the notebook:

1. **The city breathes**: 2 km hexes, height = people on the network, over 24 hours
2. **Find the app users**: per brand, by hour
3. **From hostnames to sessions**: the host embedding and one subscriber's rebuilt sessions
4. **Drivers vs customers**: animated driver trips for both ride-hailing brands
5. **Which service**: Ride / Car / Food / Mart, Fashion / Electronics / …
6. **Did they order**: users vs users who ordered, plus model quality
7. **Gas-pol vs Ne-beng**: Ne-beng's share of food orders per hex, for any area (BSD by default)

![The city breathes](docs/demo-city.png)

## Run it

```bash
pip install -r requirements.txt
cd notebooks && jupyter nbconvert --to notebook --execute --inplace omni_market_intelligence.ipynb
cd ../web && python -m http.server 8000   # open http://localhost:8000
```

## Layout

```
omni/        sim.py (hex grid, people, sessions, network features) · sessionize.py (host embedding, sessions) · model.py (driver, service, order models; export)
notebooks/   omni_market_intelligence.ipynb, the full walk-through with outputs
web/         interactive 3D map + data/omni.json
docs/        figures
```

## In production

PySpark over clickstream and cell records · host-to-app dictionaries maintained per market · MLflow for model runs ·
daily refresh per city · outputs: market size, penetration and share by area, brand and service.

---
Dhaneswara Mandrasa · MIT License
