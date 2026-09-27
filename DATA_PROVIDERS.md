# DATA_PROVIDERS.md — survivorship-free equity data: evaluation

_Researched 2026-09-27. Prices and terms change; confirm before buying. **Nothing has been purchased**
— any subscription needs the owner's explicit approval._

## Requirements (from DATA_GAPS.md G1–G3)

1. Delisted US stocks with daily prices back to ≤ 2004 (PEAD) and ideally ≤ 1998.
2. Point-in-time index membership (S&P 500 minimum; S&P 100, Russell 1000/2000 useful).
3. Permanent security id with ticker/name history.
4. Corporate actions (splits, dividends) and lifecycle events (delistings with reasons, M&A,
   bankruptcy).
5. Works on this Mac and later in the cloud from Python, without re-architecting.
6. Desirable: delisting returns, PIT fundamentals, CIK cross-reference.

## Comparison

| | Tiingo (current, free) | Norgate Data Platinum | Sharadar (Nasdaq Data Link) | CRSP |
|---|---|---|---|---|
| Delisted stocks with prices | mostly 2013+ only (see DATA_GAPS G1) | yes, back to 1990 (Diamond: 1950); 25,222 delisted securities 1950 → Sep 2022 | yes; >21,000 active + delisted tickers, prices from 1998 | yes; the reference dataset (>36,000 securities), from 1925 |
| PIT index membership | none | S&P 500 (1957), S&P 100 (1989), S&P 400/600, Russell 1000/2000/3000 (1990), Nasdaq-100, DJIA | S&P 500 additions/removals (1957) only | CRSP indices; S&P 500 membership in CRSP/Compustat products via WRDS |
| Permanent id | permaTicker (undocumented in price API) | `assetid` — stable through symbol/name/exchange changes and delisting | `permaticker` | PERMNO / PERMCO |
| Ticker history | no | yes (via assetid) | yes (ACTIONS `tickerchangefrom/to`) | yes |
| Delisting reason | no | **no** (stated in their FAQ) | yes (ACTIONS: `delisted`, `acquisitionby`, `mergerto`, `bankruptcyliquidation`, `regulatorydelisting`, `voluntarydelisting`, …) | yes (delisting codes) |
| Delisting return | no | **no** | no | **yes** (DLRET) |
| Fundamentals | DOW 30, 3 y (free) | no | yes, PIT by filing date (SF1, 1990+) | via Compustat (separate) |
| Access / OS | REST API | **Windows-only** Norgate Data Updater; Python `norgatedata` package reads its local database (Mac: Windows VM) | REST API / bulk export, any OS, cloud-friendly | WRDS (institutional) or direct CRSP licence |
| Cost | free tier (low request limits) | **US$630 / year** (US$346.50 / 6 months); Diamond US$787.50 / year | not published; quote after login (non-professional vs professional licences, monthly or annual) | institutional pricing; not realistic without a university/WRDS affiliation |

Sources: [Norgate packages & prices](https://norgatedata.com/stockmarketpackages.php),
[Norgate content tables](https://norgatedata.com/data-content-tables.php),
[Norgate FAQ](https://norgatedata.com/data-package-faq.php),
[Norgate Data Updater](https://norgatedata.com/ndu-overview.php),
[norgatedata on PyPI](https://pypi.org/project/norgatedata/),
[Sharadar bundle](https://data.nasdaq.com/databases/SFA), [Sharadar prices](https://sharadar.com/prices),
[QuantRocket Sharadar summary](https://www.quantrocket.com/pricing/data/sharadar/),
[CRSP US Stock Databases](https://www.crsp.org/research/crsp-us-stock-databases/),
[Tiingo fundamentals docs](https://www.tiingo.com/documentation/fundamentals),
[Tiingo search docs](https://www.tiingo.com/documentation/utilities/search).

## Recommendation

**Primary: Sharadar Core US Equities Bundle, subject to the non-professional quote.** It is the only
option that meets every hard requirement *and* the Mac/cloud requirement: REST access from Python on
any OS, permaticker, ticker history, delisting events with reasons and acquisition counterparties,
PIT S&P 500 membership, and PIT fundamentals for later work. Its 1998 price start covers PEAD
(2004+) and every v1 study. Gap: only S&P 500 membership (S&P 100 = the 100 largest S&P 500 members
by market cap on each date would be an approximation, flagged), and no delisting returns.

**Alternative: Norgate Platinum (US$630/yr)** if the Sharadar quote is high or broader index history
(Russell, S&P 100/400/600) matters more than portability. Best index coverage for the price, but it
needs a Windows VM on this Mac and ties research to a desktop process, which conflicts with a later
cloud deployment. No delisting reasons or returns.

**CRSP** only if a university/WRDS affiliation is available; it is the only source of delisting
returns and the academic reference, but not practical to license individually.

**Not recommended:** building the historical universe from Tiingo — it cannot supply the pre-2013
delisted prices, and it has no membership history.

## Migration contract (so the switch touches only an adapter)

A new vendor is one class implementing `DataProvider` (`data/providers/base.py`):

* `get_security_master()` → `securities`, `security_identifiers`, `security_events` with ids
  namespaced by vendor (`SHARADAR:<permaticker>`, `NORGATE:<assetid>`, `CRSP:<permno>`).
* `get_daily_bars`, `get_corporate_actions`, `get_index_membership` keyed by that id in the
  `symbol` column.
* Universes are declared `stock_pit` (MembershipUniverse over `index_membership`) and bars come from
  every security listed on each date (`SecurityMaster.listed_on`), not from today's ticker list.

The panel's columns are security ids, so features, strategies, backtests, statistics, the catalog
gate and the reports are unchanged. Only display code maps ids back to the ticker valid on each date
(`SecurityMaster.attribute`).
