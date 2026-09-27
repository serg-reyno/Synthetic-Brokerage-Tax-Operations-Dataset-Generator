"""
Synthetic Brokerage Tax Operations Dataset Generator
=====================================================

Generates five internally-consistent, fully fictional CSV files that simulate
the data environment of a brokerage/fintech Tax Operations function:

    accounts.csv      - customer/account master + tax documentation status
    transactions.csv  - trade, dividend, interest, and option activity
    cost_basis.csv    - realized gain/loss lots derived from SELL transactions
    tax_forms.csv     - year-end 1099-B/DIV/INT/1042-S/1099-R summaries
    withholding.csv   - backup / NRA / state withholding detail records

No real people, accounts, SSNs/TINs, or securities are used anywhere.

Run:
    python generate_tax_dataset.py

Output:
    ./data/accounts.csv
    ./data/transactions.csv
    ./data/cost_basis.csv
    ./data/tax_forms.csv
    ./data/withholding.csv
"""

import hashlib
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. CONFIGURATION & REPRODUCIBILITY
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "data"

NUM_ACCOUNTS = 750
TAX_YEAR = 2024

YEAR_START = date(TAX_YEAR, 1, 1)
YEAR_END = date(TAX_YEAR, 12, 31)
# Purchases are allowed to originate up to 3 years before the tax year so that
# long-term sales inside the tax year have plausible, earlier purchase
# history even though the reporting/transaction window is a single year.
HISTORY_START = date(TAX_YEAR - 3, 1, 1)

# ---------------------------------------------------------------------------
# 1. STATIC REFERENCE DATA (names, securities)
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Margaret",
    "Donald", "Sandra", "Steven", "Ashley", "Paul", "Kimberly", "Andrew",
    "Emily", "Joshua", "Donna", "Kenneth", "Michelle", "Kevin", "Carol",
    "Brian", "Amanda", "George", "Melissa", "Edward", "Deborah", "Ronald",
    "Stephanie", "Wei", "Mei", "Hiroshi", "Yuki", "Priya", "Arjun", "Fatima",
    "Ahmed", "Olga", "Ivan", "Sofia", "Luca", "Giulia", "Klaus", "Anke",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Chen", "Kumar", "Singh", "Muller", "Rossi",
    "Ivanov", "Kowalski", "Novak", "Tanaka",
]
ENTITY_WORDS_1 = ["Silverline", "Northbridge", "Bluepeak", "Ironwood", "Harborview",
                  "Cedarstone", "Meridian", "Wavecrest", "Fieldstone", "Brightpath"]
ENTITY_WORDS_2 = ["Capital", "Holdings", "Partners", "Ventures", "Trust", "Advisors"]


def random_name(is_entity=False):
    if is_entity:
        return f"{random.choice(ENTITY_WORDS_1)} {random.choice(ENTITY_WORDS_2)} LLC"
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_date(start: date, end: date) -> date:
    """Uniform random date between start and end (inclusive)."""
    delta_days = (end - start).days
    if delta_days <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta_days))


def next_business_day(d: date, days_ahead: int) -> date:
    """Advance d by days_ahead settlement days, skipping weekends (simplified)."""
    result = d
    remaining = days_ahead
    while remaining > 0:
        result += timedelta(days=1)
        if result.weekday() < 5:  # Mon-Fri
            remaining -= 1
    return result


# Fictional security universe -------------------------------------------------
# type -> settlement days (simplified, consistent convention)
SETTLEMENT_DAYS = {
    "EQUITY": 1,
    "ETF": 1,
    "MUTUAL_FUND": 1,
    "CORPORATE_BOND": 2,
    "TREASURY": 1,
    "OPTION": 1,
}


def build_security_universe():
    """Builds a fixed, fictional list of tradable securities with stable
    attributes (starting price, dividend/interest behavior)."""
    securities = []

    # Equities (mix of dividend and non-dividend payers)
    equity_roots = ["ALP", "BRT", "CNV", "DLX", "ERV", "FLM", "GRT", "HZN",
                     "IVX", "JPL", "KNT", "LMR", "MVX", "NRD", "OPL", "PQR",
                     "QTN", "RSV", "STL", "TRX"]
    for i, root in enumerate(equity_roots):
        pays_div = i % 3 != 0  # ~2/3 of equities pay dividends
        securities.append({
            "symbol": f"{root}",
            "security_type": "EQUITY",
            "base_price": round(random.uniform(8, 350), 2),
            "annual_volatility": random.uniform(0.15, 0.55),
            "pays_dividend": pays_div,
            "div_yield": round(random.uniform(0.005, 0.035), 4) if pays_div else 0.0,
            "pays_interest": False,
        })

    # ETFs
    etf_roots = ["USEQ", "GLBX", "TCHF", "DIVG", "SMCP", "BNDX", "REIX", "ESGF"]
    for root in etf_roots:
        securities.append({
            "symbol": root,
            "security_type": "ETF",
            "base_price": round(random.uniform(20, 220), 2),
            "annual_volatility": random.uniform(0.10, 0.30),
            "pays_dividend": True,
            "div_yield": round(random.uniform(0.008, 0.025), 4),
            "pays_interest": False,
        })

    # Mutual funds
    mf_roots = ["GRWX", "VALX", "BALX", "INCX", "INTX"]
    for root in mf_roots:
        securities.append({
            "symbol": root,
            "security_type": "MUTUAL_FUND",
            "base_price": round(random.uniform(10, 90), 2),
            "annual_volatility": random.uniform(0.08, 0.25),
            "pays_dividend": True,
            "div_yield": round(random.uniform(0.005, 0.02), 4),
            "pays_interest": False,
        })

    # Corporate bonds (interest, priced near par)
    bond_roots = ["CBAA", "CBBB", "CBCC", "CBDD", "CBEE"]
    for root in bond_roots:
        securities.append({
            "symbol": root,
            "security_type": "CORPORATE_BOND",
            "base_price": round(random.uniform(95, 105), 2),
            "annual_volatility": 0.03,
            "pays_dividend": False,
            "div_yield": 0.0,
            "pays_interest": True,
            "coupon_rate": round(random.uniform(0.03, 0.065), 4),
        })

    # Treasuries
    tsy_roots = ["UST2Y", "UST5Y", "UST10Y"]
    for root in tsy_roots:
        securities.append({
            "symbol": root,
            "security_type": "TREASURY",
            "base_price": round(random.uniform(97, 103), 2),
            "annual_volatility": 0.015,
            "pays_dividend": False,
            "div_yield": 0.0,
            "pays_interest": True,
            "coupon_rate": round(random.uniform(0.015, 0.045), 4),
        })

    # A small option universe (underlying equities), simplified - no cost
    # basis tracking for options in this first-pass dataset.
    for root in equity_roots[:6]:
        securities.append({
            "symbol": f"{root}",
            "security_type": "OPTION",
            "base_price": round(random.uniform(0.5, 12), 2),
            "annual_volatility": 0.8,
            "pays_dividend": False,
            "div_yield": 0.0,
            "pays_interest": False,
        })

    return securities


SECURITIES = build_security_universe()
SEC_BY_TYPE = {}
for s in SECURITIES:
    SEC_BY_TYPE.setdefault(s["security_type"], []).append(s)
SEC_LOOKUP = {(s["symbol"], s["security_type"]): s for s in SECURITIES}


def _stable_hash(text: str) -> int:
    """Deterministic string hash (Python's built-in hash() is randomized
    per-process via PYTHONHASHSEED, which would break reproducibility)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def price_on(security, d: date):
    """Deterministic random-walk price so the same (security, date) always
    yields the same price on every run (fixed-seed, reproducible), staying
    near base_price with bounded drift."""
    seed_key = _stable_hash(f"{security['symbol']}|{security['security_type']}") ^ d.toordinal()
    rng = random.Random(seed_key & 0xFFFFFFFF)
    days_from_start = (d - HISTORY_START).days
    drift = 1 + rng.uniform(-1, 1) * security["annual_volatility"] * (days_from_start / 365.0) * 0.15
    noise = 1 + rng.uniform(-0.03, 0.03)
    p = security["base_price"] * max(drift, 0.2) * noise
    return round(max(p, 0.5), 2)


# ---------------------------------------------------------------------------
# 2. GENERATE accounts.csv
# ---------------------------------------------------------------------------
def generate_accounts(n=NUM_ACCOUNTS):
    rows = []
    account_type_choices = ["INDIVIDUAL", "JOINT", "TRADITIONAL_IRA", "ROTH_IRA", "ENTITY"]
    account_type_weights = [0.52, 0.20, 0.13, 0.10, 0.05]

    for i in range(1, n + 1):
        account_id = f"ACC{i:06d}"
        account_type = np.random.choice(account_type_choices, p=account_type_weights)
        is_entity = account_type == "ENTITY"
        customer_name = random_name(is_entity=is_entity)

        # ~90-95% US tax residency
        tax_residency = "US" if random.random() < 0.93 else "NON_US"

        # Tax document type / W9 / W8 status logic, kept mutually consistent
        if tax_residency == "US":
            tax_document_type = "W9"
            # Most valid; small share missing (drives backup withholding)
            w9_status = np.random.choice(
                ["VALID", "MISSING", "NOT_APPLICABLE"], p=[0.93, 0.05, 0.02]
            )
            w8_status = "NOT_APPLICABLE"
        else:
            tax_document_type = "W8BENE" if is_entity else "W8BEN"
            w9_status = "NOT_APPLICABLE"
            w8_status = np.random.choice(
                ["VALID", "EXPIRED", "MISSING"], p=[0.80, 0.12, 0.08]
            )

        # Backup withholding applies to US accounts with missing/invalid W9
        if tax_residency == "US" and w9_status == "MISSING":
            backup_withholding_flag = random.random() < 0.85
        else:
            backup_withholding_flag = random.random() < 0.01  # rare noise case

        # NRA withholding applies to non-US accounts on US-source income
        if tax_residency == "NON_US":
            nra_withholding_flag = random.random() < 0.90
        else:
            nra_withholding_flag = False

        opening_date = random_date(date(TAX_YEAR - 8, 1, 1), date(TAX_YEAR - 1, 12, 1))
        account_status = "OPEN" if random.random() < 0.93 else "CLOSED"

        rows.append({
            "account_id": account_id,
            "customer_name": customer_name,
            "account_type": account_type,
            "tax_residency": tax_residency,
            "tax_document_type": tax_document_type,
            "w9_status": w9_status,
            "w8_status": w8_status,
            "backup_withholding_flag": backup_withholding_flag,
            "nra_withholding_flag": nra_withholding_flag,
            "opening_date": opening_date.isoformat(),
            "account_status": account_status,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. GENERATE transactions.csv + cost_basis.csv (built together so lot
#    tracking / FIFO matching is guaranteed consistent)
# ---------------------------------------------------------------------------
DIVIDEND_DATES = [date(TAX_YEAR, 3, 15), date(TAX_YEAR, 6, 15),
                   date(TAX_YEAR, 9, 15), date(TAX_YEAR, 12, 15)]
INTEREST_DATES = [date(TAX_YEAR, 6, 30), date(TAX_YEAR, 12, 31)]


def generate_transactions_and_costbasis(accounts_df):
    txn_rows = []
    basis_rows = []

    txn_counter = 1
    basis_counter = 1

    def new_txn_id():
        nonlocal txn_counter
        tid = f"TXN{txn_counter:08d}"
        txn_counter += 1
        return tid

    def new_basis_id():
        nonlocal basis_counter
        bid = f"BAS{basis_counter:08d}"
        basis_counter += 1
        return bid

    # Activity-level distribution drives how many securities/txns per account
    activity_levels = np.random.choice(
        ["LOW", "MODERATE", "HIGH"], size=len(accounts_df), p=[0.60, 0.30, 0.10]
    )

    for (idx, acct), activity in zip(accounts_df.iterrows(), activity_levels):
        account_id = acct["account_id"]
        if acct["account_status"] == "CLOSED" and random.random() < 0.5:
            # Many closed accounts have little/no activity in the tax year
            continue

        if activity == "LOW":
            n_holdings = random.randint(1, 3)
        elif activity == "MODERATE":
            n_holdings = random.randint(3, 8)
        else:
            n_holdings = random.randint(8, 16)

        # Choose a mix of non-option securities for this account
        non_option_pool = [s for s in SECURITIES if s["security_type"] != "OPTION"]
        holdings = random.sample(non_option_pool, k=min(n_holdings, len(non_option_pool)))

        # Small chance of also trading a couple of options (kept minor overall)
        trades_options = random.random() < 0.12
        option_holdings = random.sample(SEC_BY_TYPE["OPTION"], k=random.choice([1, 2])) if trades_options else []

        for security in holdings:
            # --- Build 1-4 BUY lots for this account/security ---
            n_lots = random.choice([1, 1, 2, 2, 3, 4])
            lots = []  # each: dict(quantity_remaining, cost_per_share, acquisition_date, txn_id)

            for _ in range(n_lots):
                # ~30% chance a lot originates before the tax year (pre-existing position)
                if random.random() < 0.30:
                    buy_date = random_date(HISTORY_START, YEAR_START - timedelta(days=1))
                else:
                    buy_date = random_date(YEAR_START, date(TAX_YEAR, 11, 30))

                if security["security_type"] in ("CORPORATE_BOND", "TREASURY"):
                    quantity = random.choice([10, 20, 25, 50, 100])
                elif security["security_type"] == "MUTUAL_FUND":
                    quantity = round(random.uniform(20, 500), 3)
                else:
                    quantity = random.choice([5, 10, 25, 50, 75, 100, 150, 200])

                price = price_on(security, buy_date)
                gross_amount = round(quantity * price, 2)
                settle_days = SETTLEMENT_DAYS[security["security_type"]]
                settlement_date = next_business_day(buy_date, settle_days)

                tid = new_txn_id()
                txn_rows.append({
                    "transaction_id": tid,
                    "account_id": account_id,
                    "security_symbol": security["symbol"],
                    "security_type": security["security_type"],
                    "transaction_type": "BUY",
                    "trade_date": buy_date.isoformat(),
                    "settlement_date": settlement_date.isoformat(),
                    "quantity": quantity,
                    "price": price,
                    "gross_amount": gross_amount,
                    "currency": "USD",
                    "description": f"Purchase of {security['symbol']}",
                })
                lots.append({
                    "quantity_remaining": quantity,
                    "cost_per_share": price,
                    "acquisition_date": buy_date,
                })

            lots.sort(key=lambda l: l["acquisition_date"])  # FIFO order

            # --- Possibly SELL some/all of the position within the tax year ---
            total_owned = sum(l["quantity_remaining"] for l in lots)
            sells_position = random.random() < 0.55
            if sells_position and total_owned > 0:
                last_buy_date = max(l["acquisition_date"] for l in lots)
                earliest_possible_sell = max(last_buy_date + timedelta(days=1), YEAR_START)
                if earliest_possible_sell > YEAR_END:
                    sells_position = False
                else:
                    sell_date = random_date(earliest_possible_sell, YEAR_END)
                    sell_fraction = random.choice([0.25, 0.5, 0.75, 1.0])
                    qty_to_sell = round(total_owned * sell_fraction,
                                         3 if security["security_type"] == "MUTUAL_FUND" else 0)
                    qty_to_sell = max(qty_to_sell, 1 if security["security_type"] != "MUTUAL_FUND" else 0.001)
                    qty_to_sell = min(qty_to_sell, total_owned)

                    sale_price = price_on(security, sell_date)
                    proceeds_total = round(qty_to_sell * sale_price, 2)
                    settle_days = SETTLEMENT_DAYS[security["security_type"]]
                    settlement_date = next_business_day(sell_date, settle_days)

                    sell_tid = new_txn_id()
                    txn_rows.append({
                        "transaction_id": sell_tid,
                        "account_id": account_id,
                        "security_symbol": security["symbol"],
                        "security_type": security["security_type"],
                        "transaction_type": "SELL",
                        "trade_date": sell_date.isoformat(),
                        "settlement_date": settlement_date.isoformat(),
                        "quantity": qty_to_sell,
                        "price": sale_price,
                        "gross_amount": proceeds_total,
                        "currency": "USD",
                        "description": f"Sale of {security['symbol']}",
                    })

                    # Consume lots FIFO, creating one cost_basis row per lot touched
                    remaining_to_sell = qty_to_sell
                    for lot in lots:
                        if remaining_to_sell <= 0:
                            break
                        if lot["quantity_remaining"] <= 0:
                            continue
                        take = min(lot["quantity_remaining"], remaining_to_sell)
                        lot_cost_basis = round(take * lot["cost_per_share"], 2)
                        lot_proceeds = round(take * sale_price, 2)
                        realized_gain_loss = round(lot_proceeds - lot_cost_basis, 2)

                        # Wash sale adjustment: only ever applied to a small
                        # subset of LOSS lots, and always add-back (reduces loss)
                        wash_sale_adjustment = 0.00
                        if realized_gain_loss < 0 and random.random() < 0.04:
                            wash_sale_adjustment = round(abs(realized_gain_loss) * random.uniform(0.2, 1.0), 2)
                        adjusted_gain_loss = round(realized_gain_loss + wash_sale_adjustment, 2)

                        holding_days = (sell_date - lot["acquisition_date"]).days
                        holding_period = "LONG_TERM" if holding_days > 365 else "SHORT_TERM"
                        covered_status = "NON_COVERED" if random.random() < 0.08 else "COVERED"

                        basis_rows.append({
                            "basis_record_id": new_basis_id(),
                            "transaction_id": sell_tid,
                            "account_id": account_id,
                            "security_symbol": security["symbol"],
                            "security_type": security["security_type"],
                            "acquisition_date": lot["acquisition_date"].isoformat(),
                            "sale_date": sell_date.isoformat(),
                            "quantity_sold": take,
                            "cost_basis": lot_cost_basis,
                            "proceeds": lot_proceeds,
                            "realized_gain_loss": realized_gain_loss,
                            "wash_sale_adjustment": wash_sale_adjustment,
                            "adjusted_gain_loss": adjusted_gain_loss,
                            "covered_status": covered_status,
                            "holding_period": holding_period,
                        })

                        lot["quantity_remaining"] -= take
                        remaining_to_sell -= take

            # --- Dividends across the year for shares held on each dividend date ---
            if security.get("pays_dividend"):
                for ddate in DIVIDEND_DATES:
                    shares_held = sum(
                        l["quantity_remaining"] + 0  # remaining after all sells processed above
                        for l in lots if l["acquisition_date"] <= ddate
                    )
                    # crude re-derivation: shares actually held on ddate ~ what remains
                    # (sells are modeled as a single event; this approximates pre/post state)
                    if shares_held <= 0:
                        continue
                    per_share_div = round(price_on(security, ddate) * security["div_yield"] / 4, 4)
                    gross_amount = round(shares_held * per_share_div, 2)
                    if gross_amount <= 0:
                        continue
                    txn_rows.append({
                        "transaction_id": new_txn_id(),
                        "account_id": account_id,
                        "security_symbol": security["symbol"],
                        "security_type": security["security_type"],
                        "transaction_type": "DIVIDEND",
                        "trade_date": ddate.isoformat(),
                        "settlement_date": ddate.isoformat(),
                        "quantity": None,
                        "price": None,
                        "gross_amount": gross_amount,
                        "currency": "USD",
                        "description": f"Dividend payment - {security['symbol']}",
                    })

            # --- Interest for bonds / treasuries ---
            if security.get("pays_interest"):
                for idate in INTEREST_DATES:
                    shares_held = sum(l["quantity_remaining"] for l in lots if l["acquisition_date"] <= idate)
                    if shares_held <= 0:
                        continue
                    gross_amount = round(shares_held * security["coupon_rate"] / 2, 2)
                    if gross_amount <= 0:
                        continue
                    txn_rows.append({
                        "transaction_id": new_txn_id(),
                        "account_id": account_id,
                        "security_symbol": security["symbol"],
                        "security_type": security["security_type"],
                        "transaction_type": "INTEREST",
                        "trade_date": idate.isoformat(),
                        "settlement_date": idate.isoformat(),
                        "quantity": None,
                        "price": None,
                        "gross_amount": gross_amount,
                        "currency": "USD",
                        "description": f"Interest payment - {security['symbol']}",
                    })

        # --- Light option activity (kept a small share of overall volume) ---
        for opt in option_holdings:
            n_opt_txns = random.randint(1, 3)
            for _ in range(n_opt_txns):
                d = random_date(YEAR_START, YEAR_END)
                qty = random.choice([1, 2, 5, 10])
                px = price_on(opt, d)
                gross_amount = round(qty * px * 100, 2)  # 100 shares/contract
                settlement_date = next_business_day(d, SETTLEMENT_DAYS["OPTION"])
                txn_type = random.choice(["OPTION_BUY", "OPTION_SELL"])
                txn_rows.append({
                    "transaction_id": new_txn_id(),
                    "account_id": account_id,
                    "security_symbol": opt["symbol"],
                    "security_type": "OPTION",
                    "transaction_type": txn_type,
                    "trade_date": d.isoformat(),
                    "settlement_date": settlement_date.isoformat(),
                    "quantity": qty,
                    "price": px,
                    "gross_amount": gross_amount,
                    "currency": "USD",
                    "description": f"{txn_type.replace('_', ' ').title()} {opt['symbol']}",
                })

        # --- Occasional RETURN_OF_CAPITAL event for a mutual-fund/ETF holder ---
        roc_candidates = [h for h in holdings if h["security_type"] in ("MUTUAL_FUND", "ETF")]
        if roc_candidates and random.random() < 0.06:
            sec = random.choice(roc_candidates)
            d = random_date(date(TAX_YEAR, 10, 1), YEAR_END)
            gross_amount = round(random.uniform(5, 150), 2)
            txn_rows.append({
                "transaction_id": new_txn_id(),
                "account_id": account_id,
                "security_symbol": sec["symbol"],
                "security_type": sec["security_type"],
                "transaction_type": "RETURN_OF_CAPITAL",
                "trade_date": d.isoformat(),
                "settlement_date": d.isoformat(),
                "quantity": None,
                "price": None,
                "gross_amount": gross_amount,
                "currency": "USD",
                "description": f"Return of capital - {sec['symbol']}",
            })

    transactions_df = pd.DataFrame(txn_rows)
    cost_basis_df = pd.DataFrame(basis_rows)
    return transactions_df, cost_basis_df


# ---------------------------------------------------------------------------
# 4. GENERATE withholding.csv
# ---------------------------------------------------------------------------
def generate_withholding(accounts_df, transactions_df):
    rows = []
    counter = 1

    def new_wh_id():
        nonlocal counter
        wid = f"WH{counter:08d}"
        counter += 1
        return wid

    acct_lookup = accounts_df.set_index("account_id").to_dict("index")

    reportable_types = {"DIVIDEND": "DIVIDEND", "INTEREST": "INTEREST", "SELL": "SALE_PROCEEDS"}
    relevant_txns = transactions_df[transactions_df["transaction_type"].isin(reportable_types.keys())]

    for _, txn in relevant_txns.iterrows():
        acct = acct_lookup.get(txn["account_id"])
        if acct is None:
            continue

        backup = acct["backup_withholding_flag"]
        nra = acct["nra_withholding_flag"]
        if not backup and not nra:
            # small amount of incidental state withholding on US dividends/interest
            if txn["transaction_type"] in ("DIVIDEND", "INTEREST") and random.random() < 0.03:
                gross_payment = txn["gross_amount"]
                rate = round(random.uniform(0.03, 0.07), 4)
                amount = round(gross_payment * rate, 2)
                rows.append({
                    "withholding_id": new_wh_id(),
                    "account_id": txn["account_id"],
                    "transaction_id": txn["transaction_id"],
                    "withholding_type": "STATE_WITHHOLDING",
                    "income_type": reportable_types[txn["transaction_type"]],
                    "gross_payment": gross_payment,
                    "withholding_rate": rate,
                    "withholding_amount": amount,
                    "withholding_date": txn["trade_date"],
                    "documentation_status": "VALID_DOCUMENTATION",
                })
            continue

        # Backup withholding: applies broadly to reportable payments (24% statutory rate)
        if backup and random.random() < 0.90:
            gross_payment = txn["gross_amount"]
            rate = 0.24
            amount = round(gross_payment * rate, 2)
            doc_status = "MISSING_DOCUMENTATION" if acct["w9_status"] == "MISSING" else "VALID_DOCUMENTATION"
            rows.append({
                "withholding_id": new_wh_id(),
                "account_id": txn["account_id"],
                "transaction_id": txn["transaction_id"],
                "withholding_type": "BACKUP_WITHHOLDING",
                "income_type": reportable_types[txn["transaction_type"]],
                "gross_payment": gross_payment,
                "withholding_rate": rate,
                "withholding_amount": amount,
                "withholding_date": txn["trade_date"],
                "documentation_status": doc_status,
            })
            continue

        # NRA withholding: applies to US-source dividend/interest income (not typically sale proceeds)
        if nra and txn["transaction_type"] in ("DIVIDEND", "INTEREST") and random.random() < 0.92:
            gross_payment = txn["gross_amount"]
            rate = round(random.choice([0.30, 0.15, 0.10]), 4)  # statutory or common treaty rates
            amount = round(gross_payment * rate, 2)
            if acct["w8_status"] == "MISSING":
                doc_status = "MISSING_DOCUMENTATION"
            elif acct["w8_status"] == "EXPIRED":
                doc_status = "EXPIRED_DOCUMENTATION"
            else:
                doc_status = "VALID_DOCUMENTATION"
            rows.append({
                "withholding_id": new_wh_id(),
                "account_id": txn["account_id"],
                "transaction_id": txn["transaction_id"],
                "withholding_type": "NRA_WITHHOLDING",
                "income_type": reportable_types[txn["transaction_type"]],
                "gross_payment": gross_payment,
                "withholding_rate": rate,
                "withholding_amount": amount,
                "withholding_date": txn["trade_date"],
                "documentation_status": doc_status,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. GENERATE tax_forms.csv
# ---------------------------------------------------------------------------
def generate_tax_forms(accounts_df, transactions_df, cost_basis_df, withholding_df):
    rows = []
    counter = 1

    def new_form_id():
        nonlocal counter
        fid = f"FORM{counter:08d}"
        counter += 1
        return fid

    status_choices = ["GENERATED", "FILED", "CORRECTED"]
    status_weights = [0.55, 0.43, 0.02]

    # Pre-aggregate withholding by account + income_type for federal_tax_withheld.
    # STATE_WITHHOLDING is deliberately excluded here: it is a state-level
    # amount, not a federal one, and must never feed federal_tax_withheld.
    federal_wh = withholding_df[withholding_df["withholding_type"] != "STATE_WITHHOLDING"]
    wh_by_acct_income = (
        federal_wh.groupby(["account_id", "income_type"])["withholding_amount"]
        .sum().round(2).to_dict()
    ) if not federal_wh.empty else {}

    # Pre-aggregate cost basis (1099-B) by account
    basis_by_acct = (
        cost_basis_df.groupby("account_id")
        .agg(reported_proceeds=("proceeds", "sum"),
             reported_cost_basis=("cost_basis", "sum"),
             reported_gain_loss=("adjusted_gain_loss", "sum"))
        .round(2)
    ) if not cost_basis_df.empty else pd.DataFrame()

    # Pre-aggregate dividend / interest income by account
    div_by_acct = (
        transactions_df[transactions_df["transaction_type"] == "DIVIDEND"]
        .groupby("account_id")["gross_amount"].sum().round(2)
    )
    int_by_acct = (
        transactions_df[transactions_df["transaction_type"] == "INTEREST"]
        .groupby("account_id")["gross_amount"].sum().round(2)
    )

    for _, acct in accounts_df.iterrows():
        account_id = acct["account_id"]
        is_retirement = acct["account_type"] in ("TRADITIONAL_IRA", "ROTH_IRA")
        is_non_us = acct["tax_residency"] == "NON_US"

        proceeds_row = basis_by_acct.loc[account_id] if account_id in basis_by_acct.index else None
        dividends = div_by_acct.get(account_id, 0.0)
        interest = int_by_acct.get(account_id, 0.0)

        # Non-US status is checked BEFORE retirement status: a non-US person's
        # US-source dividend/interest income is reportable on 1042-S with NRA
        # withholding regardless of whether the account wrapper is a
        # retirement account, so this ordering prevents a non-US IRA's income
        # from silently disappearing from every tax form.
        if is_retirement and not is_non_us:
            # US retirement accounts don't get 1099-B/DIV/INT; a subset get a
            # simplified 1099-R representing a distribution during the year.
            if random.random() < 0.22:
                # Gross retirement distribution is carried in reported_proceeds
                # for 1099-R rows (the schema has no dedicated distribution
                # column); reported_cost_basis/gain_loss/dividends/interest
                # are not applicable and left at 0.00.
                distribution = round(random.uniform(500, 25000), 2)
                fed_withheld = round(distribution * 0.10, 2) if random.random() < 0.5 else 0.00
                rows.append({
                    "tax_form_id": new_form_id(),
                    "account_id": account_id,
                    "form_type": "1099-R",
                    "tax_year": TAX_YEAR,
                    "reported_proceeds": distribution,
                    "reported_cost_basis": 0.00,
                    "reported_gain_loss": 0.00,
                    "reported_dividends": 0.00,
                    "reported_interest": 0.00,
                    "federal_tax_withheld": fed_withheld,
                    "status": np.random.choice(status_choices, p=status_weights),
                })
            continue

        if is_non_us:
            # Non-US accounts (including non-US IRAs): US-source
            # dividend/interest reported on 1042-S
            if dividends > 0 or interest > 0:
                fed_withheld = round(
                    wh_by_acct_income.get((account_id, "DIVIDEND"), 0.0)
                    + wh_by_acct_income.get((account_id, "INTEREST"), 0.0), 2
                )
                rows.append({
                    "tax_form_id": new_form_id(),
                    "account_id": account_id,
                    "form_type": "1042-S",
                    "tax_year": TAX_YEAR,
                    "reported_proceeds": 0.00,
                    "reported_cost_basis": 0.00,
                    "reported_gain_loss": 0.00,
                    "reported_dividends": float(dividends),
                    "reported_interest": float(interest),
                    "federal_tax_withheld": fed_withheld,
                    "status": np.random.choice(status_choices, p=status_weights),
                })
            # Non-US accounts generally do NOT receive 1099-B/DIV/INT
            continue

        # US taxable accounts (INDIVIDUAL / JOINT / ENTITY)
        if proceeds_row is not None:
            fed_withheld = round(wh_by_acct_income.get((account_id, "SALE_PROCEEDS"), 0.0), 2)
            rows.append({
                "tax_form_id": new_form_id(),
                "account_id": account_id,
                "form_type": "1099-B",
                "tax_year": TAX_YEAR,
                "reported_proceeds": float(proceeds_row["reported_proceeds"]),
                "reported_cost_basis": float(proceeds_row["reported_cost_basis"]),
                "reported_gain_loss": float(proceeds_row["reported_gain_loss"]),
                "reported_dividends": 0.00,
                "reported_interest": 0.00,
                "federal_tax_withheld": fed_withheld,
                "status": np.random.choice(status_choices, p=status_weights),
            })

        if dividends > 0:
            fed_withheld = round(wh_by_acct_income.get((account_id, "DIVIDEND"), 0.0), 2)
            rows.append({
                "tax_form_id": new_form_id(),
                "account_id": account_id,
                "form_type": "1099-DIV",
                "tax_year": TAX_YEAR,
                "reported_proceeds": 0.00,
                "reported_cost_basis": 0.00,
                "reported_gain_loss": 0.00,
                "reported_dividends": float(dividends),
                "reported_interest": 0.00,
                "federal_tax_withheld": fed_withheld,
                "status": np.random.choice(status_choices, p=status_weights),
            })

        if interest > 0:
            fed_withheld = round(wh_by_acct_income.get((account_id, "INTEREST"), 0.0), 2)
            rows.append({
                "tax_form_id": new_form_id(),
                "account_id": account_id,
                "form_type": "1099-INT",
                "tax_year": TAX_YEAR,
                "reported_proceeds": 0.00,
                "reported_cost_basis": 0.00,
                "reported_gain_loss": 0.00,
                "reported_dividends": 0.00,
                "reported_interest": float(interest),
                "federal_tax_withheld": fed_withheld,
                "status": np.random.choice(status_choices, p=status_weights),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. VALIDATION
# ---------------------------------------------------------------------------
def validate(accounts_df, transactions_df, cost_basis_df, tax_forms_df, withholding_df):
    errors = []

    # 6.1 Referential integrity: account_id
    valid_accounts = set(accounts_df["account_id"])
    for name, df in [("transactions", transactions_df), ("cost_basis", cost_basis_df),
                      ("tax_forms", tax_forms_df), ("withholding", withholding_df)]:
        bad = set(df["account_id"]) - valid_accounts
        if bad:
            errors.append(f"[{name}.csv] contains {len(bad)} account_id values not present in accounts.csv")

    # 6.2 Referential integrity: transaction_id
    valid_txns = set(transactions_df["transaction_id"])
    bad_cb = set(cost_basis_df["transaction_id"]) - valid_txns
    if bad_cb:
        errors.append(f"[cost_basis.csv] contains {len(bad_cb)} transaction_id values not present in transactions.csv")
    bad_wh = set(withholding_df["transaction_id"]) - valid_txns
    if bad_wh:
        errors.append(f"[withholding.csv] contains {len(bad_wh)} transaction_id values not present in transactions.csv")

    # 6.3 Uniqueness of primary keys
    for name, df, col in [("accounts", accounts_df, "account_id"),
                           ("transactions", transactions_df, "transaction_id"),
                           ("cost_basis", cost_basis_df, "basis_record_id"),
                           ("withholding", withholding_df, "withholding_id"),
                           ("tax_forms", tax_forms_df, "tax_form_id")]:
        if df[col].duplicated().any():
            errors.append(f"[{name}.csv] duplicate {col} values found")

    # 6.4 Acquisition date must precede sale date
    cb = cost_basis_df.copy()
    cb["acquisition_date"] = pd.to_datetime(cb["acquisition_date"])
    cb["sale_date"] = pd.to_datetime(cb["sale_date"])
    bad_dates = cb[cb["acquisition_date"] >= cb["sale_date"]]
    if len(bad_dates) > 0:
        errors.append(f"[cost_basis.csv] {len(bad_dates)} rows have acquisition_date >= sale_date")

    # 6.5 Cost basis arithmetic: realized_gain_loss = proceeds - cost_basis
    calc = (cb["proceeds"] - cb["cost_basis"]).round(2)
    mismatch = (calc - cb["realized_gain_loss"]).abs() > 0.01
    if mismatch.any():
        errors.append(f"[cost_basis.csv] {mismatch.sum()} rows where realized_gain_loss != proceeds - cost_basis")

    # 6.6 adjusted_gain_loss consistency
    expected_adj = (cb["realized_gain_loss"] + cb["wash_sale_adjustment"]).round(2)
    mismatch_adj = (expected_adj - cb["adjusted_gain_loss"]).abs() > 0.01
    if mismatch_adj.any():
        errors.append(f"[cost_basis.csv] {mismatch_adj.sum()} rows where adjusted_gain_loss != realized_gain_loss + wash_sale_adjustment")

    # 6.7 BUY/SELL gross_amount ~= quantity * price
    trades = transactions_df[transactions_df["transaction_type"].isin(["BUY", "SELL"])].copy()
    calc_gross = (trades["quantity"].astype(float) * trades["price"].astype(float)).round(2)
    mismatch_gross = (calc_gross - trades["gross_amount"]).abs() > 0.02
    if mismatch_gross.any():
        errors.append(f"[transactions.csv] {mismatch_gross.sum()} BUY/SELL rows where gross_amount != quantity * price")

    # 6.8 Withholding arithmetic
    wh = withholding_df.copy()
    calc_wh = (wh["gross_payment"] * wh["withholding_rate"]).round(2)
    mismatch_wh = (calc_wh - wh["withholding_amount"]).abs() > 0.02
    if mismatch_wh.any():
        errors.append(f"[withholding.csv] {mismatch_wh.sum()} rows where withholding_amount != gross_payment * withholding_rate")

    # 6.9 1099-B reconciliation to cost_basis.csv
    basis_totals = cost_basis_df.groupby("account_id").agg(
        proceeds=("proceeds", "sum"), cost_basis=("cost_basis", "sum"),
        gain_loss=("adjusted_gain_loss", "sum")
    ).round(2)
    b_forms = tax_forms_df[tax_forms_df["form_type"] == "1099-B"]
    recon_mismatches = 0
    for _, f in b_forms.iterrows():
        t = basis_totals.loc[f["account_id"]]
        if (abs(t["proceeds"] - f["reported_proceeds"]) > 0.05 or
                abs(t["cost_basis"] - f["reported_cost_basis"]) > 0.05 or
                abs(t["gain_loss"] - f["reported_gain_loss"]) > 0.05):
            recon_mismatches += 1
    if recon_mismatches:
        errors.append(f"[tax_forms.csv] {recon_mismatches} 1099-B rows do not reconcile to cost_basis.csv totals")

    # 6.10 W9/W8 alignment with residency
    bad_res = accounts_df[
        ((accounts_df["tax_residency"] == "US") & (accounts_df["tax_document_type"] != "W9")) |
        ((accounts_df["tax_residency"] == "NON_US") & (~accounts_df["tax_document_type"].isin(["W8BEN", "W8BENE"])))
    ]
    if len(bad_res) > 0:
        errors.append(f"[accounts.csv] {len(bad_res)} rows have tax_document_type inconsistent with tax_residency")

    if errors:
        print("\n".join(f"  - {e}" for e in errors))
        raise ValueError(f"Validation failed with {len(errors)} error type(s). See details above.")

    print("All validation checks passed.")
    return True


# ---------------------------------------------------------------------------
# 7. PHASE 1 QA REPORT
# ---------------------------------------------------------------------------
def _pf(label, passed, detail=""):
    """Print a PASS/FAIL line and return the boolean so callers can collect it."""
    tag = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {label}: {tag}{suffix}")
    return passed


def run_qa_report(accounts_df, transactions_df, cost_basis_df, tax_forms_df, withholding_df):
    """Runs a comprehensive, automated Phase-1 QA pass over the five
    generated datasets and prints a human-readable report. Nothing in this
    function modifies the underlying data - it only reads and reports.
    Returns True if every section that can meaningfully fail passed."""

    results = {}  # section_name -> bool, used for the final summary

    # Merge account attributes onto transactions/cost_basis once, reused below
    acct_attrs = accounts_df[["account_id", "account_type", "tax_residency",
                               "w9_status", "w8_status", "tax_document_type",
                               "backup_withholding_flag", "nra_withholding_flag"]]
    txn_m = transactions_df.merge(acct_attrs, on="account_id", how="left")
    cb_m = cost_basis_df.merge(acct_attrs, on="account_id", how="left")

    RETIREMENT_TYPES = ("TRADITIONAL_IRA", "ROTH_IRA")

    print("\n" + "=" * 78)
    print("PHASE 1 AUTOMATED QA REPORT")
    print("=" * 78)

    # =======================================================================
    # 1. UNIQUE ACCOUNT ACTIVITY
    # =======================================================================
    print("\n--- 1. UNIQUE ACCOUNT ACTIVITY ---")
    total_accounts = accounts_df["account_id"].nunique()
    accts_with_txn = transactions_df["account_id"].nunique()
    accts_with_tax_activity = tax_forms_df["account_id"].nunique()
    accts_no_activity = total_accounts - accounts_df["account_id"].isin(
        transactions_df["account_id"]
    ).sum()

    print(f"Total accounts:                          {total_accounts:,}")
    print(f"Accounts with >=1 transaction:            {accts_with_txn:,}")
    print(f"Accounts with reportable tax activity:    {accts_with_tax_activity:,}")
    print(f"Accounts with no transaction activity:    {accts_no_activity:,}")

    # =======================================================================
    # 2. COST BASIS RECONCILIATION
    # =======================================================================
    print("\n--- 2. COST BASIS RECONCILIATION ---")
    total_sell_txn = transactions_df.loc[transactions_df.transaction_type == "SELL", "gross_amount"].sum()
    total_cb_proceeds = cost_basis_df["proceeds"].sum()
    proceeds_diff = round(total_sell_txn - total_cb_proceeds, 2)
    # Tolerance scales with row count since each row can round by up to $0.005
    proceeds_tolerance = max(1.00, 0.01 * len(cost_basis_df))
    proceeds_ok = abs(proceeds_diff) <= proceeds_tolerance

    print(f"Total SELL transaction proceeds:   ${total_sell_txn:,.2f}")
    print(f"Total proceeds (cost_basis.csv):   ${total_cb_proceeds:,.2f}")
    print(f"Difference:                        ${proceeds_diff:,.2f}  (tolerance ${proceeds_tolerance:,.2f})")
    results["cost_basis_proceeds_tie_out"] = _pf("SELL vs cost_basis proceeds tie-out", proceeds_ok)

    total_cost_basis = cost_basis_df["cost_basis"].sum()
    total_realized = cost_basis_df["realized_gain_loss"].sum()
    total_adjusted = cost_basis_df["adjusted_gain_loss"].sum()
    print(f"Total cost basis:                  ${total_cost_basis:,.2f}")
    print(f"Total realized gain/loss:           ${total_realized:,.2f}")
    print(f"Total adjusted gain/loss:            ${total_adjusted:,.2f}")

    calc_realized = (cost_basis_df["proceeds"] - cost_basis_df["cost_basis"]).round(2)
    realized_fail_n = int((calc_realized - cost_basis_df["realized_gain_loss"]).abs().gt(0.01).sum())
    calc_adjusted = (cost_basis_df["realized_gain_loss"] + cost_basis_df["wash_sale_adjustment"]).round(2)
    adjusted_fail_n = int((calc_adjusted - cost_basis_df["adjusted_gain_loss"]).abs().gt(0.01).sum())

    print(f"Records failing realized_gain_loss = proceeds - cost_basis: {realized_fail_n}")
    print(f"Records failing adjusted_gain_loss = realized + wash_sale_adjustment: {adjusted_fail_n}")
    results["cost_basis_math"] = _pf(
        "Cost basis arithmetic", realized_fail_n == 0 and adjusted_fail_n == 0
    )

    # =======================================================================
    # 3. GAIN/LOSS SANITY CHECK
    # =======================================================================
    print("\n--- 3. GAIN/LOSS SANITY CHECK ---")
    rgl = cost_basis_df["realized_gain_loss"]
    n_dispositions = len(rgl)
    breakeven_tol = 1.00  # within $1 of zero counts as "approximately breakeven"

    n_profit = int((rgl > breakeven_tol).sum())
    n_loss = int((rgl < -breakeven_tol).sum())
    n_breakeven = n_dispositions - n_profit - n_loss

    total_gains = rgl[rgl > 0].sum()
    total_losses = rgl[rgl < 0].sum()
    net_gain_loss = rgl.sum()

    print(f"Number of dispositions:            {n_dispositions:,}")
    print(f"Average realized gain/loss:        ${rgl.mean():,.2f}")
    print(f"Median realized gain/loss:         ${rgl.median():,.2f}")
    print(f"Largest gain:                      ${rgl.max():,.2f}")
    print(f"Largest loss:                      ${rgl.min():,.2f}")
    print(f"Profitable dispositions:           {n_profit:,} ({n_profit / n_dispositions:.1%})")
    print(f"Loss dispositions:                 {n_loss:,} ({n_loss / n_dispositions:.1%})")
    print(f"Approx. breakeven (|gain| <= $1):  {n_breakeven:,} ({n_breakeven / n_dispositions:.1%})")
    print(f"Total gains:                        ${total_gains:,.2f}")
    print(f"Total losses:                       ${total_losses:,.2f}")
    print(f"Net gain/loss:                      ${net_gain_loss:,.2f}")

    holding_counts = cost_basis_df["holding_period"].value_counts()
    print(f"SHORT_TERM dispositions:           {int(holding_counts.get('SHORT_TERM', 0)):,}")
    print(f"LONG_TERM dispositions:            {int(holding_counts.get('LONG_TERM', 0)):,}")

    # Flag (not fail) if returns look unrealistically clustered at zero
    breakeven_pct = n_breakeven / n_dispositions
    distribution_flag_ok = breakeven_pct <= 0.15
    if distribution_flag_ok:
        print(f"Distribution check: PASS  (breakeven share {breakeven_pct:.1%} is within a realistic range)")
    else:
        print(f"Distribution check: REVIEW  (breakeven share {breakeven_pct:.1%} looks concentrated near zero)")
    results["gain_loss_distribution"] = distribution_flag_ok

    # =======================================================================
    # 4. TAX FORM RECONCILIATION
    # =======================================================================
    print("\n--- 4. TAX FORM RECONCILIATION ---")
    tol = 1.00  # flat $1 tolerance for aggregate reconciliations below
    form_checks_ok = []

    # -- 1099-B: only US, non-retirement accounts are eligible by design --
    us_taxable_mask_cb = (cb_m["tax_residency"] == "US") & (~cb_m["account_type"].isin(RETIREMENT_TYPES))
    expected_proceeds = cb_m.loc[us_taxable_mask_cb, "proceeds"].sum()
    expected_cost_basis = cb_m.loc[us_taxable_mask_cb, "cost_basis"].sum()
    b_forms = tax_forms_df[tax_forms_df.form_type == "1099-B"]
    reported_proceeds = b_forms["reported_proceeds"].sum()
    reported_cost_basis = b_forms["reported_cost_basis"].sum()
    diff_proceeds = round(expected_proceeds - reported_proceeds, 2)
    diff_cb = round(expected_cost_basis - reported_cost_basis, 2)
    ok_b = abs(diff_proceeds) <= tol and abs(diff_cb) <= tol
    print("1099-B (proceeds):")
    print(f"  EXPECTED (eligible cost_basis rows): ${expected_proceeds:,.2f}")
    print(f"  REPORTED (tax_forms.csv):             ${reported_proceeds:,.2f}")
    print(f"  DIFFERENCE:                           ${diff_proceeds:,.2f}")
    print("1099-B (cost basis):")
    print(f"  EXPECTED: ${expected_cost_basis:,.2f}   REPORTED: ${reported_cost_basis:,.2f}   DIFFERENCE: ${diff_cb:,.2f}")
    _pf("1099-B reconciliation", ok_b)
    form_checks_ok.append(ok_b)
    print("  Note: cost_basis rows belonging to retirement or non-US accounts are")
    print("  intentionally excluded from 1099-B by design (IRAs don't receive a")
    print("  1099-B; non-US capital gains are not modeled on 1042-S in this dataset).")

    # -- 1099-DIV: US, non-retirement accounts only --
    us_taxable_mask_txn = (txn_m["tax_residency"] == "US") & (~txn_m["account_type"].isin(RETIREMENT_TYPES))
    expected_div = txn_m.loc[us_taxable_mask_txn & (txn_m.transaction_type == "DIVIDEND"), "gross_amount"].sum()
    reported_div = tax_forms_df.loc[tax_forms_df.form_type == "1099-DIV", "reported_dividends"].sum()
    diff_div = round(expected_div - reported_div, 2)
    ok_div = abs(diff_div) <= tol
    print("1099-DIV:")
    print(f"  EXPECTED: ${expected_div:,.2f}   REPORTED: ${reported_div:,.2f}   DIFFERENCE: ${diff_div:,.2f}")
    _pf("1099-DIV reconciliation", ok_div)
    form_checks_ok.append(ok_div)

    # -- 1099-INT: US, non-retirement accounts only --
    expected_int = txn_m.loc[us_taxable_mask_txn & (txn_m.transaction_type == "INTEREST"), "gross_amount"].sum()
    reported_int = tax_forms_df.loc[tax_forms_df.form_type == "1099-INT", "reported_interest"].sum()
    diff_int = round(expected_int - reported_int, 2)
    ok_int = abs(diff_int) <= tol
    print("1099-INT:")
    print(f"  EXPECTED: ${expected_int:,.2f}   REPORTED: ${reported_int:,.2f}   DIFFERENCE: ${diff_int:,.2f}")
    _pf("1099-INT reconciliation", ok_int)
    form_checks_ok.append(ok_int)

    # -- 1042-S: non-US accounts' dividend + interest income --
    non_us_mask_txn = txn_m["tax_residency"] == "NON_US"
    expected_1042_div = txn_m.loc[non_us_mask_txn & (txn_m.transaction_type == "DIVIDEND"), "gross_amount"].sum()
    expected_1042_int = txn_m.loc[non_us_mask_txn & (txn_m.transaction_type == "INTEREST"), "gross_amount"].sum()
    s_forms = tax_forms_df[tax_forms_df.form_type == "1042-S"]
    reported_1042_div = s_forms["reported_dividends"].sum()
    reported_1042_int = s_forms["reported_interest"].sum()
    reported_1042_wh = s_forms["federal_tax_withheld"].sum()
    diff_1042_div = round(expected_1042_div - reported_1042_div, 2)
    diff_1042_int = round(expected_1042_int - reported_1042_int, 2)
    ok_1042 = abs(diff_1042_div) <= tol and abs(diff_1042_int) <= tol
    print("1042-S (non-US dividend + interest income):")
    print(f"  EXPECTED dividends: ${expected_1042_div:,.2f}   REPORTED: ${reported_1042_div:,.2f}   DIFFERENCE: ${diff_1042_div:,.2f}")
    print(f"  EXPECTED interest:  ${expected_1042_int:,.2f}   REPORTED: ${reported_1042_int:,.2f}   DIFFERENCE: ${diff_1042_int:,.2f}")
    print(f"  Reported withholding on 1042-S forms: ${reported_1042_wh:,.2f}")
    _pf("1042-S reconciliation", ok_1042)
    form_checks_ok.append(ok_1042)

    results["tax_form_reconciliation"] = all(form_checks_ok)

    # =======================================================================
    # 5. WITHHOLDING RECONCILIATION
    # =======================================================================
    print("\n--- 5. WITHHOLDING RECONCILIATION ---")
    by_type = withholding_df.groupby("withholding_type").agg(
        records=("withholding_id", "count"),
        gross_payment=("gross_payment", "sum"),
        withholding_amount=("withholding_amount", "sum"),
    ).round(2)
    for wtype in ["BACKUP_WITHHOLDING", "NRA_WITHHOLDING", "STATE_WITHHOLDING"]:
        if wtype in by_type.index:
            row = by_type.loc[wtype]
            print(f"{wtype}: {int(row['records']):,} records | gross ${row['gross_payment']:,.2f} | withheld ${row['withholding_amount']:,.2f}")
        else:
            print(f"{wtype}: 0 records")

    total_wh_detail = withholding_df["withholding_amount"].sum()
    state_wh = withholding_df.loc[withholding_df.withholding_type == "STATE_WITHHOLDING", "withholding_amount"].sum()

    print("\nWhy federal_tax_withheld on tax_forms.csv differs from withholding.csv:")
    print("  - STATE_WITHHOLDING is a state-level amount, not federal, so it is")
    print("    correctly EXCLUDED from federal_tax_withheld on every form.")
    print("  - BACKUP_WITHHOLDING and NRA_WITHHOLDING on retirement (IRA) accounts")
    print("    are excluded too: IRAs receive a 1099-R instead of a 1099-B/DIV/INT,")
    print("    and the 1099-R's federal_tax_withheld models a separate, independently")
    print("    elected plan withholding rather than passing through withholding.csv.")

    # Only US retirement accounts route through 1099-R (non-US IRAs still get
    # a 1042-S, per the generator's non-US-first branch ordering), so only
    # their backup/NRA withholding is excluded from the forms comparison here.
    us_retirement_ids = set(accounts_df.loc[
        accounts_df.account_type.isin(RETIREMENT_TYPES) & (accounts_df.tax_residency == "US"),
        "account_id"
    ])
    wh_on_retirement = withholding_df[
        withholding_df.account_id.isin(us_retirement_ids)
        & withholding_df.withholding_type.isin(["BACKUP_WITHHOLDING", "NRA_WITHHOLDING"])
    ]["withholding_amount"].sum()

    federal_expected_on_forms = round(total_wh_detail - state_wh - wh_on_retirement, 2)
    # Compare against federal_tax_withheld on every form EXCEPT 1099-R, since
    # 1099-R withholding is independently modeled and not sourced from withholding.csv
    federal_reported_excl_1099r = tax_forms_df.loc[
        tax_forms_df.form_type != "1099-R", "federal_tax_withheld"
    ].sum()
    federal_diff = round(federal_expected_on_forms - federal_reported_excl_1099r, 2)
    federal_ok = abs(federal_diff) <= tol

    print(f"\nTotal withholding detail (withholding.csv): ${total_wh_detail:,.2f}")
    print(f"Less STATE_WITHHOLDING:                      ${state_wh:,.2f}")
    print(f"Less BACKUP/NRA withholding on IRA accounts:  ${wh_on_retirement:,.2f}  (routed to 1099-R instead)")
    print(f"Federal withholding expected on forms:        ${federal_expected_on_forms:,.2f}")
    print(f"Federal withholding actually reported")
    print(f"  (all forms excluding 1099-R):               ${federal_reported_excl_1099r:,.2f}")
    print(f"Difference:                                   ${federal_diff:,.2f}")
    results["withholding_reconciliation"] = _pf("Withholding reconciliation (excl. 1099-R)", federal_ok)

    # =======================================================================
    # 6. DOCUMENTATION LOGIC
    # =======================================================================
    print("\n--- 6. DOCUMENTATION LOGIC ---")
    issues = []

    us_wrong_doc = accounts_df[
        (accounts_df.tax_residency == "US") & (accounts_df.tax_document_type.isin(["W8BEN", "W8BENE"]))
    ]
    issues.append(("US resident with W8BEN/W8BENE as primary documentation", len(us_wrong_doc)))

    nonus_wrong_doc = accounts_df[
        (accounts_df.tax_residency == "NON_US") & (accounts_df.tax_document_type == "W9")
    ]
    issues.append(("NON_US resident relying on W9", len(nonus_wrong_doc)))

    w9_valid_wrong_doctype = accounts_df[
        (accounts_df.w9_status == "VALID") & (accounts_df.tax_document_type != "W9")
    ]
    issues.append(("w9_status VALID but tax_document_type is not W9", len(w9_valid_wrong_doctype)))

    w8_valid_on_us = accounts_df[
        (accounts_df.tax_residency == "US") & (accounts_df.w8_status != "NOT_APPLICABLE")
    ]
    issues.append(("US account where w8_status is not NOT_APPLICABLE", len(w8_valid_on_us)))

    nra_flag_on_us = accounts_df[
        (accounts_df.tax_residency == "US") & (accounts_df.nra_withholding_flag == True)  # noqa: E712
    ]
    issues.append(("nra_withholding_flag set TRUE on a US resident account", len(nra_flag_on_us)))

    total_issues = 0
    for label, count in issues:
        print(f"  {label}: {count}")
        total_issues += count
    print(f"Total documentation inconsistencies found: {total_issues}")
    results["documentation_logic"] = _pf("Documentation logic", total_issues == 0)

    # =======================================================================
    # 7. REFERENTIAL INTEGRITY
    # =======================================================================
    print("\n--- 7. REFERENTIAL INTEGRITY ---")
    valid_accounts = set(accounts_df["account_id"])
    valid_txns = set(transactions_df["transaction_id"])

    ri_checks = []
    for name, df in [("transactions.csv", transactions_df), ("cost_basis.csv", cost_basis_df),
                      ("tax_forms.csv", tax_forms_df), ("withholding.csv", withholding_df)]:
        bad = set(df["account_id"]) - valid_accounts
        ri_checks.append(_pf(f"All account_id in {name} exist in accounts.csv", len(bad) == 0,
                              f"{len(bad)} orphaned" if bad else ""))

    bad_cb_txn = set(cost_basis_df["transaction_id"]) - valid_txns
    ri_checks.append(_pf("All transaction_id in cost_basis.csv exist in transactions.csv",
                          len(bad_cb_txn) == 0, f"{len(bad_cb_txn)} orphaned" if bad_cb_txn else ""))

    bad_wh_txn = set(withholding_df["transaction_id"]) - valid_txns
    ri_checks.append(_pf("All transaction_id in withholding.csv exist in transactions.csv",
                          len(bad_wh_txn) == 0, f"{len(bad_wh_txn)} orphaned" if bad_wh_txn else ""))

    for name, df, col in [("accounts.csv", accounts_df, "account_id"),
                           ("transactions.csv", transactions_df, "transaction_id"),
                           ("cost_basis.csv", cost_basis_df, "basis_record_id"),
                           ("withholding.csv", withholding_df, "withholding_id"),
                           ("tax_forms.csv", tax_forms_df, "tax_form_id")]:
        dup_n = int(df[col].duplicated().sum())
        ri_checks.append(_pf(f"No duplicate {col} in {name}", dup_n == 0, f"{dup_n} duplicates" if dup_n else ""))

    cb_dates = cost_basis_df.copy()
    cb_dates["acquisition_date"] = pd.to_datetime(cb_dates["acquisition_date"])
    cb_dates["sale_date"] = pd.to_datetime(cb_dates["sale_date"])
    bad_date_n = int((cb_dates["acquisition_date"] > cb_dates["sale_date"]).sum())
    ri_checks.append(_pf("No cost_basis acquisition_date after sale_date", bad_date_n == 0,
                          f"{bad_date_n} rows" if bad_date_n else ""))

    results["referential_integrity"] = all(ri_checks)

    # =======================================================================
    # 8. AUTOMATED ACCOUNT SPOT CHECK
    # =======================================================================
    print("\n--- 8. AUTOMATED ACCOUNT SPOT CHECK (5 random accounts, fixed seed) ---")
    spot_rng = np.random.RandomState(SEED)  # independent, reproducible draw
    sample_ids = spot_rng.choice(accounts_df["account_id"].values, size=5, replace=False)

    for acct_id in sample_ids:
        acct = accounts_df.loc[accounts_df.account_id == acct_id].iloc[0]
        acct_txns = transactions_df[transactions_df.account_id == acct_id]
        acct_cb = cost_basis_df[cost_basis_df.account_id == acct_id]
        acct_wh = withholding_df[withholding_df.account_id == acct_id]
        acct_forms = tax_forms_df[tax_forms_df.account_id == acct_id]

        n_buy = int((acct_txns.transaction_type == "BUY").sum())
        n_sell = int((acct_txns.transaction_type == "SELL").sum())
        div_income = acct_txns.loc[acct_txns.transaction_type == "DIVIDEND", "gross_amount"].sum()
        int_income = acct_txns.loc[acct_txns.transaction_type == "INTEREST", "gross_amount"].sum()
        proceeds = acct_cb["proceeds"].sum()
        cost_basis_total = acct_cb["cost_basis"].sum()
        gain_loss = acct_cb["adjusted_gain_loss"].sum()
        wh_total = acct_wh["withholding_amount"].sum()
        forms_issued = ", ".join(sorted(acct_forms["form_type"].unique())) if len(acct_forms) else "none"
        doc_status = f"{acct.tax_document_type} (W9={acct.w9_status}, W8={acct.w8_status})"

        print(f"\n  {acct_id}  |  {acct.account_type}  |  {acct.tax_residency}  |  docs: {doc_status}")
        print(f"    BUY txns: {n_buy}   SELL txns: {n_sell}")
        print(f"    Dividend income: ${div_income:,.2f}   Interest income: ${int_income:,.2f}")
        print(f"    Total sale proceeds: ${proceeds:,.2f}   Total cost basis: ${cost_basis_total:,.2f}")
        print(f"    Realized gain/loss: ${gain_loss:,.2f}")
        print(f"    Withholding: ${wh_total:,.2f}")
        print(f"    Tax forms issued: {forms_issued}")

    results["account_spot_checks"] = True  # informational section, always "complete"

    # =======================================================================
    # 9. FINAL QA SUMMARY
    # =======================================================================
    print("\n" + "=" * 78)
    print("PHASE 1 QA RESULTS")
    print("=" * 78)

    def status_str(key, review_word="PASS"):
        return "PASS" if results[key] else ("REVIEW" if key == "gain_loss_distribution" else "FAIL")

    print(f"Referential integrity:      {status_str('referential_integrity')}")
    print(f"Cost basis reconciliation:  {'PASS' if results['cost_basis_proceeds_tie_out'] and results['cost_basis_math'] else 'FAIL'}")
    print(f"Tax form reconciliation:    {status_str('tax_form_reconciliation')}")
    print(f"Withholding reconciliation: {status_str('withholding_reconciliation')}")
    print(f"Documentation logic:        {status_str('documentation_logic')}")
    print(f"Gain/loss distribution:     {'PASS' if results['gain_loss_distribution'] else 'REVIEW'}")
    print(f"Account spot checks:        COMPLETE")

    hard_fail_keys = ["referential_integrity", "cost_basis_proceeds_tie_out", "cost_basis_math",
                       "tax_form_reconciliation", "withholding_reconciliation", "documentation_logic"]
    overall_ready = all(results[k] for k in hard_fail_keys)

    print("\nOVERALL RESULT:")
    if overall_ready:
        print("READY FOR PHASE 2")
        if not results["gain_loss_distribution"]:
            print("(Note: gain/loss distribution flagged for REVIEW - not a blocking failure,")
            print(" but worth a look before building exception tests on top of it.)")
    else:
        print("REVIEW REQUIRED BEFORE PHASE 2")
        print("\nFailing checks:")
        for k in hard_fail_keys:
            if not results[k]:
                print(f"  - {k}")
    print("=" * 78)

    return overall_ready


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating accounts.csv ...")
    accounts_df = generate_accounts()

    print("Generating transactions.csv and cost_basis.csv ...")
    transactions_df, cost_basis_df = generate_transactions_and_costbasis(accounts_df)

    print("Generating withholding.csv ...")
    withholding_df = generate_withholding(accounts_df, transactions_df)

    print("Generating tax_forms.csv ...")
    tax_forms_df = generate_tax_forms(accounts_df, transactions_df, cost_basis_df, withholding_df)

    # -----------------------------------------------------------------
    # Row counts
    # -----------------------------------------------------------------
    print("\n--- ROW COUNTS ---")
    print(f"accounts.csv:      {len(accounts_df):,} rows")
    print(f"transactions.csv:  {len(transactions_df):,} rows")
    print(f"cost_basis.csv:    {len(cost_basis_df):,} rows")
    print(f"tax_forms.csv:     {len(tax_forms_df):,} rows")
    print(f"withholding.csv:   {len(withholding_df):,} rows")

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    print("\n--- VALIDATION ---")
    validate(accounts_df, transactions_df, cost_basis_df, tax_forms_df, withholding_df)

    # -----------------------------------------------------------------
    # Reconciliation summary
    # -----------------------------------------------------------------
    print("\n--- RECONCILIATION SUMMARY ---")
    total_buy = transactions_df.loc[transactions_df.transaction_type == "BUY", "gross_amount"].sum()
    total_sell = transactions_df.loc[transactions_df.transaction_type == "SELL", "gross_amount"].sum()
    total_div = transactions_df.loc[transactions_df.transaction_type == "DIVIDEND", "gross_amount"].sum()
    total_int = transactions_df.loc[transactions_df.transaction_type == "INTEREST", "gross_amount"].sum()
    total_proceeds_cb = cost_basis_df["proceeds"].sum()
    total_gain = cost_basis_df["adjusted_gain_loss"].sum()
    total_wh = withholding_df["withholding_amount"].sum()
    total_fed_withheld_forms = tax_forms_df["federal_tax_withheld"].sum()

    print(f"Total BUY volume:                 ${total_buy:,.2f}")
    print(f"Total SELL volume (transactions):  ${total_sell:,.2f}")
    print(f"Total proceeds (cost_basis.csv):   ${total_proceeds_cb:,.2f}  (should ~= SELL volume above)")
    print(f"Total realized gain/loss (adj):    ${total_gain:,.2f}")
    print(f"Total dividend income:             ${total_div:,.2f}")
    print(f"Total interest income:             ${total_int:,.2f}")
    print(f"Total withholding (detail):        ${total_wh:,.2f}")
    print(f"Total federal_tax_withheld (forms):${total_fed_withheld_forms:,.2f}  (subset of detail withholding)")
    print(f"Accounts with reportable activity: {cost_basis_df['account_id'].nunique() + transactions_df.loc[transactions_df.transaction_type=='DIVIDEND','account_id'].nunique()} (union not deduplicated in this line, see tax_forms.csv for per-account forms)")

    # -----------------------------------------------------------------
    # Phase 1 automated QA report (does not modify any data)
    # -----------------------------------------------------------------
    qa_ready = run_qa_report(accounts_df, transactions_df, cost_basis_df, tax_forms_df, withholding_df)

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------
    accounts_df.to_csv(os.path.join(OUTPUT_DIR, "accounts.csv"), index=False)
    transactions_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
    cost_basis_df.to_csv(os.path.join(OUTPUT_DIR, "cost_basis.csv"), index=False)
    tax_forms_df.to_csv(os.path.join(OUTPUT_DIR, "tax_forms.csv"), index=False)
    withholding_df.to_csv(os.path.join(OUTPUT_DIR, "withholding.csv"), index=False)

    print(f"\nAll five CSV files written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
