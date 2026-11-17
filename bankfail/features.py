import numpy as np
import pandas as pd

from bankfail.quarters import index_from_ymd

CAP = 1000.0

CAPITAL = ("equity_ratio", "tangible_equity_ratio", "leverage_ratio", "total_capital_ratio")
ASSET_QUALITY = (
    "noncurrent_loans",
    "past_due_30_89",
    "nonperforming_assets",
    "oreo_ratio",
    "texas_ratio",
    "net_chargeoffs",
    "reserve_ratio",
    "provision_ratio",
)
EARNINGS = ("roa", "nim", "efficiency", "noninterest_income", "asset_yield")
CONCENTRATION = (
    "cre_to_capital",
    "construction_to_capital",
    "construction_share",
    "residential_share",
    "business_share",
    "consumer_share",
    "farm_share",
    "loans_to_assets",
)
GROWTH = ("asset_growth", "loan_growth", "noncurrent_change", "equity_change", "roa_change")
STRUCTURE = ("log_assets", "bank_age", "young_bank", "subchapter_s", "holding_company", "offices")
FUNDING = (
    "funding_cost",
    "loans_to_deposits",
    "core_deposits",
    "brokered_deposits",
    "uninsured_deposits",
    "cash_ratio",
    "securities_ratio",
    "borrowings",
    "volatile_liabilities",
    "long_term_assets",
    "deposit_growth",
    "unrealized_loss_to_equity",
)
LOCAL = ("state_failure_rate", "state_noncurrent")
CATEGORICAL = ("charter",)

CREDIT_FEATURES = CAPITAL + ASSET_QUALITY + EARNINGS + CONCENTRATION + GROWTH + STRUCTURE
ALL_FEATURES = CREDIT_FEATURES + FUNDING + LOCAL + CATEGORICAL
LOGIT_FEATURES = (
    "equity_ratio",
    "noncurrent_loans",
    "oreo_ratio",
    "roa",
    "securities_ratio",
    "core_deposits",
    "brokered_deposits",
    "construction_share",
    "loan_growth",
    "log_assets",
    "young_bank",
)
RUN_SCREEN = ("uninsured_deposits", "unrealized_loss_to_equity")

INCREASING = {
    "noncurrent_loans",
    "past_due_30_89",
    "nonperforming_assets",
    "oreo_ratio",
    "texas_ratio",
    "net_chargeoffs",
    "provision_ratio",
    "efficiency",
    "cre_to_capital",
    "construction_to_capital",
    "construction_share",
    "noncurrent_change",
    "funding_cost",
    "brokered_deposits",
    "state_failure_rate",
    "state_noncurrent",
    "unrealized_loss_to_equity",
    "path_unemployment_rise",
}
DECREASING = {
    "equity_ratio",
    "tangible_equity_ratio",
    "leverage_ratio",
    "total_capital_ratio",
    "roa",
    "equity_change",
    "roa_change",
    "core_deposits",
    "path_house_price_change",
    "path_cre_price_change",
}

CHARTERS = ("N", "SM", "NM", "SB", "SA", "SI", "OI")


def ratio(numerator, denominator, scale: float = 100.0) -> np.ndarray:
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den * scale, np.nan)
    return out


def stressed_ratio(numerator, denominator, scale: float = 100.0) -> np.ndarray:
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    out = ratio(num, den, scale)
    broke = (den <= 0) & np.isfinite(num)
    return np.where(broke, CAP, np.minimum(out, CAP))


def lagged(frame: pd.DataFrame, values: np.ndarray, quarters: int) -> np.ndarray:
    keys = frame["CERT"].to_numpy(dtype="int64") * 100000 + frame["q"].to_numpy(dtype="int64")
    positions = pd.Index(keys).get_indexer(keys - quarters)
    out = np.asarray(values, dtype="float64")[positions]
    return np.where(positions >= 0, out, np.nan)


def growth(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(previous > 0, (current / previous - 1.0) * 100.0, np.nan)
    return np.clip(out, -100.0, CAP)


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy(dtype="float64", na_value=np.nan)


def bank_features(frame: pd.DataFrame) -> pd.DataFrame:
    c = {name: _col(frame, name) for name in frame.columns if frame[name].dtype.kind == "f"}
    zero = np.nan_to_num
    assets = c["ASSET"]
    loans = c["LNLS"]
    equity = c["EQ"]
    tangible = equity - zero(c["INTAN"])
    bad = zero(c["NCLNLS"]) + zero(c["ORE"])
    cre = zero(c["LNRECONS"]) + zero(c["LNRENRES"]) + zero(c["LNREMULT"])
    loss = (c["SCHA"] - c["SCHF"]) + (c["SCAA"] - c["SCAF"])
    out = pd.DataFrame(index=frame.index)
    out["equity_ratio"] = ratio(equity, assets)
    out["tangible_equity_ratio"] = ratio(tangible, assets)
    out["leverage_ratio"] = c["RBC1AAJ"]
    out["total_capital_ratio"] = c["RBCRWAJ"]
    out["noncurrent_loans"] = ratio(c["NCLNLS"], loans)
    out["past_due_30_89"] = ratio(c["P3ASSET"], assets)
    out["nonperforming_assets"] = ratio(bad, assets)
    out["oreo_ratio"] = ratio(c["ORE"], assets)
    out["texas_ratio"] = stressed_ratio(bad, tangible + zero(c["LNATRES"]))
    out["net_chargeoffs"] = c["NTLNLSR"]
    out["reserve_ratio"] = c["LNATRESR"]
    out["provision_ratio"] = c["ELNATRY"]
    out["roa"] = c["ROA"]
    out["nim"] = c["NIMY"]
    out["efficiency"] = c["EEFFR"]
    out["noninterest_income"] = c["NONIIAY"]
    out["asset_yield"] = c["INTINCY"]
    out["cre_to_capital"] = stressed_ratio(cre, equity)
    out["construction_to_capital"] = stressed_ratio(zero(c["LNRECONS"]), equity)
    out["construction_share"] = ratio(c["LNRECONS"], loans)
    out["residential_share"] = ratio(c["LNRERES"], loans)
    out["business_share"] = ratio(c["LNCI"], loans)
    out["consumer_share"] = ratio(c["LNCON"], loans)
    out["farm_share"] = ratio(zero(c["LNAG"]) + zero(c["LNREAG"]), loans)
    out["loans_to_assets"] = ratio(c["LNLSNET"], assets)
    out["funding_cost"] = c["INTEXPY"]
    out["loans_to_deposits"] = c["LNLSDEPR"]
    out["core_deposits"] = ratio(c["COREDEP"], assets)
    out["brokered_deposits"] = ratio(zero(c["BRO"]), c["DEP"])
    out["uninsured_deposits"] = np.clip(ratio(c["DEPUNINS"], c["DEPDOM"]), 0.0, 100.0)
    out["cash_ratio"] = ratio(c["CHBAL"], assets)
    out["securities_ratio"] = ratio(c["SC"], assets)
    out["borrowings"] = ratio(c["OTHBOR"], assets)
    out["volatile_liabilities"] = ratio(c["VOLIAB"], assets)
    out["long_term_assets"] = ratio(c["ASSTLT"], assets)
    out["unrealized_loss_to_equity"] = np.clip(ratio(loss, equity), -CAP, CAP)
    out["asset_growth"] = growth(assets, lagged(frame, assets, 4))
    out["loan_growth"] = growth(loans, lagged(frame, loans, 4))
    out["deposit_growth"] = growth(c["DEP"], lagged(frame, c["DEP"], 4))
    out["noncurrent_change"] = out["noncurrent_loans"] - lagged(frame, out["noncurrent_loans"], 4)
    out["equity_change"] = out["equity_ratio"] - lagged(frame, out["equity_ratio"], 4)
    out["roa_change"] = out["roa"] - lagged(frame, out["roa"], 4)
    out["log_assets"] = np.log(np.where(assets > 0, assets, np.nan))
    founded = index_from_ymd(frame["ESTYMD"].astype("string").fillna(""))
    age = (frame["q"].to_numpy(dtype="float64") - founded) / 4.0
    out["bank_age"] = np.where(age >= 0, age, np.nan)
    out["young_bank"] = np.where(np.isnan(out["bank_age"]), np.nan, (out["bank_age"] < 5) * 1.0)
    out["subchapter_s"] = np.nan_to_num(c["SUBCHAPS"]).clip(0, 1)
    out["holding_company"] = (np.nan_to_num(c["RSSDHCR"]) > 0) * 1.0
    out["offices"] = np.log1p(np.nan_to_num(c["OFFDOM"]).clip(0, None))
    charter = frame["BKCLASS"].astype("string").fillna("NA")
    out["charter"] = pd.Categorical(
        charter.where(charter.isin(CHARTERS), "NA"), categories=CHARTERS + ("NA",)
    ).codes.astype("float64")
    return out.astype({name: "float32" for name in out.columns})


def local_features(
    frame: pd.DataFrame, failures: pd.DataFrame, noncurrent: pd.Series
) -> pd.DataFrame:
    state = frame["STALP"].astype("string").fillna("NA")
    q = frame["q"].to_numpy(dtype="int64")
    banks = pd.DataFrame({"state": state.to_numpy(), "q": q})
    counts = banks.groupby(["state", "q"]).size()
    failed_state = failures["CERT"].map(frame.groupby("CERT", observed=True)["STALP"].last())
    events = pd.DataFrame({"state": failed_state.astype("string"), "fail_q": failures["fail_q"]})
    events = events.dropna()
    per_quarter = events.groupby(["state", "fail_q"]).size()
    table = counts.to_frame("banks").reset_index()
    table["failed"] = [
        sum(per_quarter.get((s, v - k), 0) for k in range(4))
        for s, v in zip(table["state"], table["q"], strict=True)
    ]
    table["rate"] = table["failed"] / table["banks"] * 100.0
    medians = pd.DataFrame({"state": state.to_numpy(), "q": q, "nc": noncurrent.to_numpy()})
    medians = medians.groupby(["state", "q"])["nc"].median().rename("median").reset_index()
    joined = banks.merge(table[["state", "q", "rate"]], on=["state", "q"], how="left")
    joined = joined.merge(medians, on=["state", "q"], how="left")
    return pd.DataFrame(
        {
            "state_failure_rate": joined["rate"].to_numpy(dtype="float32"),
            "state_noncurrent": joined["median"].to_numpy(dtype="float32"),
        },
        index=frame.index,
    )


def build_features(labeled: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    bank = bank_features(labeled)
    local = local_features(labeled, failures, bank["noncurrent_loans"])
    keys = labeled[
        ["CERT", "q", "NAME", "STALP", "ASSET", "event", "quarters_to_event", "quarters_to_other"]
    ]
    return pd.concat([keys, bank, local], axis=1)
