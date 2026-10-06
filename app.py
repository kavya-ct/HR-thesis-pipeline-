"""
HR Data Quality Cleaning Pipeline
Master's Thesis — Kavya Chiganarapplara Thippeswamy
FH St. Pölten — MSc Digital Innovation and Research

Final production version.
Works safely on any HR CSV dataset without requiring code changes.
"""

import re
import warnings
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HR Data Quality Pipeline",
    page_icon="📊",
    layout="wide"
)

# ================================================================
# HELPER — CURRENCY-AWARE NUMBER PARSING
# Handles: $50,000  €41,185  £32,500  ¥500,000  ₹45,000
#          USD 78,000  4,715,450 INR  55,659,714 NGN
#          61,833.17  50000
# Does NOT handle: European decimal notation (1.234,56)
# ================================================================
_CURRENCY_SYMBOLS = re.compile(
    r'[$€£¥₹₦₩₫₺₴₸₽₾₼₻₺₹₸₷₶₵₴₳₲₱₰₯₮₭€₫€₪₩₨₧₦₥₤₣₢₡₠]'
)
_CURRENCY_CODE_PREFIX = re.compile(r'^\s*[A-Z]{2,3}\s+')   # "USD 50000"
_CURRENCY_CODE_SUFFIX = re.compile(r'\s+[A-Z]{2,3}\s*$')   # "50000 INR"
_THOUSANDS_COMMA      = re.compile(r',(?=\d{3}(\.|,|$))')  # "50,000" but not "50,00" (EU)


def _strip_currency(val: str) -> str:
    """
    Remove currency symbols and codes from a string value so
    pd.to_numeric can parse it.
    Example: "€41,185 " -> "41185"
             "4,715,450 INR" -> "4715450"
             "USD 78,000"    -> "78000"
             "61,833.17"     -> "61833.17"
    """
    s = str(val).strip()
    s = _CURRENCY_SYMBOLS.sub("", s)
    s = _CURRENCY_CODE_PREFIX.sub("", s)
    s = _CURRENCY_CODE_SUFFIX.sub("", s)
    s = _THOUSANDS_COMMA.sub("", s)
    return s.strip()


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    """
    Try to parse a text series to numeric with currency awareness.
    Returns numeric Series; values that cannot be parsed become NaN.
    """
    return series.astype(str).apply(_strip_currency).pipe(
        lambda s: pd.to_numeric(s, errors="coerce")
    )


def _nan_loss_is_safe(original: pd.Series, converted: pd.Series,
                      threshold: float = 0.05) -> bool:
    """
    Returns True if converting to numeric is safe.
    Safe means: the number of NEW NaN values created is at most
    `threshold` fraction of the originally populated values.

    Example: 96,900 populated, 5 fail → 0.005% loss → safe
             96,900 populated, 82,000 fail → 84% loss → NOT safe
    """
    originally_populated = original.notna().sum()
    if originally_populated == 0:
        return True
    new_nans = converted.isna().sum() - original.isna().sum()
    new_nans = max(new_nans, 0)
    loss_rate = new_nans / originally_populated
    return loss_rate <= threshold


# ================================================================
# HELPER — PROTECTED COLUMNS (skip Title Case and type conversion)
# ================================================================
def _is_protected_column(col_name: str) -> bool:
    """
    Returns True for columns where automatic transformation would
    likely corrupt data.  These columns are handled by specific
    rules or left as-is.

    Protected categories:
      - Email / URL / hash / token columns
      - ID / code / reference columns
      - Name columns (person names need case preservation)
      - Department / job role columns (handled by Rule 19)
      - Country / nationality columns (USA must stay USA)
    """
    name = col_name.lower()

    # Email, URL, identifiers, keys
    if any(w in name for w in [
        "email", "mail", "url", "link",
        "id", "code", "ref", "key", "hash", "token"
    ]):
        return True

    # Department / job columns — Rule 19 uses smart_title()
    if any(w in name for w in [
        "department", "dept", "division",
        "jobrole", "job_role", "jobtitle", "job_title",
        "position", "role"
    ]):
        return True

    # Person name columns
    if any(w in name for w in [
        "name", "firstname", "lastname",
        "first_name", "last_name", "full_name",
        "surname", "givenname", "given_name"
    ]):
        return True

    # Country / nationality — abbreviations like USA must stay uppercase
    if any(w in name for w in [
        "country", "nationality", "citizenship", "nation", "region"
    ]):
        return True

    # Salary / compensation columns — when currency formats like €, £, INR
    # are preserved as text (Rule 11 safety guard), they must not be
    # title-cased by Rule 7 (e.g. "4,715,450 INR" → "4,715,450 Inr" is wrong)
    if any(w in name for w in [
        "salary", "income", "pay", "wage", "compensation",
        "remuneration", "ctc", "package"
    ]):
        return True

    return False


# ================================================================
# MAIN CLEANING FUNCTION
# ================================================================
def clean_hr_data(df: pd.DataFrame):
    """
    Apply 20 sequential cleaning rules to any HR CSV dataset.

    Design principles:
    1. Never destroy valid data silently.
    2. When uncertain, preserve original values and warn.
    3. No dataset-specific logic — works on any HR CSV.
    4. Every change is recorded in the fix log.

    Returns: (cleaned_dataframe, fix_log)
    """
    fixes = []

    # Record baseline NaN counts BEFORE any rule runs
    # Used by Rule 6 to detect upstream data destruction
    baseline_nulls = df.isnull().sum().to_dict()

    # ── RULE 1 — Standardise column names ──────────────────────
    # Why: Different HR systems export columns as "Employee Age",
    #      "GENDER", "emp-id". Standardisation enables consistent
    #      downstream detection by all other rules.
    # Safe: Renaming never changes data values.
    original_col_names = list(df.columns)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ",  "_", regex=False)
        .str.replace("-",  "_", regex=False)
        .str.replace("(",  "",  regex=False)
        .str.replace(")",  "",  regex=False)
        .str.replace("/",  "_", regex=False)
        .str.replace(".",  "_", regex=False)
    )
    changed = sum(1 for o, n in zip(original_col_names, df.columns) if o != n)
    fixes.append({
        "n": 1,
        "name": "Column Names Standardised",
        "detail": f"{changed} column names converted to lowercase_underscore format",
        "count": changed,
    })

    # ── RULE 2 — Remove useless columns ────────────────────────
    # Why: Columns where every row has the same value carry zero
    #      analytical information (e.g. EmployeeCount always = 1).
    # Safe: Constant columns have no information to lose.
    useless = [c for c in df.columns if df[c].nunique() <= 1]
    if useless:
        df = df.drop(columns=useless)
    fixes.append({
        "n": 2,
        "name": "Useless Columns Removed",
        "detail": (f"Removed {len(useless)} constant-value columns: {useless}"
                   if useless else "No constant-value columns found"),
        "count": len(useless),
    })

    # ── RULE 3 — Remove exact duplicate rows ───────────────────
    # Why: Identical rows indicate data merging or export errors.
    # Safe: Only 100% identical rows are removed.
    dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()
    fixes.append({
        "n": 3,
        "name": "Duplicate Rows Removed",
        "detail": (f"Removed {dupes} duplicate rows"
                   if dupes > 0 else "No duplicate rows found"),
        "count": dupes,
    })

    # ── RULE 4 — Convert whitespace-only cells to NaN ──────────
    # Why: A cell containing "   " looks empty to a human but is
    #      not NaN to Python — it breaks count and fill operations.
    # Safe: Only cells whose stripped content is empty are touched.
    ws_fixed = 0
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].astype(str).str.strip() == ""
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
            ws_fixed += int(mask.sum())
    fixes.append({
        "n": 4,
        "name": "Whitespace-Only Cells Converted to Empty",
        "detail": (f"{ws_fixed} cells containing only spaces treated as missing"
                   if ws_fixed > 0 else "No whitespace-only cells found"),
        "count": ws_fixed,
    })

    # ── RULE 5 — Fix wrong data types ──────────────────────────
    # Why: Salary stored as "5,000" (text) cannot be used in
    #      calculations. We convert text columns to numbers ONLY
    #      when safe to do so.
    # Safe: Uses currency-aware stripping + NaN loss guard.
    #       Only converts if >80% of values parse AND the conversion
    #       does not destroy more than 5% of populated values.
    #       Protected columns (email, ID, name, dept) are skipped.
    type_fixed = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if _is_protected_column(col):
            continue
        converted = _parse_numeric_series(df[col])
        parse_rate = converted.notna().sum() / max(len(df), 1)
        if parse_rate >= 0.8 and _nan_loss_is_safe(df[col], converted):
            df[col] = converted
            type_fixed.append(col)
    fixes.append({
        "n": 5,
        "name": "Data Types Fixed",
        "detail": (f"Converted to numbers: {type_fixed}"
                   if type_fixed else "All columns already have correct data types"),
        "count": len(type_fixed),
    })

    # ── RULE 6 — Fill missing values ───────────────────────────
    # Why: Blank cells break aggregations, charts, and joins.
    # Strategy: numbers → column median (preserves distribution
    #           better than mean); text → "Unknown".
    # Safety note: Records if a column has MORE NaN now than in
    #              the original file — this indicates an upstream
    #              rule may have created NaN from valid values.
    total_missing = int(df.isnull().sum().sum())
    fill_details = []
    warnings_list = []

    for col in df.columns:
        current_nulls = int(df[col].isnull().sum())
        if current_nulls == 0:
            continue

        original_nulls = baseline_nulls.get(col, 0)
        new_nulls_created = max(current_nulls - original_nulls, 0)

        if new_nulls_created > 0:
            warnings_list.append(
                f"⚠️ '{col}': had {original_nulls} missing originally, "
                f"now has {current_nulls} — {new_nulls_created} values "
                f"may have been created by a cleaning step that could not "
                f"parse certain formats (e.g. currency symbols like €, £, INR)"
            )

        if df[col].dtype in ["int64", "float64"]:
            fill_val = round(float(df[col].median()), 2)
            df[col] = df[col].fillna(fill_val)
            fill_details.append(f"{col}: {current_nulls} filled with median ({fill_val})")
        else:
            df[col] = df[col].fillna("Unknown")
            fill_details.append(f"{col}: {current_nulls} filled with 'Unknown'")

    detail_str = " | ".join(fill_details) if fill_details else "No missing values found"
    if warnings_list:
        detail_str += " || WARNINGS: " + " | ".join(warnings_list)

    fixes.append({
        "n": 6,
        "name": "Missing Values Fixed",
        "detail": detail_str,
        "count": total_missing,
    })

    # ── RULE 7 — Fix text formatting ───────────────────────────
    # Why: "  SALES  " and "sales" and "Sales" are the same value
    #      but different strings. Title Case + strip normalises them.
    # Safe: Protected columns (email, ID, name, dept, country) are
    #       only stripped of whitespace — not title-cased.
    tc_fixed = 0
    for col in df.select_dtypes(include="object").columns:
        if _is_protected_column(col):
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = df[col].astype(str).str.strip().str.title()
            tc_fixed += 1
    fixes.append({
        "n": 7,
        "name": "Text Formatting Cleaned",
        "detail": f"Title Case applied to {tc_fixed} text columns; "
                  f"protected columns (email, ID, name, country) only stripped",
        "count": tc_fixed,
    })

    # ── RULE 8 — Standardise Yes/No values ─────────────────────
    # Why: Yes/No data appears as True/False, y/n, 1/0, YES/NO
    #      across different HR systems.
    # Three passes: string columns, bool dtype, integer 0/1 columns.
    # Safe: Only converts columns whose entire value set is binary.
    yn_map = {
        "yes": "Yes", "no": "No",
        "y":   "Yes", "n":  "No",
        "1":   "Yes", "0":  "No",
        "true": "Yes", "false": "No",
    }
    yn_skip = [
        "salary", "income", "pay", "wage", "age", "year",
        "rate", "hours", "score", "level", "count",
        "number", "percent", "salary",
    ]
    yn_fixed = 0

    # Pass 1 — string columns
    for col in df.select_dtypes(include="object").columns:
        vals = set(df[col].astype(str).str.lower().str.strip().unique())
        vals.discard("nan")
        if vals and vals.issubset(set(yn_map.keys())):
            df[col] = (df[col].astype(str).str.lower().str.strip()
                       .map(yn_map).fillna("Unknown"))
            yn_fixed += 1

    # Pass 2 — boolean dtype columns
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].map({True: "Yes", False: "No"})
        yn_fixed += 1

    # Pass 3 — integer columns containing only {0, 1}
    for col in df.select_dtypes(include=["int64", "int32", "Int64"]).columns:
        if any(w in col.lower() for w in yn_skip):
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, np.int64(0), np.int64(1)}):
            df[col] = df[col].map(
                {0: "No", 1: "Yes",
                 np.int64(0): "No", np.int64(1): "Yes"}
            )
            yn_fixed += 1

    fixes.append({
        "n": 8,
        "name": "Yes/No Values Standardised",
        "detail": f"{yn_fixed} binary columns standardised to Yes/No format",
        "count": yn_fixed,
    })

    # ── RULE 9 — Standardise gender values ─────────────────────
    # Why: Gender appears as M/F, male/MALE, 1/2, Man/Woman.
    # Note: The mapping 0→Female, 1→Male follows common HR system
    #       conventions but is an assumption — different systems
    #       may use different encodings. Users should verify.
    gender_col = next(
        (c for c in df.columns if "gender" in c.lower() or "sex" in c.lower()),
        None
    )
    gender_fixed = 0
    if gender_col:
        gender_map = {
            "M": "Male",     "F": "Female",
            "m": "Male",     "f": "Female",
            "male": "Male",  "female": "Female",
            "MALE": "Male",  "FEMALE": "Female",
            "Male": "Male",  "Female": "Female",
            "man": "Male",   "woman": "Female",
            "Man": "Male",   "Woman": "Female",
            "1": "Male",     "2": "Female",
            "0": "Female",
        }
        df[gender_col] = df[gender_col].astype(str).str.strip().replace(gender_map)
        gender_fixed = 1
    fixes.append({
        "n": 9,
        "name": "Gender Values Standardised",
        "detail": (f"Column '{gender_col}' standardised to Male/Female "
                   f"(mapping assumption: 1=Male, 2=Female, 0=Female)"
                   if gender_fixed else "No gender column detected"),
        "count": gender_fixed,
    })

    # ── RULE 10 — Validate age values ──────────────────────────
    # Why: Ages outside 16–80 indicate data entry errors for a
    #      working population.
    # Safe:
    #   1. Uses word-boundary regex to avoid matching "managername"
    #      (which contains "age" as a substring).
    #   2. Strips common age suffixes ("28 years" → "28") before
    #      conversion so valid values are not destroyed.
    #   3. Detects birth years (median > 1900) and skips rule
    #      to avoid replacing all birth years with median age.
    #   4. NaN loss guard: if conversion creates too many NaN,
    #      skips the rule and warns the user.
    age_col = next(
        (c for c in df.columns if re.search(r'(^|_)age($|_)', c.lower())),
        None
    )
    age_fixed = 0
    if age_col:
        try:
            # Strip common age suffixes first
            cleaned_age = (df[age_col].astype(str)
                           .apply(lambda v: re.sub(
                               r'\s*(years?|yrs?|y/?o|yo)\s*$', '',
                               v, flags=re.IGNORECASE).strip()))
            converted_age = pd.to_numeric(cleaned_age, errors="coerce")

            # Skip if this looks like birth years (median > 1900)
            median_val = converted_age.median()
            if pd.notna(median_val) and median_val > 1900:
                fixes.append({
                    "n": 10,
                    "name": "Age Validation Skipped",
                    "detail": (f"Column '{age_col}' appears to contain birth years "
                               f"(median={median_val:.0f}) — skipped to preserve data"),
                    "count": 0,
                })
            elif _nan_loss_is_safe(df[age_col], converted_age):
                df[age_col] = converted_age
                invalid = int(
                    df[(df[age_col] < 16) | (df[age_col] > 80)].shape[0]
                )
                if invalid > 0:
                    df.loc[
                        (df[age_col] < 16) | (df[age_col] > 80),
                        age_col
                    ] = df[age_col].median()
                    age_fixed = invalid
                fixes.append({
                    "n": 10,
                    "name": "Invalid Ages Fixed",
                    "detail": (f"{age_fixed} ages outside 16–80 replaced with median"
                               if age_fixed > 0 else "All ages within valid range (16–80)"),
                    "count": age_fixed,
                })
            else:
                fixes.append({
                    "n": 10,
                    "name": "Age Validation Skipped",
                    "detail": (f"Column '{age_col}' skipped — conversion would create "
                               f"too many missing values (possible mixed formats)"),
                    "count": 0,
                })
        except Exception:
            fixes.append({
                "n": 10,
                "name": "Age Validation Skipped",
                "detail": f"Column '{age_col}' could not be processed",
                "count": 0,
            })
    else:
        fixes.append({
            "n": 10,
            "name": "Age Validation",
            "detail": "No age column detected",
            "count": 0,
        })

    # ── RULE 11 — Validate salary values ───────────────────────
    # Why: Salary of zero or negative is not a valid employment record.
    # Safe:
    #   1. Uses currency-aware stripping before conversion so
    #      values like "€41,185" or "4,715,450 INR" are preserved.
    #   2. NaN loss guard: if conversion destroys more than 5% of
    #      populated values, skips the conversion and warns the user.
    #   3. Only deletes rows where salary is CONFIRMED <= 0 after
    #      successful parsing.
    sal_col = next(
        (c for c in df.columns
         if any(w in c.lower() for w in ["salary", "income", "pay", "wage"])),
        None
    )
    sal_fixed = 0
    if sal_col:
        try:
            converted_sal = _parse_numeric_series(df[sal_col])
            if _nan_loss_is_safe(df[sal_col], converted_sal):
                df[sal_col] = converted_sal
                invalid = int(df[df[sal_col] <= 0].shape[0])
                if invalid > 0:
                    df = df[df[sal_col] > 0]
                    sal_fixed = invalid
                fixes.append({
                    "n": 11,
                    "name": "Invalid Salary Removed",
                    "detail": (f"{sal_fixed} rows with zero or negative salary removed"
                               if sal_fixed > 0 else "All salary values are valid"),
                    "count": sal_fixed,
                })
            else:
                # Count how many NaN would be created
                new_nans = (converted_sal.isna().sum()
                            - df[sal_col].isna().sum())
                fixes.append({
                    "n": 11,
                    "name": "Salary Validation Skipped",
                    "detail": (f"⚠️ Column '{sal_col}' contains currency formats "
                               f"(e.g. €, £, INR, NGN) that could not all be parsed. "
                               f"Conversion skipped to preserve {new_nans:,} valid values. "
                               f"Please verify the salary column format."),
                    "count": 0,
                })
        except Exception:
            fixes.append({
                "n": 11,
                "name": "Salary Validation Skipped",
                "detail": f"Column '{sal_col}' could not be processed",
                "count": 0,
            })
    else:
        fixes.append({
            "n": 11,
            "name": "Salary Validation",
            "detail": "No salary column detected",
            "count": 0,
        })

    # ── RULE 12 — Fix negative numbers in non-negative columns ─
    # Why: Phone numbers stored as integers can overflow to negative.
    #      Age, year, hours cannot logically be negative.
    # Safe: Only applies to columns whose name matches specific
    #       HR-domain keywords. Rate columns excluded to avoid
    #       replacing legitimate negative rates (attrition_rate).
    neg_keywords = [
        "age", "hours", "count", "salary", "income",
        "phone", "mobile", "tel", "contact",
    ]
    neg_fixed = 0
    for col in df.select_dtypes(include="number").columns:
        col_lower = col.lower()
        if not any(w in col_lower for w in neg_keywords):
            continue
        neg_count = int((df[col] < 0).sum())
        if neg_count == 0:
            continue
        if any(w in col_lower for w in ["phone", "mobile", "tel", "contact"]):
            df[col] = df[col].abs()
        else:
            df.loc[df[col] < 0, col] = df[col].median()
        neg_fixed += neg_count
    fixes.append({
        "n": 12,
        "name": "Negative Values Fixed",
        "detail": (f"{neg_fixed} negative values fixed "
                   f"(phone: absolute value; others: replaced with median)"
                   if neg_fixed > 0 else "No invalid negative values found"),
        "count": neg_fixed,
    })

    # ── RULE 13 — Detect extreme outliers ──────────────────────
    # Why: Values far outside the normal range indicate possible
    #      data entry errors and should be flagged for review.
    # Method: IQR × 3 — conservative threshold to avoid false alarms.
    # Safe: Detection only — no values are changed.
    outlier_findings = []
    for col in df.select_dtypes(include="number").columns:
        try:
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                cnt = int(df[
                    (df[col] < Q1 - 3 * IQR) |
                    (df[col] > Q3 + 3 * IQR)
                ].shape[0])
                if cnt > 0:
                    outlier_findings.append(f"{col}: {cnt} extreme values")
        except Exception:
            pass
    fixes.append({
        "n": 13,
        "name": "Outliers Detected",
        "detail": (" | ".join(outlier_findings)
                   if outlier_findings else "No extreme outliers detected"),
        "count": len(outlier_findings),
    })

    # ── RULE 14 — Validate email addresses ─────────────────────
    # Why: Malformed emails fail in HR communications and payroll
    #      systems. Invalid values are flagged, not deleted.
    # Safe: Protected by _is_protected_column — email columns are
    #       never title-cased before this rule runs.
    _email_re = re.compile(
        r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    )
    email_col = next(
        (c for c in df.columns if "email" in c.lower() or "mail" in c.lower()),
        None
    )
    email_fixed = 0
    if email_col:
        mask = ~df[email_col].astype(str).apply(
            lambda v: bool(_email_re.match(v.strip()))
        )
        email_fixed = int(mask.sum())
        if email_fixed > 0:
            df.loc[mask, email_col] = "invalid_email"
    fixes.append({
        "n": 14,
        "name": "Email Addresses Validated",
        "detail": (f"{email_fixed} invalid email addresses flagged as 'invalid_email'"
                   if email_col else "No email column detected"),
        "count": email_fixed,
    })

    # ── RULE 15 — Clean phone numbers ──────────────────────────
    # Why: "(555) 123-4567" and "555-123-4567" are the same number
    #      stored differently.
    # Note: Applies only to text-format phone columns.
    #       Integer phone columns are handled by Rule 12 (negatives).
    phone_col = next(
        (c for c in df.columns
         if any(w in c.lower() for w in ["phone", "mobile", "contact", "tel"])),
        None
    )
    phone_fixed = 0
    if phone_col and df[phone_col].dtype == object:
        original_phones = df[phone_col].astype(str).copy()

        def _clean_phone(v: str) -> str:
            cleaned = re.sub(r'[\s\-\(\)\.\+]', '', str(v))
            # Keep if it is a valid phone length after cleaning
            if cleaned.lstrip("-").isdigit() and 7 <= len(cleaned.lstrip("-")) <= 15:
                return cleaned
            return str(v)  # return original if cleaning fails

        df[phone_col] = df[phone_col].apply(_clean_phone)
        phone_fixed = int((df[phone_col] != original_phones).sum())
    fixes.append({
        "n": 15,
        "name": "Phone Numbers Cleaned",
        "detail": (f"{phone_fixed} phone numbers standardised "
                   f"(removed spaces, dashes, and brackets)"
                   if phone_col else "No phone column detected"),
        "count": phone_fixed,
    })

    # ── RULE 16 — Remove special characters from name columns ──
    # Why: Names containing "!", "@", "#" are data entry errors.
    # Note: Apostrophes are NOT removed (O'Brien is valid).
    #       Only columns named first_name / last_name / full_name.
    special_fixed = 0
    for col in df.select_dtypes(include="object").columns:
        if any(w in col.lower() for w in
               ["first_name", "last_name", "full_name"]):
            original_vals = df[col].astype(str).copy()
            df[col] = (df[col].astype(str)
                       .str.replace(r"[^a-zA-Z\s\-\.']", "", regex=True)
                       .str.strip())
            special_fixed += int((df[col] != original_vals).sum())
    fixes.append({
        "n": 16,
        "name": "Special Characters Removed from Names",
        "detail": (f"{special_fixed} name field values cleaned"
                   if special_fixed > 0 else "No special characters found in name columns"),
        "count": special_fixed,
    })

    # ── RULE 17 — Detect duplicate employee IDs ────────────────
    # Why: Each employee should have a unique identifier.
    # Safe: Detection only — no values are changed.
    id_keywords = [
        "employeeid", "employee_id", "emp_id", "empid",
        "staffid", "staff_id", "worker_id", "personnel_id",
    ]
    id_col = next(
        (c for c in df.columns if any(w in c.lower() for w in id_keywords)),
        None
    )
    id_dupes = 0
    if id_col:
        id_dupes = int(df[id_col].duplicated().sum())
    fixes.append({
        "n": 17,
        "name": "Employee ID Validated",
        "detail": (f"{id_dupes} duplicate employee IDs detected — manual review recommended"
                   if id_col and id_dupes > 0
                   else "All employee IDs are unique"
                   if id_col else "No employee ID column detected"),
        "count": id_dupes,
    })

    # ── RULE 18 — Standardise date formats ─────────────────────
    # Why: "4/2/2021", "02-Apr-2021", "2021/04/02" are the same
    #      date but cannot be sorted or compared reliably.
    # Safe:
    #   1. Excludes "Unknown" placeholder values (added by Rule 6)
    #      from the success rate calculation.
    #   2. Only converts if ≥80% of real date values parse.
    #   3. "Unknown" values stay as "Unknown" after conversion.
    # Note: Uses dayfirst=False (US MM/DD/YYYY convention).
    #       For EU datasets where DD/MM/YYYY is standard,
    #       dates where day ≤ 12 may be parsed with swapped values.
    date_keywords = [
        "date", "dob", "birth", "hired", "joined",
        "start", "end", "termination", "joining",
    ]
    dates_converted = 0
    for col in df.columns:
        if not any(w in col.lower() for w in date_keywords):
            continue
        if col not in df.columns:
            continue
        if df[col].dtype in ["int64", "float64"]:
            continue
        try:
            # Exclude "Unknown" when measuring parse success
            real_vals = df[col][
                df[col].astype(str).str.strip().str.lower() != "unknown"
            ]
            if len(real_vals) == 0:
                continue
            converted_dates = pd.to_datetime(
                real_vals, errors="coerce", dayfirst=False
            )
            success = converted_dates.notna().sum() / max(len(real_vals), 1)
            if success >= 0.8:
                full_converted = pd.to_datetime(
                    df[col], errors="coerce", dayfirst=False
                )
                # Where conversion succeeded → YYYY-MM-DD
                # Where it failed (Unknown, unparseable) → keep original
                df[col] = full_converted.dt.strftime("%Y-%m-%d").where(
                    full_converted.notna(), other=df[col]
                )
                dates_converted += 1
        except Exception:
            pass
    fixes.append({
        "n": 18,
        "name": "Date Formats Standardised",
        "detail": (f"{dates_converted} date columns converted to YYYY-MM-DD format"
                   if dates_converted > 0 else "No date columns detected or converted"),
        "count": dates_converted,
    })

    # ── RULE 19 — Standardise department / job role names ──────
    # Why: "SALES", "sales", "Sales" are the same department.
    #      Standard str.title() corrupts DevOps→Devops, HR→Hr, IT→It.
    # Safe: Uses smart_title() which preserves mixed-case terms
    #       and keeps short all-caps abbreviations unchanged.
    def smart_title(val: str) -> str:
        """
        Capitalise text intelligently for department/role names.
        - all-lowercase word → Title Case  (finance → Finance)
        - ALL-CAPS ≤5 chars → keep as-is  (HR, IT, AI, BI)
        - ALL-CAPS >5 chars → Title Case  (FINANCE → Finance)
        - Mixed-case → preserve exactly   (DevOps, FinTech)
        - Handles hyphenated values       (DevOps-California)
        """
        val = str(val).strip()
        parts = val.split("-")
        result_parts = []
        for part in parts:
            words = part.split()
            result_words = []
            for w in words:
                if w.islower():
                    result_words.append(w.capitalize())
                elif w.isupper() and len(w) <= 5:
                    result_words.append(w)
                elif w.isupper() and len(w) > 5:
                    result_words.append(w.capitalize())
                else:
                    result_words.append(w)
            result_parts.append(" ".join(result_words))
        return "-".join(result_parts)

    dept_keywords = [
        "department", "dept", "division",
        "jobrole", "job_role", "jobtitle", "job_title",
        "position", "role",
    ]
    dept_fixed = 0
    for col in df.columns:
        if any(w in col.lower() for w in dept_keywords):
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).apply(smart_title)
                dept_fixed += 1
    fixes.append({
        "n": 19,
        "name": "Department and Job Role Names Standardised",
        "detail": (f"{dept_fixed} department/job role columns standardised "
                   f"(abbreviations like HR, IT, DevOps preserved)"
                   if dept_fixed > 0 else "No department or job role columns detected"),
        "count": dept_fixed,
    })

    # ── RULE 20 — Flag coded number columns ────────────────────
    # Why: Columns like Education=1,2,3,4,5 look numeric but are
    #      actually ordered categories. Using them as numbers in
    #      calculations produces meaningless results.
    # Safe: Detection only — no values are changed.
    # Note: Condition: 2–6 unique values, all ≥ 1.
    #       May produce false positives for quarter (1–4) or
    #       shift numbers. False negatives for 0-indexed codes.
    coded_cols = []
    for col in df.select_dtypes(include="number").columns:
        try:
            unique_vals = sorted(df[col].dropna().unique())
            if 2 <= len(unique_vals) <= 6 and float(min(unique_vals)) >= 1:
                coded_cols.append(col)
        except Exception:
            pass
    fixes.append({
        "n": 20,
        "name": "Coded Columns Flagged for Review",
        "detail": (f"Columns that may use numeric category codes "
                   f"(check data dictionary): {coded_cols}"
                   if coded_cols else "No coded numeric columns detected"),
        "count": len(coded_cols),
    })

    return df, fixes


# ================================================================
# QUALITY SCORING — 5 DAMA-DMBOK DIMENSIONS
# Score = Completeness(25) + Uniqueness(20) + Consistency(20)
#       + Validity(20) + Structure(15) = 100
# ================================================================
def calc_score(df: pd.DataFrame) -> float:
    """
    Calculate a Data Quality Index (DQI) score from 0 to 100.

    Based on five DAMA-DMBOK data quality dimensions,
    adapted for HR datasets.

    Limitations (documented in thesis):
    - Completeness counts imputed values as complete.
    - Consistency measures whitespace only, not semantic consistency.
    - Validity uses age range as the primary proxy.
    - Before/after scores may differ in dataset shape.
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0

    # Completeness (max 25)
    missing_pct = df.isnull().sum().sum() / total_cells * 100
    completeness = max(0.0, 25.0 - missing_pct * 0.25)

    # Uniqueness (max 20)
    dup_pct = df.duplicated().sum() / df.shape[0] * 100
    uniqueness = max(0.0, 20.0 - dup_pct * 0.5)

    # Consistency (max 20) — measures whitespace in text columns
    text_cols = df.select_dtypes(include="object").columns
    inconsistent = sum(
        1 for c in text_cols
        if (df[c].astype(str).str.strip() != df[c].astype(str)).sum() > 0
    )
    consistency = max(0.0, 20.0 - (inconsistent / max(len(text_cols), 1)) * 20)

    # Validity (max 20) — uses age range as primary proxy
    # Word-boundary regex prevents false matches (e.g. "managername")
    age_col = next(
        (c for c in df.columns if re.search(r'(^|_)age($|_)', c.lower())),
        None
    )
    if age_col:
        try:
            ages = pd.to_numeric(df[age_col], errors="coerce")
            invalid = int(((ages < 16) | (ages > 80)).sum())
            validity = max(0.0, 20.0 - (invalid / max(len(ages), 1)) * 100)
        except Exception:
            validity = 18.0
    else:
        validity = 18.0  # default when no age column

    # Structure (max 15) — column name cleanliness
    bad_cols = sum(
        1 for c in df.columns
        if c != c.strip() or " " in c or c != c.lower()
    )
    structure = max(0.0, 15.0 - (bad_cols / df.shape[1]) * 15)

    return round(completeness + uniqueness + consistency + validity + structure, 1)


# ================================================================
# STREAMLIT UI
# ================================================================
st.title("📊 HR Data Quality Cleaning Pipeline")
st.markdown(
    "### Automated HR Data Cleaning — Master's Thesis Tool\n"
    "Upload any HR CSV file. The pipeline applies **20 adaptive "
    "cleaning rules** and gives you a clean file to download."
)
st.divider()

# ── Step 1: Upload ──────────────────────────────────────────────
st.header("📁 Step 1 — Upload Your HR CSV File")
uploaded = st.file_uploader(
    "Choose any HR CSV file",
    type=["csv"],
    help="Upload any HR dataset in CSV format from any organisation.",
)

if uploaded is not None:
    try:
        df_orig = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.success(
        f"✅ Uploaded: **{uploaded.name}** — "
        f"{df_orig.shape[0]:,} rows, {df_orig.shape[1]} columns"
    )
    st.divider()

    # ── Step 2: Preview original ────────────────────────────────
    st.header("👀 Step 2 — Original Data Preview")
    st.write("This is how your data looks **before** cleaning.")
    st.dataframe(df_orig.head(10), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",          f"{df_orig.shape[0]:,}")
    c2.metric("Columns",       df_orig.shape[1])
    c3.metric("Missing Values", int(df_orig.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df_orig.duplicated().sum()))
    st.divider()

    # ── Step 3: Clean ───────────────────────────────────────────
    st.header("⚙️ Step 3 — Clean the Data")
    st.write("Click the button to run all 20 cleaning rules automatically.")

    if st.button("🚀 Clean My HR Data Now", type="primary"):
        with st.spinner("Applying 20 cleaning rules…"):
            score_before = calc_score(df_orig)
            df_clean, fixes = clean_hr_data(df_orig.copy())
            score_after  = calc_score(df_clean)
            improvement  = round(score_after - score_before, 1)

        st.success("✅ Cleaning complete!")
        st.divider()

        # ── Step 4: Quality scores ──────────────────────────────
        st.header("📈 Step 4 — Quality Score Before vs After")
        st.write(
            "Score is based on five DAMA-DMBOK dimensions: "
            "Completeness (25) + Uniqueness (20) + Consistency (20) "
            "+ Validity (20) + Structure (15) = 100"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Score Before", f"{score_before}/100")
        c2.metric("Score After",  f"{score_after}/100",
                  delta=f"+{improvement} points")
        c3.metric("Improvement",  f"+{improvement} points")

        if score_after >= 90:
            st.success("🏆 Final Grade: EXCELLENT")
        elif score_after >= 75:
            st.info("👍 Final Grade: GOOD")
        elif score_after >= 60:
            st.warning("⚠️ Final Grade: FAIR")
        else:
            st.error("❌ Final Grade: POOR")

        st.divider()

        # ── Step 5: Rules applied ───────────────────────────────
        st.header("🔧 Step 5 — All 20 Rules Applied")

        # Show warnings first (they are critical)
        warnings_present = [
            f for f in fixes
            if "⚠️" in f["detail"] or "WARNING" in f["detail"]
        ]
        if warnings_present:
            st.subheader("⚠️ Warnings — Please Review")
            for w in warnings_present:
                st.warning(f"Rule {w['n']}: {w['detail']}")
            st.divider()

        for fix in fixes:
            if fix["count"] > 0:
                st.success(
                    f"✅ Rule {fix['n']}: **{fix['name']}** — {fix['detail']}"
                )
            else:
                st.info(
                    f"☑️ Rule {fix['n']}: **{fix['name']}** — {fix['detail']}"
                )

        st.divider()

        # ── Step 6: Clean data preview ──────────────────────────
        st.header("✅ Step 6 — Clean Data Preview")
        st.write("This is how your data looks **after** cleaning.")
        st.dataframe(df_clean.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean Rows",     f"{df_clean.shape[0]:,}")
        c2.metric("Clean Columns",  df_clean.shape[1])
        c3.metric("Missing Values", int(df_clean.isnull().sum().sum()))
        c4.metric("Duplicate Rows", int(df_clean.duplicated().sum()))

        st.divider()

        # ── Step 7: Download ────────────────────────────────────
        st.header("⬇️ Step 7 — Download Clean File")
        clean_csv  = df_clean.to_csv(index=False).encode("utf-8")
        clean_name = uploaded.name.replace(".csv", "_CLEANED.csv")

        st.download_button(
            label=f"⬇️ Download {clean_name}",
            data=clean_csv,
            file_name=clean_name,
            mime="text/csv",
            type="primary",
        )
        st.caption(
            "HR Data Quality Pipeline — Master's Thesis | "
            "FH St. Pölten | 20 Adaptive Cleaning Rules | "
            "Tested on 32,801 employee records across 4 independent datasets"
        )

else:
    # ── Landing page when no file is uploaded ───────────────────
    st.info("👆 Upload a CSV file above to get started.")
    st.markdown("""
| Rule | What It Fixes |
|------|--------------|
| 1  | Messy column names → clean lowercase_underscore format |
| 2  | Useless constant columns → removed |
| 3  | Duplicate rows → removed |
| 4  | Whitespace-only cells → treated as empty |
| 5  | Numbers stored as text → converted (currency-aware, with safety check) |
| 6  | Missing values → filled with median or Unknown (with upstream warning) |
| 7  | Inconsistent casing → Title Case (email, ID, name, country columns protected) |
| 8  | Yes/No/True/False/1/0 → standardised to Yes/No |
| 9  | M/F/male/MALE/1/2 → standardised to Male/Female |
| 10 | Ages outside 16–80 → replaced with median (with safety checks) |
| 11 | Zero/negative salary → removed (currency-aware, with safety check) |
| 12 | Negative phone/numeric values → fixed |
| 13 | Extreme outliers → detected and reported (no data deleted) |
| 14 | Invalid email addresses → flagged |
| 15 | Phone numbers with dashes/brackets → cleaned |
| 16 | Special characters in name columns → removed |
| 17 | Duplicate employee IDs → detected and reported |
| 18 | Inconsistent date formats → YYYY-MM-DD |
| 19 | Department/job role casing → standardised (DevOps, HR, IT preserved) |
| 20 | Coded number columns (1,2,3…) → flagged for review |
""")

st.markdown(
    """
    <div style="text-align:center; color:grey; font-size:13px; margin-top:40px;">
        HR Data Quality Cleaning Pipeline |
        Kavya Chiganarapplara Thippeswamy |
        MSc Digital Innovation and Research |
        FH St. Pölten — University of Applied Sciences
    </div>
    """,
    unsafe_allow_html=True,
)
