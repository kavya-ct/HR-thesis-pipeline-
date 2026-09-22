import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="HR Data Quality Pipeline",
    page_icon="📊",
    layout="wide"
)

# ================================================================
# HELPER — columns that must NOT be title-cased
# These are detected automatically, not hardcoded per dataset
# ================================================================
def _is_protected_column(col_name: str) -> bool:
    """
    Returns True for columns where str.title() would corrupt data.
    These columns are handled by specific rules (14, 17, 19) instead.
    - Email columns    (bob@example.com → Bob@Example.Com is wrong)
    - ID columns       (EMP1000 → Emp1000 changes the format)
    - Dept/role cols   (DevOps → Devops, HR → Hr is wrong — Rule 19 handles these)
    - URL / code cols
    """
    name = col_name.lower()
    # Email and ID — must keep original casing
    if any(w in name for w in ["email", "mail", "url", "link",
                                "id", "code", "ref", "key", "hash", "token"]):
        return True
    # Department and job columns — Rule 19 handles these with smart_title
    if any(w in name for w in ["department", "dept", "division",
                                "jobrole", "job_role", "jobtitle",
                                "job_title", "position", "role"]):
        return True
    return False


# ================================================================
# 20 CLEANING RULES — fixed version
# ================================================================
def clean_hr_data(df: pd.DataFrame):
    fixes = []

    # RULE 1 — Standardise column names
    orig = list(df.columns)
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )
    n = sum(1 for o, c in zip(orig, df.columns) if o != c)
    fixes.append({"n": 1, "name": "Column Names Standardised",
                  "detail": f"{n} column names cleaned to lowercase_underscore", "count": n})

    # RULE 2 — Remove useless columns (constant value in every row)
    useless = [c for c in df.columns if df[c].nunique() <= 1]
    if useless:
        df = df.drop(columns=useless)
    fixes.append({"n": 2, "name": "Useless Columns Removed",
                  "detail": f"Removed: {useless}" if useless else "None found", "count": len(useless)})

    # RULE 3 — Remove exact duplicate rows
    d = int(df.duplicated().sum())
    df = df.drop_duplicates()
    fixes.append({"n": 3, "name": "Duplicate Rows Removed",
                  "detail": f"Removed {d} duplicate rows" if d > 0 else "No duplicates found", "count": d})

    # RULE 4 — Fix whitespace-only cells (treat as missing)
    ws = 0
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].astype(str).str.strip() == ""
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
            ws += int(mask.sum())
    fixes.append({"n": 4, "name": "Whitespace Cells Fixed",
                  "detail": f"{ws} empty-space cells converted to blank" if ws > 0 else "None found", "count": ws})

    # RULE 5 — Fix wrong data types (text stored as numbers)
    # Skip columns that are email / ID type — they should stay as text
    tf = []
    for col in df.columns:
        if df[col].dtype == object and not _is_protected_column(col):
            t = df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
            c = pd.to_numeric(t, errors="coerce")
            if c.notna().sum() > len(df) * 0.8:
                df[col] = c
                tf.append(col)
    fixes.append({"n": 5, "name": "Data Types Fixed",
                  "detail": f"Converted to numbers: {tf}" if tf else "All types correct", "count": len(tf)})

    # RULE 6 — Fill missing values (numbers → median, text → Unknown)
    total = int(df.isnull().sum().sum())
    details = []
    for col in df.columns:
        n2 = int(df[col].isnull().sum())
        if n2 > 0:
            if df[col].dtype in ["int64", "float64"]:
                v = round(float(df[col].median()), 1)
                df[col] = df[col].fillna(v)
                details.append(f"{col}: {n2} filled with median({v})")
            else:
                df[col] = df[col].fillna("Unknown")
                details.append(f"{col}: {n2} filled with Unknown")
    fixes.append({"n": 6, "name": "Missing Values Fixed",
                  "detail": " | ".join(details) if details else "No missing values", "count": total})

    # RULE 7 — Fix text formatting (strip spaces + Title Case)
    # FIX: SKIP protected columns (email, ID, code, url)
    # so that emails and IDs are not corrupted
    tc_fixed = 0
    for col in df.select_dtypes(include="object").columns:
        if _is_protected_column(col):
            # Only strip spaces — do NOT apply title case
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = df[col].astype(str).str.strip().str.title()
            tc_fixed += 1
    fixes.append({"n": 7, "name": "Text Formatting Cleaned",
                  "detail": f"Title-case applied to {tc_fixed} columns (ID/email columns preserved)", "count": tc_fixed})

    # RULE 8 — Standardise Yes/No values
    # FIX: added 'true'/'false' to the map so boolean columns work
    yn = 0
    ym = {
        "yes": "Yes", "no": "No",
        "y": "Yes", "n": "No",
        "1": "Yes", "0": "No",
        "true": "Yes", "false": "No",       # ← was missing before
        "True": "Yes", "False": "No"        # ← was missing before
    }
    for col in df.select_dtypes(include="object").columns:
        v = set(df[col].astype(str).str.lower().str.strip().unique())
        v.discard("nan")
        if v and v.issubset({"yes", "no", "y", "n", "1", "0", "true", "false"}):
            df[col] = df[col].astype(str).str.lower().str.strip().map(ym).fillna("Unknown")
            yn += 1
    # FIX: also handle actual Python boolean columns (not just string)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].map({True: "Yes", False: "No"})
        yn += 1
    # FIX: also standardise integer columns that only contain 0 and 1
    # These are binary flag columns (married=0/1, terminated=0/1 etc)
    # that should be Yes/No for clarity — but only if clearly binary flags
    # (skip columns like salary, age that happen to have 0 values)
    # Skip keywords for 0/1 → Yes/No conversion
    # Note: "id" removed from skip list because "marriedid" contains "id"
    # Instead we use a smarter check: skip only if column has many unique values
    # (real ID cols have hundreds of unique values; binary flags have only 0 and 1)
    yn_skip_keywords = ["salary","income","pay","wage","age","year","rate","hours",
                        "score","level","count","number","percent"]
    for col in df.select_dtypes(include=["int64","int32","Int64"]).columns:
        if any(w in col.lower() for w in yn_skip_keywords):
            continue
        u = set(df[col].dropna().unique())
        if u.issubset({0, 1, np.int64(0), np.int64(1)}):
            df[col] = df[col].map({0: "No", 1: "Yes", np.int64(0): "No", np.int64(1): "Yes"})
            yn += 1
    fixes.append({"n": 8, "name": "Yes/No Standardised",
                  "detail": f"{yn} columns standardised to Yes/No", "count": yn})

    # RULE 9 — Standardise gender values
    # FIX: Rule 7 now preserves case for non-text columns but gender col
    # is a normal text col so after title() values look like "Male"/"Female"
    # Map covers pre-title and post-title variations to be safe
    gc = next((c for c in df.columns if "gender" in c.lower() or "sex" in c.lower()), None)
    gf = 0
    if gc:
        gm = {
            # original raw variations
            "M": "Male", "F": "Female", "m": "Male", "f": "Female",
            "male": "Male", "female": "Female",
            "MALE": "Male", "FEMALE": "Female",
            "man": "Male", "woman": "Female",
            "1": "Male", "2": "Female", "0": "Female",
            # post-title() variations
            "Male": "Male", "Female": "Female",
            "Man": "Male", "Woman": "Female",
        }
        df[gc] = df[gc].astype(str).str.strip().replace(gm)
        gf = 1
    fixes.append({"n": 9, "name": "Gender Standardised",
                  "detail": f"Column '{gc}' → Male/Female" if gf else "No gender column", "count": gf})

    # RULE 10 — Validate age (must be 16-80 for working employees)
    # FIX: use word-boundary matching to avoid matching "age" inside
    # "managername", "managerid", "engagementsurvey" etc.
    # Only match columns where "age" is a standalone word/segment.
    import re as _re
    ac = next(
        (c for c in df.columns
         if _re.search(r'(^|_)age($|_)', c.lower())),
        None
    )
    af = 0
    if ac:
        try:
            df[ac] = pd.to_numeric(df[ac], errors="coerce")
            inv = int(df[(df[ac] < 16) | (df[ac] > 80)].shape[0])
            if inv > 0:
                df.loc[(df[ac] < 16) | (df[ac] > 80), ac] = df[ac].median()
                af = inv
        except Exception:
            pass
    fixes.append({"n": 10, "name": "Invalid Ages Fixed",
                  "detail": f"{af} impossible ages replaced with median" if af > 0 else "All ages valid (16-80)", "count": af})

    # RULE 11 — Remove zero/negative salary rows
    sc = next((c for c in df.columns if any(w in c.lower() for w in ["salary", "income", "pay", "wage"])), None)
    sf = 0
    if sc:
        try:
            df[sc] = pd.to_numeric(df[sc], errors="coerce")
            inv = int(df[df[sc] <= 0].shape[0])
            if inv > 0:
                df = df[df[sc] > 0]
                sf = inv
        except Exception:
            pass
    fixes.append({"n": 11, "name": "Invalid Salary Removed",
                  "detail": f"{sf} zero/negative rows removed" if sf > 0 else "All salary values valid", "count": sf})

    # RULE 12 — Fix negative numbers in columns that cannot logically be negative
    # FIX: added "phone", "mobile", "tel", "contact" to the keyword list
    # Phone numbers stored as integers are often negative due to overflow — flag and fix
    nf = 0
    neg_keywords = ["age", "year", "rate", "hours", "count", "salary",
                    "income", "phone", "mobile", "tel", "contact"]
    for col in df.select_dtypes(include="number").columns:
        if any(w in col.lower() for w in neg_keywords):
            neg = int((df[col] < 0).sum())
            if neg > 0:
                # For phone: take absolute value (negative = data entry sign error)
                # For others: replace with median
                if any(w in col.lower() for w in ["phone", "mobile", "tel", "contact"]):
                    df[col] = df[col].abs()
                    nf += neg
                else:
                    df.loc[df[col] < 0, col] = df[col].median()
                    nf += neg
    fixes.append({"n": 12, "name": "Negative Values Fixed",
                  "detail": f"{nf} negative values fixed (phone: abs value, others: median)" if nf > 0 else "No invalid negatives", "count": nf})

    # RULE 13 — Detect extreme outliers using IQR method (report only, do not remove)
    oc = []
    for col in df.select_dtypes(include="number").columns:
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                cnt = int(df[(df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)].shape[0])
                if cnt > 0:
                    oc.append(f"{col}: {cnt} outliers")
        except Exception:
            pass
    fixes.append({"n": 13, "name": "Outliers Detected",
                  "detail": " | ".join(oc) if oc else "No extreme outliers", "count": len(oc)})

    # RULE 14 — Validate email addresses
    # FIX: validate BEFORE any case corruption — Rule 7 now preserves email columns
    # so emails still have original case when Rule 14 runs (after fix to Rule 7)
    ec = next((c for c in df.columns if "email" in c.lower() or "mail" in c.lower()), None)
    ef = 0
    if ec:
        pattern = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
        def is_valid_email(v):
            return bool(pattern.match(str(v).strip()))
        mask = ~df[ec].apply(is_valid_email)
        ef = int(mask.sum())
        if ef > 0:
            df.loc[mask, ec] = "invalid_email"
    fixes.append({"n": 14, "name": "Emails Validated",
                  "detail": f"{ef} invalid emails flagged" if ec else "No email column", "count": ef})

    # RULE 15 — Clean phone numbers (remove dashes, brackets, spaces)
    pc = next((c for c in df.columns if any(w in c.lower() for w in ["phone", "mobile", "contact", "tel"])), None)
    pf = 0
    if pc:
        if df[pc].dtype == object:
            def cp(v):
                cl = re.sub(r'[\s\-\(\)\.\+]', '', str(v))
                return cl if cl.lstrip("-").isdigit() and 7 <= len(cl.lstrip("-")) <= 15 else str(v)
            orig2 = df[pc].astype(str).copy()
            df[pc] = df[pc].apply(cp)
            pf = int((df[pc] != orig2).sum())
    fixes.append({"n": 15, "name": "Phone Numbers Cleaned",
                  "detail": f"{pf} phones standardised" if pc else "No phone column", "count": pf})

    # RULE 16 — Remove special characters from name columns only
    spf = 0
    for col in df.select_dtypes(include="object").columns:
        if any(w in col.lower() for w in ["first_name", "last_name", "full_name"]):
            o2 = df[col].astype(str).copy()
            df[col] = df[col].astype(str).str.replace(r'[^a-zA-Z\s\-\.]', '', regex=True).str.strip()
            spf += int((df[col] != o2).sum())
    fixes.append({"n": 16, "name": "Special Characters Removed",
                  "detail": f"{spf} name fields cleaned" if spf > 0 else "No special characters found", "count": spf})

    # RULE 17 — Detect duplicate employee IDs
    id_keywords = ["employeeid", "employee_id", "emp_id", "empid", "staffid", "staff_id"]
    ic = next((c for c in df.columns if any(w in c.lower() for w in id_keywords)), None)
    idf = 0
    if ic:
        idf = int(df[ic].duplicated().sum())
    fixes.append({"n": 17, "name": "Employee ID Validated",
                  "detail": f"{idf} duplicate IDs detected" if ic else "No ID column found", "count": idf})

    # RULE 18 — Standardise date columns to YYYY-MM-DD
    # FIX 1: removed infer_datetime_format=True (removed in pandas 2.x)
    # FIX 2: exclude "Unknown" placeholder values (added by Rule 6) when
    #         calculating success rate — otherwise termination date columns
    #         (which have many NaN → "Unknown") appear to have low parse rate
    #         and get skipped even though their real date values are valid.
    df2c = 0
    date_keywords = ["date", "dob", "birth", "hired", "joined", "start", "end", "termination"]
    for col in df.columns:
        if any(w in col.lower() for w in date_keywords):
            if col not in df.columns:
                continue
            # Skip numeric columns (e.g. genderid falsely matches "id")
            if df[col].dtype in ["int64", "float64"]:
                continue
            try:
                # Exclude "Unknown" placeholder when measuring parse success
                real_vals = df[col][df[col].astype(str).str.strip().str.lower() != "unknown"]
                if len(real_vals) == 0:
                    continue
                converted_real = pd.to_datetime(real_vals, errors="coerce", dayfirst=False)
                success_rate = converted_real.notna().sum() / max(len(real_vals), 1)
                if success_rate >= 0.8:
                    # Convert the full column — Unknown stays as Unknown (unparseable)
                    converted_full = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
                    # Where conversion succeeded, use YYYY-MM-DD; else keep original value
                    df[col] = converted_full.dt.strftime("%Y-%m-%d").where(
                        converted_full.notna(), other=df[col]
                    )
                    df2c += 1
            except Exception:
                pass
    fixes.append({"n": 18, "name": "Date Formats Standardised",
                  "detail": f"{df2c} date columns → YYYY-MM-DD" if df2c > 0 else "No date columns found", "count": df2c})

    # RULE 19 — Standardise department and job role columns
    # FIX: use smart_title() instead of str.title() to preserve
    # internal capitals like DevOps, HR, IT, so they are not
    # corrupted to Devops, Hr, It
    def smart_title(val: str) -> str:
        """
        Capitalise first letter of each word intelligently:
        - All-lowercase word (finance, sales) → capitalise (Finance, Sales)
        - All-uppercase short word (HR, IT, AI, BI) → keep uppercase
        - All-uppercase long word (FINANCE) → Title-case (Finance)
        - Mixed-case word (DevOps, FinTech) → preserve exactly
        Handles hyphenated values like DevOps-California correctly.
        """
        val = str(val).strip()
        # Handle hyphen-separated parts (DevOps-California)
        parts = val.split("-")
        result_parts = []
        for part in parts:
            words = part.split()
            result_words = []
            for w in words:
                if w.islower():
                    result_words.append(w.capitalize())
                elif w.isupper() and len(w) <= 5:
                    result_words.append(w)          # HR, IT, AI, BI, CLOUD → keep
                elif w.isupper() and len(w) > 5:
                    result_words.append(w.capitalize())  # FINANCE → Finance
                else:
                    result_words.append(w)          # DevOps, FinTech → preserve
            result_parts.append(" ".join(result_words))
        return "-".join(result_parts)

    dptf = 0
    dept_keywords = ["department", "dept", "division", "jobrole", "job_role",
                     "jobtitle", "job_title", "position", "role"]
    for col in df.columns:
        if any(w in col.lower() for w in dept_keywords):
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).apply(smart_title)
                dptf += 1
    fixes.append({"n": 19, "name": "Departments Standardised",
                  "detail": f"{dptf} dept/job columns standardised (mixed-case preserved)" if dptf > 0 else "No dept columns", "count": dptf})

    # RULE 20 — Flag coded number columns (1,2,3 used as category labels)
    coded = []
    for col in df.select_dtypes(include="number").columns:
        try:
            u = sorted(df[col].dropna().unique())
            if 2 <= len(u) <= 6 and float(min(u)) >= 1:
                coded.append(col)
        except Exception:
            pass
    fixes.append({"n": 20, "name": "Coded Columns Flagged",
                  "detail": f"{coded} may be category codes — check data dictionary" if coded else "No coded columns detected", "count": len(coded)})

    return df, fixes


# ================================================================
# QUALITY SCORE — 5 dimensions, 0-100
# ================================================================
def calc_score(df: pd.DataFrame) -> float:
    tc = df.shape[0] * df.shape[1]
    if tc == 0:
        return 0.0
    # Completeness (25 pts)
    m = max(0.0, 25.0 - (df.isnull().sum().sum() / tc * 100 * 0.25))
    # Uniqueness (20 pts)
    u = max(0.0, 20.0 - (df.duplicated().sum() / df.shape[0] * 100 * 0.5))
    # Consistency (20 pts)
    txc = df.select_dtypes(include="object").columns
    inc = sum(1 for c in txc if (df[c].astype(str).str.strip() != df[c].astype(str)).sum() > 0)
    co = max(0.0, 20.0 - (inc / max(len(txc), 1) * 20))
    # Validity (20 pts)
    ac = next((c for c in df.columns if "age" in c.lower()), None)
    if ac:
        try:
            ages = pd.to_numeric(df[ac], errors="coerce")
            v = max(0.0, 20.0 - (int(((ages < 16) | (ages > 80)).sum()) / max(len(ages), 1) * 100))
        except Exception:
            v = 18.0
    else:
        v = 18.0
    # Structure (15 pts)
    bc = sum(1 for c in df.columns if c != c.strip() or " " in c or c != c.lower())
    s = max(0.0, 15.0 - (bc / df.shape[1] * 15))
    return round(m + u + co + v + s, 1)


# ================================================================
# STREAMLIT UI
# ================================================================
st.title("📊 HR Data Quality Cleaning Pipeline")
st.markdown("### Automatically Detect, Clean, and Improve HR Data Quality")
st.markdown(
    "Upload any messy HR CSV file. The pipeline applies **20 adaptive cleaning rules** "
    "and gives you a clean file to download."
)
st.divider()

st.header("📁 Step 1 — Upload Your HR CSV File")
f = st.file_uploader("Choose any HR CSV file", type=["csv"])

if f is not None:
    try:
        df_orig = pd.read_csv(f)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"✅ Uploaded: **{f.name}** — {df_orig.shape[0]:,} rows, {df_orig.shape[1]} columns")
    st.divider()

    st.header("👀 Step 2 — Your Original Data")
    st.dataframe(df_orig.head(10), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_orig.shape[0]:,}")
    c2.metric("Columns", df_orig.shape[1])
    c3.metric("Missing Values", int(df_orig.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df_orig.duplicated().sum()))
    st.divider()

    st.header("⚙️ Step 3 — Clean the Data")
    if st.button("🚀 Clean My HR Data Now", type="primary"):
        with st.spinner("Applying 20 cleaning rules..."):
            sb = calc_score(df_orig)
            df_clean, fixes = clean_hr_data(df_orig.copy())
            sa = calc_score(df_clean)
            imp = round(sa - sb, 1)

        st.success("✅ Done! All 20 rules applied.")
        st.divider()

        st.header("📈 Quality Score — Before vs After")
        c1, c2, c3 = st.columns(3)
        c1.metric("Before Cleaning", f"{sb}/100")
        c2.metric("After Cleaning", f"{sa}/100", delta=f"+{imp} points")
        c3.metric("Improvement", f"+{imp} points")

        if sa >= 90:
            st.success("🏆 Final Grade: EXCELLENT")
        elif sa >= 75:
            st.info("👍 Final Grade: GOOD")
        elif sa >= 60:
            st.warning("⚠️ Final Grade: FAIR")
        else:
            st.error("❌ Final Grade: POOR")

        st.divider()
        st.header("🔧 All 20 Rules — What Was Fixed")
        for fix in fixes:
            if fix["count"] > 0:
                st.success(f"✅ Rule {fix['n']}: **{fix['name']}** — {fix['detail']}")
            else:
                st.info(f"☑️ Rule {fix['n']}: **{fix['name']}** — {fix['detail']}")

        st.divider()
        st.header("✅ Step 4 — Your Clean Data")
        st.dataframe(df_clean.head(10), use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean Rows", f"{df_clean.shape[0]:,}")
        c2.metric("Clean Columns", df_clean.shape[1])
        c3.metric("Missing Values", int(df_clean.isnull().sum().sum()))
        c4.metric("Duplicate Rows", int(df_clean.duplicated().sum()))

        st.divider()
        st.header("⬇️ Step 5 — Download Clean File")
        clean_csv = df_clean.to_csv(index=False).encode("utf-8")
        fname = f.name.replace(".csv", "_CLEANED.csv")
        st.download_button(
            label=f"⬇️ Download {fname}",
            data=clean_csv,
            file_name=fname,
            mime="text/csv",
            type="primary"
        )
        st.caption(
            "HR Data Quality Pipeline v2.1 — Master's Thesis | "
            "20 Adaptive Cleaning Rules | Tested on 32,801 records"
        )

else:
    st.info("👆 Upload a CSV file above to get started.")
    st.markdown("""
| Rule | What It Fixes |
|------|--------------|
| 1 | Messy column names → clean lowercase format |
| 2 | Useless constant columns → removed |
| 3 | Duplicate rows → removed |
| 4 | Whitespace-only cells → treated as empty |
| 5 | Numbers stored as text → converted (ID/email columns protected) |
| 6 | Missing values → filled with median or Unknown |
| 7 | Random CAPS and spaces → fixed (email/ID columns preserved) |
| 8 | Yes/No/Y/N/1/0/True/False → standardised to Yes/No |
| 9 | M/F/male/MALE/1/2 → standardised to Male/Female |
| 10 | Age under 16 or over 80 → replaced with median |
| 11 | Zero or negative salary → removed |
| 12 | Negative phone/numeric values → fixed (phone: absolute value) |
| 13 | Extreme outliers → detected and reported |
| 14 | Invalid email addresses → flagged |
| 15 | Phone numbers with dashes/brackets → cleaned |
| 16 | Special characters in name columns → removed |
| 17 | Duplicate employee IDs → detected |
| 18 | Inconsistent date formats → YYYY-MM-DD (if >80% parseable) |
| 19 | Department/job role inconsistencies → Title Case |
| 20 | Coded number columns (1,2,3) → flagged for review |
""")

st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 14px; margin-top: 40px;">
        Created by Kavya Chiganarapplara Thippeswamy | USTP – University of Applied Sciences St. Pölten<br>
        Master's Thesis Project | Digital Innovation and Research
    </div>
    """,
    unsafe_allow_html=True
)
