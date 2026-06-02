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
# SAME 20 CLEANING RULES AS ultimate_pipeline.py
# ================================================================
def clean_hr_data(df):
    fixes = []

    # RULE 1 — Column names
    orig = list(df.columns)
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ","_").str.replace("-","_")
                  .str.replace("(","").str.replace(")","")
                  .str.replace("/","_").str.replace(".","_"))
    n = sum(1 for o,c in zip(orig,df.columns) if o!=c)
    fixes.append({"n":1,"name":"Column Names Standardised","detail":f"{n} column names cleaned","count":n})

    # RULE 2 — Useless columns
    useless=[c for c in df.columns if df[c].nunique()<=1]
    if useless: df=df.drop(columns=useless)
    fixes.append({"n":2,"name":"Useless Columns Removed","detail":f"Removed: {useless}" if useless else "None found","count":len(useless)})

    # RULE 3 — Duplicates
    d=int(df.duplicated().sum()); df=df.drop_duplicates()
    fixes.append({"n":3,"name":"Duplicate Rows Removed","detail":f"Removed {d} duplicate rows" if d>0 else "No duplicates found","count":d})

    # RULE 4 — Whitespace cells
    ws=0
    for col in df.select_dtypes(include="object").columns:
        mask=df[col].astype(str).str.strip()==""
        if mask.sum()>0: df.loc[mask,col]=np.nan; ws+=int(mask.sum())
    fixes.append({"n":4,"name":"Whitespace Cells Fixed","detail":f"{ws} empty-space cells fixed" if ws>0 else "None found","count":ws})

    # RULE 5 — Data types
    tf=[]
    for col in df.columns:
        if df[col].dtype==object:
            t=df[col].astype(str).str.replace(",","").str.replace("$","").str.strip()
            c=pd.to_numeric(t,errors="coerce")
            if c.notna().sum()>len(df)*0.8: df[col]=c; tf.append(col)
    fixes.append({"n":5,"name":"Data Types Fixed","detail":f"Converted: {tf}" if tf else "All types correct","count":len(tf)})

    # RULE 6 — Missing values
    total=int(df.isnull().sum().sum()); details=[]
    for col in df.columns:
        n2=int(df[col].isnull().sum())
        if n2>0:
            if df[col].dtype in ["int64","float64"]:
                v=round(float(df[col].median()),1); df[col]=df[col].fillna(v); details.append(f"{col}: {n2} filled with {v}")
            else:
                df[col]=df[col].fillna("Unknown"); details.append(f"{col}: {n2} filled with Unknown")
    fixes.append({"n":6,"name":"Missing Values Fixed","detail":" | ".join(details) if details else "No missing values","count":total})

    # RULE 7 — Text formatting
    tc=df.select_dtypes(include="object").columns
    for col in tc: df[col]=df[col].astype(str).str.strip().str.title()
    fixes.append({"n":7,"name":"Text Formatting Cleaned","detail":f"Fixed in {len(tc)} text columns","count":len(tc)})

    # RULE 8 — Yes/No
    yn=0; ym={"yes":"Yes","no":"No","y":"Yes","n":"No","1":"Yes","0":"No","true":"Yes","false":"No"}
    for col in df.select_dtypes(include="object").columns:
        v=set(df[col].astype(str).str.lower().str.strip().unique()); v.discard("nan")
        if v.issubset(set(ym.keys())): df[col]=df[col].astype(str).str.lower().str.strip().map(ym).fillna("Unknown"); yn+=1
    fixes.append({"n":8,"name":"Yes/No Standardised","detail":f"{yn} columns standardised","count":yn})

    # RULE 9 — Gender
    gc=next((c for c in df.columns if "gender" in c.lower() or "sex" in c.lower()),None); gf=0
    if gc:
        gm={"M":"Male","F":"Female","m":"Male","f":"Female","male":"Male","female":"Female","MALE":"Male","FEMALE":"Female","Male":"Male","Female":"Female","man":"Male","woman":"Female","Man":"Male","Woman":"Female","1":"Male","2":"Female"}
        df[gc]=df[gc].astype(str).str.strip().replace(gm); gf=1
    fixes.append({"n":9,"name":"Gender Standardised","detail":f"Column '{gc}' → Male/Female" if gf else "No gender column","count":gf})

    # RULE 10 — Age validation
    ac=next((c for c in df.columns if "age" in c.lower()),None); af=0
    if ac:
        try:
            df[ac]=pd.to_numeric(df[ac],errors="coerce")
            inv=int(df[(df[ac]<16)|(df[ac]>80)].shape[0])
            if inv>0: df.loc[(df[ac]<16)|(df[ac]>80),ac]=df[ac].median(); af=inv
        except: pass
    fixes.append({"n":10,"name":"Invalid Ages Fixed","detail":f"{af} impossible ages fixed" if af>0 else "All ages valid (16-80)","count":af})

    # RULE 11 — Salary validation
    sc=next((c for c in df.columns if any(w in c.lower() for w in ["salary","income","pay","wage"])),None); sf=0
    if sc:
        try:
            df[sc]=pd.to_numeric(df[sc],errors="coerce")
            inv=int(df[df[sc]<=0].shape[0])
            if inv>0: df=df[df[sc]>0]; sf=inv
        except: pass
    fixes.append({"n":11,"name":"Invalid Salary Removed","detail":f"{sf} zero/negative rows removed" if sf>0 else "All salary values valid","count":sf})

    # RULE 12 — Negative numbers
    nf=0
    for col in df.select_dtypes(include="number").columns:
        if any(w in col.lower() for w in ["age","year","rate","hours","count","salary","income"]):
            neg=int((df[col]<0).sum())
            if neg>0: df.loc[df[col]<0,col]=df[col].median(); nf+=neg
    fixes.append({"n":12,"name":"Negative Values Fixed","detail":f"{nf} negative values replaced with median" if nf>0 else "No invalid negatives","count":nf})

    # RULE 13 — Outliers
    oc=[]
    for col in df.select_dtypes(include="number").columns:
        try:
            Q1=df[col].quantile(0.25); Q3=df[col].quantile(0.75); IQR=Q3-Q1
            if IQR>0:
                cnt=int(df[(df[col]<Q1-3*IQR)|(df[col]>Q3+3*IQR)].shape[0])
                if cnt>0: oc.append(f"{col}:{cnt}")
        except: pass
    fixes.append({"n":13,"name":"Outliers Detected","detail":" | ".join(oc) if oc else "No extreme outliers","count":len(oc)})

    # RULE 14 — Email validation
    ec=next((c for c in df.columns if "email" in c.lower() or "mail" in c.lower()),None); ef=0
    if ec:
        def ve(v): return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',str(v)))
        mask=~df[ec].apply(ve); ef=int(mask.sum())
        if ef>0: df.loc[mask,ec]="Invalid_Email"
    fixes.append({"n":14,"name":"Emails Validated","detail":f"{ef} invalid emails flagged" if ec else "No email column","count":ef})

    # RULE 15 — Phone cleaning
    pc=next((c for c in df.columns if any(w in c.lower() for w in ["phone","mobile","contact","tel"])),None); pf=0
    if pc:
        def cp(v):
            cl=re.sub(r'[\s\-\(\)\.\+]','',str(v))
            return cl if cl.isdigit() and 7<=len(cl)<=15 else str(v)
        orig2=df[pc].astype(str).copy(); df[pc]=df[pc].apply(cp); pf=int((df[pc]!=orig2).sum())
    fixes.append({"n":15,"name":"Phone Numbers Cleaned","detail":f"{pf} phones standardised" if pc else "No phone column","count":pf})

    # RULE 16 — Special characters
    spf=0
    for col in df.select_dtypes(include="object").columns:
        if any(w in col.lower() for w in ["name","first","last"]):
            o2=df[col].astype(str).copy()
            df[col]=df[col].astype(str).str.replace(r'[^a-zA-Z\s\-\.]','',regex=True).str.strip()
            spf+=int((df[col]!=o2).sum())
    fixes.append({"n":16,"name":"Special Characters Removed","detail":f"{spf} name fields cleaned" if spf>0 else "No special characters in names","count":spf})

    # RULE 17 — Employee ID
    ic=next((c for c in df.columns if any(w in c.lower() for w in ["employeeid","employee_id","emp_id","empid","staffid"])),None); idf=0
    if ic: idf=int(df[ic].duplicated().sum())
    fixes.append({"n":17,"name":"Employee ID Validated","detail":f"{idf} duplicate IDs found" if ic else "No ID column found","count":idf})

    # RULE 18 — Date formats
    df2=0
    for col in df.columns:
        if any(w in col.lower() for w in ["date","dob","birth","hired","joined","start","end"]):
            try: df[col]=pd.to_datetime(df[col],errors="coerce"); df[col]=df[col].dt.strftime("%Y-%m-%d"); df2+=1
            except: pass
    fixes.append({"n":18,"name":"Date Formats Standardised","detail":f"{df2} date columns → YYYY-MM-DD" if df2>0 else "No date columns found","count":df2})

    # RULE 19 — Department standardisation
    dptf=0
    for col in df.columns:
        if any(w in col.lower() for w in ["department","dept","division","jobrole","job_role","jobtitle","job_title","position"]):
            if df[col].dtype==object: df[col]=df[col].astype(str).str.strip().str.title(); dptf+=1
    fixes.append({"n":19,"name":"Departments Standardised","detail":f"{dptf} dept/job columns standardised" if dptf>0 else "No dept columns","count":dptf})

    # RULE 20 — Coded columns
    coded=[]
    for col in df.select_dtypes(include="number").columns:
        try:
            u=sorted(df[col].dropna().unique())
            if 2<=len(u)<=6 and float(min(u))>=1: coded.append(col)
        except: pass
    fixes.append({"n":20,"name":"Coded Columns Flagged","detail":f"{coded} may be categories" if coded else "No coded columns","count":len(coded)})

    return df, fixes


def calc_score(df):
    tc=df.shape[0]*df.shape[1]
    if tc==0: return 0
    m=max(0,25-(df.isnull().sum().sum()/tc*100*0.25))
    u=max(0,20-(df.duplicated().sum()/df.shape[0]*100*0.5))
    txc=df.select_dtypes(include="object").columns
    inc=sum(1 for c in txc if (df[c].astype(str).str.strip()!=df[c].astype(str)).sum()>0)
    co=max(0,20-(inc/max(len(txc),1)*20))
    ac=next((c for c in df.columns if "age" in c.lower()),None)
    if ac:
        try: ages=pd.to_numeric(df[ac],errors="coerce"); v=max(0,20-(int(((ages<16)|(ages>80)).sum())/max(len(ages),1)*100))
        except: v=18
    else: v=18
    bc=sum(1 for c in df.columns if c!=c.strip() or " " in c or c!=c.lower())
    s=max(0,15-(bc/df.shape[1]*15))
    return round(m+u+co+v+s,1)


# ================================================================
# WEBSITE UI
# ================================================================
st.title("📊 HR Data Quality Cleaning Pipeline")
st.markdown("### Automated HR Data Cleaning — Master's Thesis Tool")
st.markdown("Upload any messy HR CSV file. The pipeline applies **20 cleaning rules** automatically and gives you a clean file to download.")
st.divider()

st.header("📁 Step 1 — Upload Your HR CSV File")
f = st.file_uploader("Choose any HR CSV file", type=["csv"])

if f is not None:
    df_orig = pd.read_csv(f)
    st.success(f"✅ Uploaded: **{f.name}** — {df_orig.shape[0]:,} employees, {df_orig.shape[1]} columns")
    st.divider()

    st.header("👀 Step 2 — Your Original Messy Data")
    st.dataframe(df_orig.head(10), use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Employees", f"{df_orig.shape[0]:,}")
    c2.metric("Columns", df_orig.shape[1])
    c3.metric("Missing Values", int(df_orig.isnull().sum().sum()))
    c4.metric("Duplicates", int(df_orig.duplicated().sum()))
    st.divider()

    st.header("⚙️ Step 3 — Clean the Data")
    if st.button("🚀 Clean My HR Data Now", type="primary"):
        with st.spinner("Applying 20 cleaning rules..."):
            sb = calc_score(df_orig)
            df_clean, fixes = clean_hr_data(df_orig.copy())
            sa = calc_score(df_clean)
            imp = round(sa-sb,1)

        st.success("✅ Done! All 20 rules applied.")
        st.divider()

        st.header("📈 Quality Score — Before vs After")
        c1,c2,c3=st.columns(3)
        c1.metric("Before Cleaning", f"{sb}/100")
        c2.metric("After Cleaning", f"{sa}/100", delta=f"+{imp} points")
        c3.metric("Improvement", f"+{imp} points")

        if sa>=90: st.success("🏆 Final Grade: EXCELLENT")
        elif sa>=75: st.info("👍 Final Grade: GOOD")
        elif sa>=60: st.warning("⚠️ Final Grade: FAIR")
        else: st.error("❌ Final Grade: POOR")

        st.divider()
        st.header("🔧 All 20 Rules — What Was Fixed")
        for fix in fixes:
            if fix["count"]>0:
                st.success(f"✅ Rule {fix['n']}: **{fix['name']}** — {fix['detail']}")
            else:
                st.info(f"☑️ Rule {fix['n']}: **{fix['name']}** — {fix['detail']}")

        st.divider()
        st.header("✅ Step 4 — Your Clean Data")
        st.dataframe(df_clean.head(10), use_container_width=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Clean Employees", f"{df_clean.shape[0]:,}")
        c2.metric("Clean Columns", df_clean.shape[1])
        c3.metric("Missing Values", int(df_clean.isnull().sum().sum()))
        c4.metric("Duplicates", int(df_clean.duplicated().sum()))

        st.divider()
        st.header("⬇️ Step 5 — Download Clean File")
        clean_csv = df_clean.to_csv(index=False).encode("utf-8")
        fname = f.name.replace(".csv","_CLEANED.csv")
        st.download_button(
            label=f"⬇️ Download {fname}",
            data=clean_csv,
            file_name=fname,
            mime="text/csv",
            type="primary"
        )
        st.caption("HR Data Quality Pipeline v2.0 — Master's Thesis | 20 Cleaning Rules | Tested on 32,801 records")

else:
    st.info("👆 Upload a CSV file above to get started.")
    st.markdown("""
| Rule | What It Fixes |
|------|--------------|
| 1 | Messy column names → clean lowercase format |
| 2 | Useless columns (same value everywhere) → removed |
| 3 | Duplicate rows → removed |
| 4 | Whitespace-only cells → treated as empty |
| 5 | Numbers stored as text → converted to numbers |
| 6 | Missing values → filled with median or Unknown |
| 7 | Random CAPS and spaces in text → fixed |
| 8 | Yes/No/Y/N/1/0 → standardised to Yes/No |
| 9 | M/F/male/MALE/1/2 → standardised to Male/Female |
| 10 | Age under 16 or over 80 → replaced with median |
| 11 | Zero or negative salary → removed |
| 12 | Negative numbers in wrong columns → fixed |
| 13 | Extreme outliers → detected and reported |
| 14 | Invalid email addresses → flagged |
| 15 | Phone numbers with dashes/brackets → cleaned |
| 16 | Special characters in names → removed |
| 17 | Duplicate employee IDs → detected |
| 18 | Inconsistent date formats → YYYY-MM-DD |
| 19 | Department/job role inconsistencies → fixed |
| 20 | Coded number columns (1,2,3) → flagged |
""")