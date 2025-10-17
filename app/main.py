from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pycountry
import re
import io
from datetime import datetime
from typing import Optional, List, Dict
import traceback

# ---------------- App setup ----------------
app = FastAPI(title="Promo Cleaning API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------------- Configuration --------------
DEFAULT_YEAR = 2025
PRODUCT_MAP: Dict[str, str] = {
    "terminal": "SumUp Terminal",
    "solo lite": "Solo Lite Only - New Packaging",
    "solo lite bundle": "Solo Lite & charging station",
    "solo": "Solo+New Cradle LTE",
    "soloprinter": "Solo+Printer Bundle 20W Plug, LTE",
    "solocounterprinter": "Solo & Till Printer Bundle",
    "air": "Air V5",
    "airbundle": "Bundle GB-800600016 (card_reader.air_bundle)",
    "poslite solo": "POS Lite Solo",
}
_processed_df: Optional[pd.DataFrame] = None

# ---------------- Utilities -----------------
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers, trim, collapse internal spaces, convert underscores to spaces."""
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", " ", regex=True)
          .str.replace("_", " ", regex=False)
    )
    return df

def collapse_spaces_in_string_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and collapse multiple spaces in all object columns (cells)."""
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = (
                df[c].astype(str)
                     .str.replace(r"\s+", " ", regex=True)
                     .str.strip()
            )
    return df

# ----------- Cleaning functions -------------
def clean_sku_columns(df: pd.DataFrame) -> pd.Index:
    """Clean and normalize product-related column names (after first 4 columns) — your exact steps kept."""
    sku_cols = df.columns[4:]
    sku_cols = (
        sku_cols.str.lower()
                .str.replace('discount', '', regex=False)
                .str.replace(r'\nname', '', regex=True)
                .str.replace(r'\n', ' ', regex=True)
                .str.replace('-', ' ', regex=False)
                .str.replace('_', ' ', regex=False)
                .str.replace('voucher name', 'name', regex=False)
                .str.replace('+', ' ', regex=False)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace('ed','')  # kept per your request
    )
    return sku_cols

def detect_product_columns(df: pd.DataFrame) -> list[str]:
    attr_pattern = r"(?:price|name|voucher)$"
    return [c for c in df.columns if re.search(attr_pattern, c)]

# -------------- Country mapping -------------
def map_country_code(code: str) -> str:
    """Exactly like notebook cell 3"""
    try:
        return pycountry.countries.get(alpha_2=code.upper()).name
    except:
        return code  # fallback if not found

# --------------- Date parsing ----------------
def normalize_date(row):
    """Exactly like notebook cell 2"""
    info = str(row)
    if len(info) > 1:
        start, end = info.split(' - ')
        start_date = datetime.strptime(f"2025.{start}", '%Y.%d.%m').replace(hour=0, minute=0, second=0)
        end_date = datetime.strptime(f"2025.{end}", '%Y.%d.%m').replace(hour=23, minute=59, second=59)
        return start_date, end_date
    else:
        return None, None

def extend_end_date(series: pd.Series) -> pd.Series:
    """Always add 7 days, preserving time if present."""
    s = pd.to_datetime(series, errors="coerce")
    return s + pd.Timedelta(days=7)

# -------------- Core processor --------------
def process_promotions_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Process the promotion data exactly like the notebook"""
    import numpy as np
    
    # Step 1: Clean column names (from notebook cell 1)
    sku_bundle = df.columns[4:]
    sku_bundle = sku_bundle.str.lower()
    sku_bundle = sku_bundle.str.replace('discount','')
    sku_bundle = sku_bundle.str.replace('\nname', '')
    sku_bundle = sku_bundle.str.replace('\n',' ').str.strip()
    sku_bundle = sku_bundle.str.replace('-',' ')
    sku_bundle = sku_bundle.str.replace('_',' ')
    sku_bundle = sku_bundle.str.replace('voucher name', 'name')
    sku_bundle = sku_bundle.str.replace('+',' ')
    sku_bundle = sku_bundle.str.strip(' ')
    sku_bundle = sku_bundle.str.replace('ed','')
    
    # Step 2: Rename columns (from notebook cell 5)
    rename_map = dict(zip(df.columns[4:], sku_bundle))
    df = df.rename(columns=rename_map)
    df.columns = (df.columns.str.strip()
                        .str.lower()
                        .str.replace(r"\s+", " ", regex=True))
    
    # Step 3: Process dates (from notebook cell 4)
    date_list = []
    for row in df['date']:
        start_date, end_date = normalize_date(row)
        date_list.append({
            'start_date': start_date,
            'end_date': end_date
        })
    
    date_info = pd.DataFrame(date_list)
    df['start date'] = date_info['start_date']
    df['end date'] = date_info['end_date']
    
    # Step 4: Melt and pivot data (from notebook cells 7-10)
    id_cols = [c for c in ["start date", "end date", "country", "currency"] if c in df.columns]
    attr_pattern = r"(?:price|name|voucher)$"
    product_cols = [c for c in df.columns if re.search(attr_pattern, c)]
    product_cols = [c for c in product_cols if c not in id_cols]
    
    long = df.melt(id_vars=id_cols, value_vars=product_cols,
                   var_name="prod_attr", value_name="value")
    split = long["prod_attr"].str.strip().str.extract(r"^(.*)\s+(price|name|voucher)$")
    long["product"] = split[0].str.strip()
    long["attribute"] = split[1].str.lower().replace({"voucher": "coupon"})
    
    tidy = (long.pivot_table(index=id_cols + ["product"],
                             columns="attribute",
                             values="value",
                             aggfunc="first")
                 .reset_index())
    
    tidy.columns.name = None
    order = [c for c in ["start date", "end date", "country", "currency", "product", "coupon", "price", "name"] if c in tidy.columns]
    tidy = tidy[order + [c for c in tidy.columns if c not in order]]
    
    # Step 5: Add country names and extra date (from notebook cell 11)
    tidy = tidy.copy()
    tidy['country_name'] = tidy['country'].apply(map_country_code)
    tidy["end date extra"] = tidy["end date"] + pd.Timedelta(days=7)
    
    # Step 6: Add SKU bundle mapping (from notebook cell 15)
    tidy['sku_bundle_name'] = tidy['product'].apply(lambda x: PRODUCT_MAP.get(x, x))
    
    # Step 7: Replace 'x' with NaN (from notebook cell 16)
    columns_to_replace = ['price', 'coupon', 'name']
    tidy[columns_to_replace] = tidy[columns_to_replace].replace('x', np.nan)
    
    # Step 8: Drop rows with null price (from notebook cell 18)
    tidy = tidy.dropna(subset=['price'])
    
    return tidy

# --------------- API Models -----------------
class UploadResponse(BaseModel):
    message: str
    total_records: int
    columns: List[str]

class HealthResponse(BaseModel):
    message: str
    total_records: int

# ----------------- Routes -------------------
@app.get("/", response_model=HealthResponse)
def root():
    return {
        "message": "Promo Cleaning API is running!",
        "total_records": 0 if _processed_df is None else int(len(_processed_df))
    }

# Add JSON cleaning function
def clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for JSON serialization"""
    df = df.copy()
    # Replace NaN with None for JSON compatibility
    df = df.where(pd.notnull(df), None)
    # Convert datetime to strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

@app.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    global _processed_df
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        _processed_df = process_promotions_csv(df)
        return {
            "message": "CSV processed successfully!",
            "total_records": int(len(_processed_df)),
            "columns": list(map(str, _processed_df.columns)),
        }
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}\n\nTraceback:\n{tb}")

@app.get("/promotions")
def get_promotions():
    try:
        if _processed_df is None:
            raise HTTPException(404, "No data available. Upload a CSV first.")
        print(f"🔍 Debug: DataFrame shape: {_processed_df.shape}")
        print(f"🔍 Debug: DataFrame columns: {list(_processed_df.columns)}")
        cleaned_df = clean_for_json(_processed_df)
        print(f"🔍 Debug: Cleaned DataFrame shape: {cleaned_df.shape}")
        return cleaned_df.to_dict(orient="records")
    except Exception as e:
        print(f"❌ Error in get_promotions: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.get("/promotions/{country}")
def get_promotions_by_country(country: str):
    if _processed_df is None:
        raise HTTPException(404, "No data available. Upload a CSV first.")
    mask = _processed_df["country"].astype(str).str.upper() == country.upper()
    data = _processed_df[mask]
    if data.empty:
        raise HTTPException(404, f"No promotions found for country: {country}")
    cleaned_df = clean_for_json(data)
    return cleaned_df.to_dict(orient="records")

@app.get("/promotions/{country}/{product}")
def get_promotions_by_country_and_product(country: str, product: str):
    if _processed_df is None:
        raise HTTPException(404, "No data available. Upload a CSV first.")
    mask_country = _processed_df["country"].astype(str).str.upper() == country.upper()
    mask_product = _processed_df["product"].astype(str).str.contains(product, case=False, na=False)
    data = _processed_df[mask_country & mask_product]
    if data.empty:
        raise HTTPException(status_code=404, detail=f"No promotions found for '{product}' in '{country}'")
    cleaned_df = clean_for_json(data)
    return cleaned_df.to_dict(orient="records")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)
