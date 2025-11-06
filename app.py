"""
Construction In/Out Inventory System — Streamlit UI (if available) + Non‑Interactive CLI
Database: SQLite (local file `inventory.db`)
Author: ChatGPT

New features (as requested)
--------------------------
- **Multi‑warehouse** with default warehouse `SHELL` (create more via CLI).
- **Per‑warehouse stock** and **warehouse‑to‑warehouse transfers**.
- **Per‑project issue** (OUT requires a project; stock checked per source warehouse).
- **PO → Receiving → DR flow**:
  - Create POs with items and costs.
  - Receive against a PO (partial OK) into a chosen warehouse.
  - Issue Delivery Receipts (DR) to projects (implemented as OUT transactions with reference numbers).
- CSV import/export remains backward compatible; new columns are optional.
- **Policy**: OUT is **blocked** when insufficient stock in the specified warehouse (default). You may override **only** via `--allow-negative`.

Why you previously saw errors
----------------------------
- `ModuleNotFoundError: streamlit` — Streamlit is optional; this app works without it.
- `OSError: [Errno 29]` — Some sandboxes disallow `input()`. We now use **argparse subcommands** only.
- `SyntaxError: '(' was never closed` — caused by **truncated calls** (e.g., `import_csv(args.`). Fully fixed below.

How to run
----------
A) Streamlit UI (if you have it):
   1) `pip install streamlit pandas`
   2) Save as `app.py`
   3) `streamlit run app.py`

B) Non‑interactive CLI examples (works in sandboxes / CI):
   - Init DB: `python app.py init-db`
   - Warehouses: `python app.py add-warehouse --code SHELL --name "Main Shell"` (auto‑created if missing), `python app.py list-warehouses`
   - Items/suppliers/projects: `add-item`, `add-supplier`, `add-project`
   - IN to a warehouse: `python app.py tx-in --sku CEM-40kg --qty 50 --supplier "ABC Supply" --wh SHELL`
   - OUT to a project from a warehouse: `python app.py tx-out --sku CEM-40kg --qty 12 --project PJ-001 --wh SHELL`
   - Transfer between warehouses: `python app.py transfer --sku CEM-40kg --qty 10 --src SHELL --dst SATELLITE`
   - Create PO + items:
     - `python app.py po-create --po PO-001 --supplier "ABC Supply"`
     - `python app.py po-add-item --po PO-001 --sku CEM-40kg --qty 100 --unit-cost 250`
   - Receive against PO into warehouse: `python app.py receive --po PO-001 --sku CEM-40kg --qty 40 --wh SHELL`
   - Issue a DR (OUT with reference): `python app.py dr-out --sku CEM-40kg --qty 12 --project PJ-001 --wh SHELL --dr DR-0001`
   - Stock views: `python app.py stock` (global), `python app.py stock --by-warehouse`
   - Tests: `python app.py --test`

"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, date
from typing import Iterable, Optional, Tuple, Dict, Any

# Optional dependencies
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import streamlit as st  # UI only, optional
    _STREAMLIT_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    _STREAMLIT_AVAILABLE = False

DB_PATH = os.environ.get("INVENTORY_DB", "inventory.db")
DEFAULT_WAREHOUSE_CODE = os.environ.get("DEFAULT_WAREHOUSE", "SHELL")

# -------------------------- Database & Schema (+migrations) -------------------------- #

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    contact TEXT,
    phone TEXT,
    email TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    location TEXT
);

CREATE TABLE IF NOT EXISTS warehouses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    location TEXT
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    unit TEXT NOT NULL,
    category TEXT,
    min_stock REAL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('IN','OUT','XFER_OUT','XFER_IN')),
    item_id INTEGER NOT NULL,
    qty REAL NOT NULL,
    unit_cost REAL,
    supplier_id INTEGER,
    project_id INTEGER,
    warehouse_id INTEGER,
    reference TEXT,
    remarks TEXT,
    FOREIGN KEY(item_id) REFERENCES items(id) ON DELETE CASCADE,
    FOREIGN KEY(supplier_id) REFERENCES suppliers(id) ON DELETE SET NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(warehouse_id) REFERENCES warehouses(id) ON DELETE SET NULL
);

-- Purchase Orders
CREATE TABLE IF NOT EXISTS purchase_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_no TEXT NOT NULL UNIQUE,
    supplier_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','CLOSED')),
    created_ts TEXT NOT NULL,
    FOREIGN KEY(supplier_id) REFERENCES suppliers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS purchase_order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    qty_ordered REAL NOT NULL,
    unit_cost REAL,
    qty_received REAL NOT NULL DEFAULT 0,
    FOREIGN KEY(po_id) REFERENCES purchase_orders(id) ON DELETE CASCADE,
    FOREIGN KEY(item_id) REFERENCES items(id) ON DELETE CASCADE
);
"""


def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c[1] == col for c in cols)


def init_db(db_path: str = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        # ensure default warehouse exists
        conn.execute(
            "INSERT OR IGNORE INTO warehouses(code, name) VALUES(?, ?)",
            (DEFAULT_WAREHOUSE_CODE, "Default Warehouse"),
        )
        # ensure transactions.warehouse_id exists (old DBs)
        if not _column_exists(conn, "transactions", "warehouse_id"):
            conn.execute("ALTER TABLE transactions ADD COLUMN warehouse_id INTEGER")
        conn.commit()

# -------------------------- Core Operations -------------------------- #

@dataclass
class Item:
    sku: str
    name: str
    unit: str
    category: Optional[str] = None
    min_stock: float = 0.0
    notes: Optional[str] = None


def upsert(
    table: str,
    data: Dict[str, Any],
    unique_field: Optional[str] = None,
    *,
    db_path: str = DB_PATH,
) -> Tuple[bool, str]:
    with closing(get_conn(db_path)) as conn:
        cols = ",".join(data.keys())
        placeholders = ",".join(["?" for _ in data])
        values = list(data.values())
        try:
            conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", values)
            conn.commit()
            return True, "Saved."
        except sqlite3.IntegrityError as e:
            if unique_field and unique_field in data:
                set_clause = ",".join([f"{k}=?" for k in data.keys() if k != unique_field])
                update_vals = [v for k, v in data.items() if k != unique_field]
                update_vals.append(data[unique_field])
                conn.execute(
                    f"UPDATE {table} SET {set_clause} WHERE {unique_field} = ?",
                    update_vals,
                )
                conn.commit()
                return True, "Updated."
            else:
                return False, f"Integrity error: {e}"


def add_item(item: Item, *, db_path: str = DB_PATH) -> Tuple[bool, str]:
    return upsert(
        "items",
        {
            "sku": item.sku.strip(),
            "name": item.name.strip(),
            "unit": item.unit.strip(),
            "category": (item.category or None),
            "min_stock": float(item.min_stock or 0),
            "notes": (item.notes or None),
        },
        unique_field="sku",
        db_path=db_path,
    )


def _get_id_by_code(
    conn: sqlite3.Connection, table: str, code_col: str, code_val: str
) -> Optional[int]:
    row = conn.execute(
        f"SELECT id FROM {table} WHERE {code_col}=?",
        (code_val,),
    ).fetchone()
    return int(row["id"]) if row else None


def ensure_warehouse(code: str, name: Optional[str] = None, *, db_path: str = DB_PATH) -> int:
    with closing(get_conn(db_path)) as conn:
        wid = _get_id_by_code(conn, 'warehouses', 'code', code)
        if wid is None:
            conn.execute("INSERT INTO warehouses(code, name) VALUES(?, ?)", (code, name or code))
            conn.commit()
            wid = _get_id_by_code(conn, 'warehouses', 'code', code)
        assert wid is not None
        return wid


def list_df(sql: str, params: Iterable[Any] | None = None, *, db_path: str = DB_PATH):
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute(sql, tuple(params or []))
        rows = cur.fetchall()
        if pd is not None:
            return pd.DataFrame([dict(r) for r in rows])
        return [dict(r) for r in rows]


def _warehouse_stock(conn: sqlite3.Connection, item_id: int, warehouse_id: int) -> float:
    cur = conn.execute(
        """
        SELECT COALESCE(SUM(CASE 
            WHEN type IN ('IN','XFER_IN') THEN qty 
            WHEN type IN ('OUT','XFER_OUT') THEN -qty 
            ELSE 0 END),0) AS stock
        FROM transactions WHERE item_id=? AND warehouse_id=?
        """,
        (item_id, warehouse_id),
    )
    return float(cur.fetchone()["stock"])  # type: ignore


def record_transaction(*, t_type: str, sku: str, qty: float, ts: Optional[str] = None,
                        unit_cost: Optional[float] = None, supplier_name: Optional[str] = None,
                        project_code: Optional[str] = None, reference: Optional[str] = None,
                        remarks: Optional[str] = None, warehouse_code: Optional[str] = None,
                        db_path: str = DB_PATH, allow_negative_stock: bool = False) -> Tuple[bool, str]:
    t_type = t_type.upper()
    assert t_type in {"IN", "OUT", "XFER_OUT", "XFER_IN"}
    if qty <= 0:
        return False, "Quantity must be > 0"

    with closing(get_conn(db_path)) as conn:
        # resolve item
        item = conn.execute("SELECT id FROM items WHERE sku=?", (sku,)).fetchone()
        if not item:
            return False, f"Unknown SKU: {sku}"
        item_id = int(item["id"])

        supplier_id = None
        project_id = None
        if supplier_name:
            srow = conn.execute("SELECT id FROM suppliers WHERE name=?", (supplier_name,)).fetchone()
            if srow:
                supplier_id = int(srow["id"])  
        if project_code:
            prow = conn.execute("SELECT id FROM projects WHERE code=?", (project_code,)).fetchone()
            if prow:
                project_id = int(prow["id"])  

        # resolve warehouse (required for all stock‑moving tx)
        wh_code = (warehouse_code or DEFAULT_WAREHOUSE_CODE)
        wh_row = conn.execute("SELECT id FROM warehouses WHERE code=?", (wh_code,)).fetchone()
        if not wh_row:
            return False, f"Unknown warehouse code: {wh_code}"
        warehouse_id = int(wh_row["id"])  

        # stock check per warehouse for OUT/XFER_OUT
        if t_type in {"OUT", "XFER_OUT"} and not allow_negative_stock:
            current_stock = _warehouse_stock(conn, item_id, warehouse_id)
            if qty > current_stock:
                return False, f"Insufficient stock in {wh_code}: have {current_stock}, need {qty}"

        ts_str = ts or datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO transactions (ts, type, item_id, qty, unit_cost, supplier_id, project_id, warehouse_id, reference, remarks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_str, t_type, item_id, float(qty), unit_cost, supplier_id, project_id, warehouse_id, reference, remarks),
        )
        conn.commit(); return True, "Transaction saved."


def current_stock_by_item(*, db_path: str = DB_PATH, by_warehouse: bool = False):
    with closing(get_conn(db_path)) as conn:
        if by_warehouse:
            cur = conn.execute(
                """
                SELECT i.sku, i.name, i.unit, w.code AS warehouse, 
                       COALESCE(SUM(CASE WHEN t.type IN ('IN','XFER_IN') THEN t.qty ELSE -t.qty END),0) AS stock
                FROM items i
                JOIN transactions t ON t.item_id = i.id
                LEFT JOIN warehouses w ON w.id = t.warehouse_id
                GROUP BY i.id, w.id
                ORDER BY i.name, w.code
                """
            )
        else:
            cur = conn.execute(
                """
                SELECT i.sku, i.name, i.unit, i.category, i.min_stock,
                       COALESCE(SUM(CASE WHEN t.type IN ('IN','XFER_IN') THEN t.qty ELSE -t.qty END),0) AS stock
                FROM items i
                LEFT JOIN transactions t ON t.item_id = i.id
                GROUP BY i.id
                ORDER BY i.name
                """
            )
        rows = cur.fetchall()
        if pd is not None:
            return pd.DataFrame([dict(r) for r in rows])
        return [dict(r) for r in rows]


def low_stock_items(*, db_path: str = DB_PATH):
    data = current_stock_by_item(db_path=db_path)
    if pd is not None and hasattr(data, "__getitem__") and hasattr(data, "loc"):
        return data[data["stock"] < data["min_stock"]]
    return [r for r in data if float(r.get("stock", 0)) < float(r.get("min_stock", 0))]

# -------------------------- PO / Receiving helpers -------------------------- #

def po_create(po_no: str, supplier_name: str, *, db_path: str = DB_PATH) -> Tuple[bool, str]:
    with closing(get_conn(db_path)) as conn:
        srow = conn.execute("SELECT id FROM suppliers WHERE name=?", (supplier_name,)).fetchone()
        if not srow:
            return False, f"Unknown supplier: {supplier_name}"
        conn.execute(
            "INSERT INTO purchase_orders(po_no, supplier_id, status, created_ts) VALUES(?,?, 'OPEN', ?)",
            (po_no, int(srow["id"]), datetime.now().isoformat(timespec='seconds')),
        )
        conn.commit(); return True, "PO created."


def po_add_item(po_no: str, sku: str, qty: float, unit_cost: Optional[float] = None, *, db_path: str = DB_PATH) -> Tuple[bool, str]:
    if qty <= 0: return False, "Quantity must be > 0"
    with closing(get_conn(db_path)) as conn:
        prow = conn.execute("SELECT id FROM purchase_orders WHERE po_no=?", (po_no,)).fetchone()
        if not prow: return False, f"Unknown PO: {po_no}"
        irow = conn.execute("SELECT id FROM items WHERE sku=?", (sku,)).fetchone()
        if not irow: return False, f"Unknown SKU: {sku}"
        conn.execute(
            "INSERT INTO purchase_order_items(po_id, item_id, qty_ordered, unit_cost) VALUES(?,?,?,?)",
            (int(prow["id"]), int(irow["id"]), float(qty), unit_cost),
        )
        conn.commit(); return True, "PO item added."


def receive_against_po(po_no: str, sku: str, qty: float, warehouse_code: Optional[str] = None, *, db_path: str = DB_PATH) -> Tuple[bool, str]:
    if qty <= 0: return False, "Quantity must be > 0"
    wh_code = warehouse_code or DEFAULT_WAREHOUSE_CODE
    with closing(get_conn(db_path)) as conn:
        prow = conn.execute("SELECT id, status FROM purchase_orders WHERE po_no=?", (po_no,)).fetchone()
        if not prow: return False, f"Unknown PO: {po_no}"
        if prow["status"] == 'CLOSED': return False, "PO is already closed"
        irow = conn.execute("SELECT id FROM items WHERE sku=?", (sku,)).fetchone()
        if not irow: return False, f"Unknown SKU: {sku}"
        pi = conn.execute("SELECT id, qty_ordered, qty_received FROM purchase_order_items WHERE po_id=? AND item_id=?", (int(prow["id"]), int(irow["id"])) ).fetchone()
        if not pi: return False, "Item not found on PO"
        new_recv = float(pi["qty_received"]) + qty
        if new_recv > float(pi["qty_ordered"]):
            return False, f"Receiving exceeds ordered qty. Ordered {pi['qty_ordered']}, current received {pi['qty_received']}, trying to receive {qty}."
        # record IN
        ok, msg = record_transaction(
            t_type='IN', sku=sku, qty=qty, reference=f"PO:{po_no}", warehouse_code=wh_code, db_path=db_path,
            allow_negative_stock=False
        )
        if not ok: return False, msg
        conn.execute("UPDATE purchase_order_items SET qty_received=? WHERE id=?", (new_recv, int(pi["id"])) )
        # close PO if all received
        agg = conn.execute("SELECT SUM(qty_ordered-qty_received) AS remaining FROM purchase_order_items WHERE po_id=?", (int(prow["id"]),)).fetchone()
        if float(agg["remaining"] or 0) <= 0:
            conn.execute("UPDATE purchase_orders SET status='CLOSED' WHERE id=?", (int(prow["id"]),))
        conn.commit(); return True, "Received."

# -------------------------- Streamlit UI (optional, updated) -------------------------- #

def run_streamlit_ui():  # pragma: no cover
    assert _STREAMLIT_AVAILABLE and st is not None
    st.set_page_config(page_title="Construction Inventory", page_icon="🏗️", layout="wide")
    init_db()

    st.title("🏗️ Construction In/Out Inventory")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Dashboard", "Items", "Transactions", "PO/Receiving", "Suppliers & Projects", "Import/Export", "Settings / Help"], index=0)

    if page == "Dashboard":
        cur = current_stock_by_item()
        if pd is not None:
            st.metric("Total Items", len(cur))
            st.metric("Low Stock Alerts", int((cur["stock"] < cur["min_stock"]).sum()))
            st.markdown("### Current Stock (Global)")
            st.dataframe(cur, use_container_width=True)
            st.markdown("### Current Stock by Warehouse")
            st.dataframe(current_stock_by_item(by_warehouse=True), use_container_width=True)
        else:
            st.write(current_stock_by_item(by_warehouse=False))

    elif page == "Transactions":
        st.subheader("Record Stock Movement (IN / OUT / Transfer)")
        items_df = list_df("SELECT * FROM items ORDER BY name")
        wh_df = list_df("SELECT code FROM warehouses ORDER BY code")
        wh_list = (wh_df["code"].tolist() if pd is not None and hasattr(wh_df, "__getitem__") else [DEFAULT_WAREHOUSE_CODE])
        with st.form("tx_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mode = st.selectbox("Mode", ["IN", "OUT", "TRANSFER"], index=0)
                item_name = st.selectbox("Item", options=(items_df["name"].tolist() if pd is not None else []))
                qty = st.number_input("Quantity", min_value=0.0, step=1.0, value=0.0)
            with c2:
                wh_src = st.selectbox("From Warehouse", options=wh_list, index=max(0, wh_list.index(DEFAULT_WAREHOUSE_CODE) if DEFAULT_WAREHOUSE_CODE in wh_list else 0))
                wh_dst = st.selectbox("To Warehouse (for Transfer)", options=wh_list)
            with c3:
                project = st.text_input("Project code (for OUT)")
                supplier = st.text_input("Supplier (for IN)")
                ref = st.text_input("Reference (PO/DR/JO)")
            if st.form_submit_button("Save"):
                sku = items_df.loc[items_df["name"] == item_name, "sku"].iloc[0]
                if mode == "IN":
                    ok, msg = record_transaction(t_type='IN', sku=sku, qty=qty, supplier_name=supplier or None, reference=ref or None, warehouse_code=wh_src, allow_negative_stock=False)
                elif mode == "OUT":
                    ok, msg = record_transaction(t_type='OUT', sku=sku, qty=qty, project_code=project or None, reference=ref or None, warehouse_code=wh_src, allow_negative_stock=False)
                else:
                    # transfer: XFER_OUT then XFER_IN
                    ok, msg = record_transaction(t_type='XFER_OUT', sku=sku, qty=qty, reference=ref or None, warehouse_code=wh_src, allow_negative_stock=False)
                    if ok:
                        ok, msg = record_transaction(t_type='XFER_IN', sku=sku, qty=qty, reference=ref or None, warehouse_code=wh_dst, allow_negative_stock=False)
                (st.success if ok else st.error)(msg)

    elif page == "PO/Receiving":
        st.subheader("Purchase Orders & Receiving")
        po_no = st.text_input("PO No.")
        supplier = st.text_input("Supplier Name")
        if st.button("Create PO"):
            ok, msg = po_create(po_no, supplier)
            (st.success if ok else st.error)(msg)
        with st.form("po_item_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                sku = st.text_input("SKU")
            with c2:
                qty = st.number_input("Qty Ordered", min_value=0.0, value=0.0)
            with c3:
                uc = st.number_input("Unit Cost", min_value=0.0, value=0.0)
            if st.form_submit_button("Add Item"):
                ok, msg = po_add_item(po_no, sku, qty, uc)
                (st.success if ok else st.error)(msg)
        with st.form("recv_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                r_sku = st.text_input("Receive SKU")
            with c2:
                r_qty = st.number_input("Receive Qty", min_value=0.0, value=0.0)
            with c3:
                r_wh = st.text_input("Warehouse Code", value=DEFAULT_WAREHOUSE_CODE)
            if st.form_submit_button("Receive"):
                ok, msg = receive_against_po(po_no, r_sku, r_qty, r_wh)
                (st.success if ok else st.error)(msg)

    elif page == "Items":
        st.subheader("Items Master List")
        with st.form("item_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                sku = st.text_input("SKU / Code *")
                name = st.text_input("Item Name *")
                unit = st.text_input("Unit (e.g., pcs, bag, kg, m) *")
                category = st.text_input("Category")
            with c2:
                min_stock = st.number_input("Minimum Stock Threshold", min_value=0.0, value=0.0, step=1.0)
                notes = st.text_area("Notes")
            if st.form_submit_button("Save Item"):
                if not (sku and name and unit):
                    st.warning("Please fill out SKU, Name, and Unit.")
                else:
                    ok, msg = add_item(Item(sku, name, unit, category or None, float(min_stock or 0), notes or None))
                    (st.success if ok else st.error)(msg)
        st.dataframe(list_df("SELECT * FROM items ORDER BY name"), use_container_width=True)

    elif page == "Suppliers & Projects":
        tab1, tab2 = st.tabs(["Suppliers", "Projects"])
        with tab1:
            with st.form("supplier_form"):
                s_name = st.text_input("Supplier Name *"); s_contact = st.text_input("Contact"); s_phone = st.text_input("Phone"); s_email = st.text_input("Email")
                if st.form_submit_button("Save Supplier"):
                    if not s_name:
                        st.warning("Supplier name is required.")
                    else:
                        ok, msg = upsert("suppliers", {"name": s_name.strip(), "contact": s_contact or None, "phone": s_phone or None, "email": s_email or None}, unique_field="name")
                        (st.success if ok else st.error)(msg)
            st.dataframe(list_df("SELECT * FROM suppliers ORDER BY name"), use_container_width=True)
        with tab2:
            with st.form("project_form"):
                p_code = st.text_input("Project Code *"); p_name = st.text_input("Project Name *"); p_loc = st.text_input("Location")
                if st.form_submit_button("Save Project"):
                    if not (p_code and p_name):
                        st.warning("Project code and name are required.")
                    else:
                        ok, msg = upsert("projects", {"code": p_code.strip(), "name": p_name.strip(), "location": p_loc or None}, unique_field="code")
                        (st.success if ok else st.error)(msg)
            st.dataframe(list_df("SELECT * FROM projects ORDER BY code"), use_container_width=True)

    elif page == "Import/Export":
        st.subheader("Import / Export Data")
        st.caption("Items CSV: sku,name,unit,category,min_stock,notes")
        st.caption("Transactions CSV (optional new cols): ts,type,sku,qty,unit_cost,warehouse_code,supplier_name,project_code,reference,remarks")
        if pd is not None:
            st.dataframe(list_df("SELECT * FROM transactions ORDER BY ts DESC, id DESC").head(100), use_container_width=True)

    else:
        st.subheader("Settings & Help")
        st.write("Set env INVENTORY_DB for DB file; DEFAULT_WAREHOUSE for default code.")

# -------------------------- CSV Import/Export (warehouse‑aware) -------------------------- #

def import_csv(path: str, *, db_path: str = DB_PATH) -> None:
    if not os.path.exists(path):
        print(f"File not found: {path}"); return
    base = os.path.basename(path).lower()
    if "item" in base:
        if pd is not None:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                add_item(Item(
                    sku=str(r.get("sku", "")).strip(),
                    name=str(r.get("name", "")).strip(),
                    unit=str(r.get("unit", "")).strip(),
                    category=(str(r.get("category", "")).strip() or None),
                    min_stock=float(r.get("min_stock", 0) or 0),
                    notes=(str(r.get("notes", "")).strip() or None),
                ), db_path=db_path)
        else:
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    add_item(Item(
                        sku=(r.get("sku") or "").strip(), name=(r.get("name") or "").strip(), unit=(r.get("unit") or "").strip(),
                        category=((r.get("category") or "").strip() or None), min_stock=float((r.get("min_stock") or 0) or 0), notes=((r.get("notes") or "").strip() or None)
                    ), db_path=db_path)
        print("Items import complete.")
    else:
        if pd is not None:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                record_transaction(
                    t_type=str(r.get("type", "IN")).strip().upper(),
                    sku=str(r.get("sku", "")).strip(),
                    qty=float(r.get("qty", 0) or 0),
                    ts=str(r.get("ts", datetime.now().isoformat(timespec="seconds"))),
                    unit_cost=(float(r.get("unit_cost", 0)) if r.get("unit_cost") == r.get("unit_cost") else None),
                    warehouse_code=(str(r.get("warehouse_code", DEFAULT_WAREHOUSE_CODE)).strip() or DEFAULT_WAREHOUSE_CODE),
                    supplier_name=(str(r.get("supplier_name", "")).strip() or None),
                    project_code=(str(r.get("project_code", "")).strip() or None),
                    reference=(str(r.get("reference", "")).strip() or None),
                    remarks=(str(r.get("remarks", "")).strip() or None),
                    db_path=db_path,
                )
        else:
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    record_transaction(
                        t_type=(r.get("type") or "IN").strip().upper(), sku=(r.get("sku") or "").strip(), qty=float((r.get("qty") or 0) or 0),
                        ts=(r.get("ts") or datetime.now().isoformat(timespec="seconds")),
                        unit_cost=(float(r.get("unit_cost")) if (r.get("unit_cost") not in (None, "")) else None),
                        warehouse_code=((r.get("warehouse_code") or DEFAULT_WAREHOUSE_CODE).strip() or DEFAULT_WAREHOUSE_CODE),
                        supplier_name=((r.get("supplier_name") or "").strip() or None), project_code=((r.get("project_code") or "").strip() or None),
                        reference=((r.get("reference") or "").strip() or None), remarks=((r.get("remarks") or "").strip() or None), db_path=db_path,
                    )
        print("Transactions import complete.")


def export_csv(folder: str, *, db_path: str = DB_PATH) -> None:
    os.makedirs(folder, exist_ok=True)
    with closing(get_conn(db_path)) as conn:
        items = [dict(r) for r in conn.execute("SELECT * FROM items ORDER BY name").fetchall()]
        tx = [dict(r) for r in conn.execute(
            "SELECT t.id, t.ts, t.type, i.sku, t.qty, t.unit_cost, w.code AS warehouse_code, s.name AS supplier_name, p.code AS project_code, t.reference, t.remarks "
            "FROM transactions t "
            "LEFT JOIN items i ON i.id=t.item_id "
            "LEFT JOIN warehouses w ON w.id=t.warehouse_id "
            "LEFT JOIN suppliers s ON s.id=t.supplier_id "
            "LEFT JOIN projects p ON p.id=t.project_id "
            "ORDER BY t.ts DESC, t.id DESC").fetchall()]
    ipath = os.path.join(folder, f"items_export_{date.today()}.csv")
    tpath = os.path.join(folder, f"transactions_export_{date.today()}.csv")
    with open(ipath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys()) if items else ["id","sku","name","unit","category","min_stock","notes"])
        w.writeheader(); w.writerows(items)
    with open(tpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tx[0].keys()) if tx else ["id","ts","type","sku","qty","unit_cost","warehouse_code","supplier_name","project_code","reference","remarks"]) 
        w.writeheader(); w.writerows(tx)
    print("Exported:", ipath); print("Exported:", tpath)

# -------------------------- CLI -------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Construction Inventory App")
    p.add_argument("--test", action="store_true", help="Run unit tests and exit")
    sub = p.add_subparsers(dest="cmd", help="Subcommands")

    sub.add_parser("init-db", help="Initialize the database")

    sp = sub.add_parser("add-warehouse", help="Add or update a warehouse")
    sp.add_argument("--code", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--location", default=None)

    sub.add_parser("list-warehouses", help="List warehouses")

    sp = sub.add_parser("add-item", help="Add or update an item")
    sp.add_argument("--sku", required=True); sp.add_argument("--name", required=True); sp.add_argument("--unit", required=True)
    sp.add_argument("--category", default=None); sp.add_argument("--min-stock", type=float, default=0); sp.add_argument("--notes", default=None)

    sp = sub.add_parser("add-supplier", help="Add or update a supplier")
    sp.add_argument("--name", required=True); sp.add_argument("--contact", default=None); sp.add_argument("--phone", default=None); sp.add_argument("--email", default=None)

    sp = sub.add_parser("add-project", help="Add or update a project")
    sp.add_argument("--code", required=True); sp.add_argument("--name", required=True); sp.add_argument("--location", default=None)

    sp = sub.add_parser("tx-in", help="Record an IN transaction")
    sp.add_argument("--sku", required=True); sp.add_argument("--qty", required=True, type=float); sp.add_argument("--supplier", default=None)
    sp.add_argument("--unit-cost", type=float, default=None); sp.add_argument("--ref", default=None); sp.add_argument("--remarks", default=None); sp.add_argument("--wh", default=DEFAULT_WAREHOUSE_CODE)

    sp = sub.add_parser("tx-out", help="Record an OUT transaction (to project)")
    sp.add_argument("--sku", required=True); sp.add_argument("--qty", required=True, type=float); sp.add_argument("--project", required=True)
    sp.add_argument("--ref", default=None); sp.add_argument("--remarks", default=None); sp.add_argument("--wh", default=DEFAULT_WAREHOUSE_CODE)
    sp.add_argument("--allow-negative", action="store_true", help="Allow negative stock for this transaction (override default block)")

    sp = sub.add_parser("transfer", help="Transfer stock between warehouses")
    sp.add_argument("--sku", required=True); sp.add_argument("--qty", required=True, type=float); sp.add_argument("--src", required=True); sp.add_argument("--dst", required=True)
    sp.add_argument("--ref", default=None); sp.add_argument("--remarks", default=None)

    sp = sub.add_parser("po-create", help="Create a purchase order")
    sp.add_argument("--po", required=True); sp.add_argument("--supplier", required=True)

    sp = sub.add_parser("po-add-item", help="Add an item to a PO")
    sp.add_argument("--po", required=True); sp.add_argument("--sku", required=True); sp.add_argument("--qty", required=True, type=float); sp.add_argument("--unit-cost", type=float, default=None)

    sp = sub.add_parser("receive", help="Receive against a PO into a warehouse")
    sp.add_argument("--po", required=True); sp.add_argument("--sku", required=True); sp.add_argument("--qty", required=True, type=float); sp.add_argument("--wh", default=DEFAULT_WAREHOUSE_CODE)

    sp = sub.add_parser("dr-out", help="Issue a Delivery Receipt (OUT)")
    sp.add_argument("--sku", required=True); sp.add_argument("--qty", required=True, type=float); sp.add_argument("--project", required=True); sp.add_argument("--wh", default=DEFAULT_WAREHOUSE_CODE)
    sp.add_argument("--dr", required=True); sp.add_argument("--remarks", default=None)

    sp = sub.add_parser("stock", help="Show current stock")
    sp.add_argument("--by-warehouse", action="store_true")

    sub.add_parser("low-stock", help="Show items below minimum stock (global)")
    sub.add_parser("list-tx", help="List recent transactions")

    sp = sub.add_parser("import", help="Import CSV (items or transactions)")
    sp.add_argument("--path", required=True)

    sp = sub.add_parser("export", help="Export CSVs to folder")
    sp.add_argument("--folder", required=True)

    return p


def run_cmd(args: argparse.Namespace) -> int:␊
    if args.cmd == "init-db": init_db(); print("DB initialized."); return 
    if args.cmd == "add-warehouse":
        ok, msg = upsert("warehouses", {"code": args.code, "name": args.name, "location": args.location}, unique_field="code"); print(msg); return 0 if ok else 1
    if args.cmd == "list-warehouses":
        rows = list_df("SELECT code, name, location FROM warehouses ORDER BY code"); print(rows.to_string(index=False) if pd is not None else rows); return 0
    if args.cmd == "add-item":
        ok, msg = add_item(Item(args.sku, args.name, args.unit, args.category, float(args.min_stock), args.notes)); print(msg); return 0 if ok else 1
    if args.cmd == "add-supplier":
        ok, msg = upsert("suppliers", {"name": args.name, "contact": args.contact, "phone": args.phone, "email": args.email}, unique_field="name"); print(msg); return 0 if ok else 1
    if args.cmd == "add-project":
        ok, msg = upsert("projects", {"code": args.code, "name": args.name, "location": args.location}, unique_field="code"); print(msg); return 0 if ok else 1
    if args.cmd == "tx-in":
        ok, msg = record_transaction(t_type='IN', sku=args.sku, qty=float(args.qty), supplier_name=args.supplier, unit_cost=args.unit_cost, reference=args.ref, remarks=args.remarks, warehouse_code=args.wh, allow_negative_stock=False); print(msg); return 0 if ok else 1
    if args.cmd == "tx-out":
        ok, msg = record_transaction(t_type='OUT', sku=args.sku, qty=float(args.qty), project_code=args.project, reference=args.ref, remarks=args.remarks, warehouse_code=args.wh, allow_negative_stock=bool(getattr(args, 'allow_negative', False))); print(msg); return 0 if ok else 1
    if args.cmd == "transfer":
        ok, msg = record_transaction(t_type='XFER_OUT', sku=args.sku, qty=float(args.qty), reference=args.ref, remarks=args.remarks, warehouse_code=args.src, allow_negative_stock=False)
        if ok:
            ok, msg = record_transaction(t_type='XFER_IN', sku=args.sku, qty=float(args.qty), reference=args.ref, remarks=args.remarks, warehouse_code=args.dst, allow_negative_stock=False)
        print(msg); return 0 if ok else 1
    if args.cmd == "po-create": ok, msg = po_create(args.po, args.supplier); print(msg); return 0 if ok else 1
    if args.cmd == "po-add-item": ok, msg = po_add_item(args.po, args.sku, float(args.qty), args.unit_cost); print(msg); return 0 if ok else 1
    if args.cmd == "receive": ok, msg = receive_against_po(args.po, args.sku, float(args.qty), args.wh); print(msg); return 0 if ok else 1
    if args.cmd == "dr-out": ok, msg = record_transaction(t_type='OUT', sku=args.sku, qty=float(args.qty), project_code=args.project, reference=f"DR:{args.dr}", remarks=args.remarks, warehouse_code=args.wh, allow_negative_stock=False); print(msg); return 0 if ok else 1
    if args.cmd == "stock":
        rows = current_stock_by_item(by_warehouse=bool(getattr(args, 'by_warehouse', False)))
        if pd is not None: print(rows.to_string(index=False))
        else: print(rows)
        return 0
    if args.cmd == "low-stock":
        rows = low_stock_items(); print(rows.to_string(index=False) if pd is not None else rows); return 0
    if args.cmd == "list-tx":
        with closing(get_conn()) as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM transactions ORDER BY ts DESC, id DESC LIMIT 200").fetchall()]
        print(rows.to_string(index=False) if (pd is not None and hasattr(pd, 'DataFrame')) else rows); return 0
    if args.cmd == "import":
        import_csv(args.path)
        return 0
    if args.cmd == "export":
        export_csv(args.folder)
        return 0
    return 0

# -------------------------- Tests -------------------------- #

def _run_tests() -> bool:
    import os as _os
    import tempfile
    import unittest

    class InventoryTests(unittest.TestCase):
        def setUp(self) -> None:
            self.tmp = tempfile.NamedTemporaryFile(delete=False)
            self.tmp.close()
            self.db_path = self.tmp.name
            init_db(self.db_path)
            add_item(Item("CEM-40kg", "Cement 40kg", "bag", min_stock=10), db_path=self.db_path)
            add_item(Item("RBR-10mm", "Rebar 10mm", "pc", min_stock=20), db_path=self.db_path)
            upsert("suppliers", {"name": "ABC Supply"}, unique_field="name", db_path=self.db_path)
            upsert("projects", {"code": "PJ-001", "name": "Bungalow"}, unique_field="code", db_path=self.db_path)
            upsert("warehouses", {"code": "SHELL", "name": "Main Shell"}, unique_field="code", db_path=self.db_path)
            upsert("warehouses", {"code": "SAT", "name": "Satellite"}, unique_field="code", db_path=self.db_path)

        def tearDown(self) -> None:
            try:
                _os.unlink(self.db_path)
            except Exception:
                pass

        def test_in_out_per_warehouse_blocking(self) -> None:
                        ok, msg = record_transaction(
                t_type="IN",
                sku="CEM-40kg",
                qty=50,
                supplier_name="ABC Supply",
                warehouse_code="SHELL",
                db_path=self.db_path,
            )
            self.assertTrue(ok, msg)
            ok, msg = record_transaction(
                t_type="OUT",
                sku="CEM-40kg",
                qty=20,
                project_code="PJ-001",
                warehouse_code="SHELL",
                db_path=self.db_path,
            )
            self.assertTrue(ok, msg)
            ok, msg = record_transaction(
                t_type="OUT",
                sku="CEM-40kg",
                qty=1,
                project_code="PJ-001",
                warehouse_code="SAT",
                db_path=self.db_path,
            )
            self.assertFalse(ok)
            self.assertIn("Insufficient stock", msg)

        def test_transfer_between_warehouses(self) -> None:
            ok, msg = record_transaction(
                t_type="IN",
                sku="RBR-10mm",
                qty=10,
                supplier_name="ABC Supply",
                warehouse_code="SHELL",
                db_path=self.db_path,
            )
            self.assertTrue(ok, msg)
            ok, msg = record_transaction(
                t_type="XFER_OUT",
                sku="RBR-10mm",
                qty=4,
                warehouse_code="SHELL",
                db_path=self.db_path,
            )
            self.assertTrue(ok, msg)
            ok, msg = record_transaction(
                t_type="XFER_IN",
                sku="RBR-10mm",
                qty=4,
                warehouse_code="SAT",
                db_path=self.db_path,
            )
            self.assertTrue(ok, msg)
            with closing(get_conn(self.db_path)) as conn:
                item_id = conn.execute("SELECT id FROM items WHERE sku=?", ("RBR-10mm",)).fetchone()["id"]
                shell_id = conn.execute("SELECT id FROM warehouses WHERE code=?", ("SHELL",)).fetchone()["id"]
                sat_id = conn.execute("SELECT id FROM warehouses WHERE code=?", ("SAT",)).fetchone()["id"]
                shell_stock = _warehouse_stock(conn, int(item_id), int(shell_id))
                sat_stock = _warehouse_stock(conn, int(item_id), int(sat_id))
            self.assertEqual(shell_stock, 6.0)
            self.assertEqual(sat_stock, 4.0)

        def test_receive_against_po_closes_when_complete(self) -> None:
            ok, msg = po_create("PO-001", "ABC Supply", db_path=self.db_path)
            self.assertTrue(ok, msg)
            ok, msg = po_add_item("PO-001", "CEM-40kg", 30, 250.0, db_path=self.db_path)
            self.assertTrue(ok, msg)
            ok, msg = receive_against_po("PO-001", "CEM-40kg", 20, "SHELL", db_path=self.db_path)
            self.assertTrue(ok, msg)
            with closing(get_conn(self.db_path)) as conn:
                status = conn.execute(
                    "SELECT status FROM purchase_orders WHERE po_no=?",
                    ("PO-001",),
                ).fetchone()["status"]
                received = conn.execute(
                    "SELECT qty_received FROM purchase_order_items WHERE po_id=(SELECT id FROM purchase_orders WHERE po_no=?)",
                    ("PO-001",),
                ).fetchone()["qty_received"]
            self.assertEqual(status, "OPEN")
            self.assertEqual(float(received), 20.0)
            ok, msg = receive_against_po("PO-001", "CEM-40kg", 10, "SHELL", db_path=self.db_path)
            self.assertTrue(ok, msg)
            with closing(get_conn(self.db_path)) as conn:
                status = conn.execute(
                    "SELECT status FROM purchase_orders WHERE po_no=?",
                    ("PO-001",),
                ).fetchone()["status"]
                received = conn.execute(
                    "SELECT qty_received FROM purchase_order_items WHERE po_id=(SELECT id FROM purchase_orders WHERE po_no=?)",
                    ("PO-001",),
                ).fetchone()["qty_received"]
                item_id = conn.execute("SELECT id FROM items WHERE sku=?", ("CEM-40kg",)).fetchone()["id"]
                shell_id = conn.execute("SELECT id FROM warehouses WHERE code=?", ("SHELL",)).fetchone()["id"]
                shell_stock = _warehouse_stock(conn, int(item_id), int(shell_id))
            self.assertEqual(status, "CLOSED")
            self.assertEqual(float(received), 30.0)
            self.assertEqual(shell_stock, 30.0)

        def test_low_stock_reporting(self) -> None:
            ok, msg = record_transaction(
                t_type="IN",
                sku="CEM-40kg",
                qty=5,
                supplier_name="ABC Supply",
                warehouse_code="SHELL",
                db_path=self.db_path,
            )
            self.assertTrue(ok, msg)
            lows = low_stock_items(db_path=self.db_path)
            if pd is not None and hasattr(lows, "__getitem__") and hasattr(lows, "to_dict"):
                sku_list = set(lows["sku"].tolist()) if len(lows) else set()
            else:
                sku_list = {row["sku"] for row in lows}
            self.assertIn("CEM-40kg", sku_list)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InventoryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test:
        return 0 if _run_tests() else 1

    if args.cmd == "init-db":
        init_db()
        print("DB initialized.")
        return 0

    if args.cmd:
        init_db()
        return run_cmd(args)

    if _STREAMLIT_AVAILABLE:
        run_streamlit_ui()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())