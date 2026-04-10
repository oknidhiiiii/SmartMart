from flask import Flask, render_template, request, jsonify, redirect, session
import pandas as pd
import sqlite3
import csv
import json
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db()
conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    items TEXT,
    total REAL,
    address TEXT
)
""")
conn.commit()
conn.close()

# ---------------- RECOMMENDATION ENGINE ----------------
def generate_recommendations():
    try:
        df_raw = pd.read_csv("dataset.csv")
    except:
        return [{"items": ["Milk", "Bread"], "discount": 50, "support": 0.1},
                {"items": ["Bread", "Butter"], "discount": 30, "support": 0.1}]

    if df_raw.empty:
        return [{"items": ["Milk", "Bread"], "discount": 50, "support": 0.1},
                {"items": ["Bread", "Butter"], "discount": 30, "support": 0.1}]

    # 🔥 Recency Weight
    df_raw['weight'] = 1
    recent_ids = df_raw['TID'].unique()[-20:]
    df_raw.loc[df_raw['TID'].isin(recent_ids), 'weight'] = 3

    # 🔥 Expand transactions
    transactions = []
    for _, row in df_raw.iterrows():
        if pd.isna(row['Items']):
            continue
        # Split items by comma and trim whitespace
        items = [item.strip() for item in str(row['Items']).split(',')]
        weight = int(row['weight'])
        for _ in range(weight):
            transactions.append(items)

    if len(transactions) < 2:
        return [{"items": ["Milk", "Bread"], "discount": 50, "support": 0.1},
                {"items": ["Bread", "Butter"], "discount": 30, "support": 0.1}]

    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_data, columns=te.columns_)

    # 🔥 Adaptive support
    min_sup = max(0.05, 1 / len(transactions))
    freq = apriori(df, min_support=min_sup, use_colnames=True)

    if freq.empty:
        return [{"items": ["Milk", "Bread"], "discount": 50, "support": 0.1},
                {"items": ["Bread", "Butter"], "discount": 30, "support": 0.1}]

    freq_filtered = freq[freq['itemsets'].apply(lambda x: len(x) > 1)]

    if freq_filtered.empty:
        freq_filtered = freq.sort_values(by='support', ascending=False).head(2)

    freq_sorted = freq_filtered.sort_values(by='support', ascending=False)
    top_combos = freq_sorted.head(2)

    recs = []
    for i, (_, row) in enumerate(top_combos.iterrows()):
        items = list(row['itemsets'])
        support = row['support']
        discount = 50 if i == 0 else 30

        recs.append({
            "items": items,
            "discount": discount,
            "support": round(support, 2)
        })

    return recs

# ---------------- ROUTES ----------------

@app.route('/')
def start():
    return redirect('/login')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form['username']
        p = request.form['password']

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (u, p)
        ).fetchone()
        conn.close()

        if user:
            session['user'] = u
            return redirect('/home')
        else:
            return "Invalid login"

    return render_template("login.html")

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u = request.form['username']
        p = request.form['password']

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (u, p)
            )
            conn.commit()
        except:
            return "User already exists!"
        finally:
            conn.close()

        return redirect('/login')

    return render_template("signup.html")

@app.route('/home')
def home():
    if "user" not in session:
        return redirect('/login')

    recs = generate_recommendations()
    return render_template("index.html", recs=recs)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# 🔥 SAVE ORDER + RUN APRIORI + RETURN UPDATED OFFERS
@app.route('/save_order', methods=['POST'])
def save_order():
    if "user" not in session:
        return redirect('/login')

    data = request.json
    items = data['items']

    # Save order
    conn = get_db()
    conn.execute(
        "INSERT INTO orders (username, items, total, address) VALUES (?, ?, ?, ?)",
        (session['user'], json.dumps(items), data['total'], data['address'])
    )
    conn.commit()
    conn.close()

    # 🔥 Update dataset
    try:
        df_existing = pd.read_csv("dataset.csv")
        new_tid = df_existing['TID'].max() + 1
    except:
        new_tid = 1

    with open("dataset.csv", "a", newline="") as f:
        writer = csv.writer(f)

        items_str = ", ".join(items)
        writer.writerow([new_tid, items_str])

    # 🔥 Run Apriori immediately
    updated_recs = generate_recommendations()

    return jsonify({
        "status": "saved",
        "recs": updated_recs
    })

@app.route('/get_orders')
def get_orders():
    if "user" not in session:
        return redirect('/login')

    conn = get_db()
    user_orders = conn.execute("SELECT * FROM orders WHERE username=?", (session['user'],)).fetchall()
    conn.close()
    
    result = []
    for order in user_orders:
        result.append({
            "items": order['items'],
            "total": order['total'],
            "address": order['address']
        })
    return jsonify(result)

@app.route('/api/analytics')
def analytics():
    if "user" not in session:
        return redirect('/login')

    conn = get_db()
    orders_db = conn.execute("SELECT items FROM orders").fetchall()
    conn.close()

    item_sales = {}
    for order in orders_db:
        try:
            items_dict = json.loads(order['items'])
            for item, details in items_dict.items():
                qty = details['qty']
                item_sales[item] = item_sales.get(item, 0) + qty
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    return jsonify(item_sales)

if __name__ == "__main__":
    app.run(debug=True, port=80000)