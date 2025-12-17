import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import PIL.Image
import json
import re
import time
import io
import gzip
import base64

# ==========================================
# 🔐 初期設定 & デザインカスタマイズ
# ==========================================
st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide", page_icon="🏔️")

# --- 🎨 カスタムCSS (QB風デザイン & スマホ最適化 & 固定カウントダウン) ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif;
        color: #333;
    }

    /* 背景色を少しグレーにしてカードを目立たせる */
    .stApp {
        background-color: #f0f2f5;
    }

    /* 🔹 ヘッダーの固定カウントダウン */
    .fixed-header {
        position: fixed;
        top: 0;
        right: 0;
        z-index: 99999;
        background: rgba(255, 255, 255, 0.95);
        padding: 8px 16px;
        border-bottom-left-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: right;
        border-left: 4px solid #007bff;
    }
    .countdown-label {
        font-size: 10px;
        color: #666;
        display: block;
        line-height: 1;
        margin-bottom: 2px;
    }
    .countdown-days {
        font-size: 18px;
        font-weight: 800;
        color: #d9534f;
    }

    /* 🔹 カード風デザイン (QB風) */
    div.stContainer {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e1e4e8;
    }
    
    /* ボタンのスタイル改良 */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.2s;
    }
    /* プライマリボタン (青) */
    .stButton > button[kind="primary"] {
        background-color: #007bff;
        color: white;
        box-shadow: 0 4px 6px rgba(0,123,255,0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0056b3;
    }
    
    /* 重要数字の強調 */
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }

    /* スマホ調整 */
    @media (max-width: 640px) {
        .block-container {
            padding-top: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .fixed-header {
            top: 50px; /* Streamlitの標準ヘッダーの下あたり */
            padding: 4px 10px;
        }
        .countdown-days { font-size: 14px; }
    }
</style>
""", unsafe_allow_html=True)

# --- ⏳ カウントダウン計算と表示 ---
exam_date = datetime.date(2026, 3, 4)
today = datetime.date.today()
days_left = (exam_date - today).days
if days_left < 0: days_left = 0

st.markdown(f"""
<div class="fixed-header">
    <span class="countdown-label">新潟高校入試まで</span>
    <span class="countdown-days">あと {days_left} 日</span>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 🤖 API & モデル設定
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

if not api_key:
    st.warning("⚠️ SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=api_key)

# モデル自動検出 (ロジック維持)
def get_available_models():
    try:
        return [m.name.replace("models/", "") for m in genai.list_models()]
    except:
        return []

ALL_MODELS = get_available_models()

def get_best_pro_model(all_models):
    priority_list = [
        "gemini-1.5-pro-002", "gemini-1.5-pro-latest", "gemini-1.5-pro", "gemini-pro"
    ]
    for m in priority_list:
        if m in all_models: return m
    return "gemini-1.5-flash"

def get_best_flash_model(all_models):
    priority_list = [
        "gemini-1.5-flash-002", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-flash-8b"
    ]
    for m in priority_list:
        if m in all_models: return m
    return get_best_pro_model(all_models)

MODEL_NAME_PRO = get_best_pro_model(ALL_MODELS)
MODEL_NAME_FLASH = get_best_flash_model(ALL_MODELS)

try:
    model_pro = genai.GenerativeModel(MODEL_NAME_PRO)
    model_flash = genai.GenerativeModel(MODEL_NAME_FLASH)
    model_vision = genai.GenerativeModel(MODEL_NAME_PRO)
except Exception as e:
    st.error(f"❌ モデル起動エラー: {e}")
    st.stop()

# ---------------------------------------------------------
# 💾 データ管理 & 圧縮・復元ロジック
# ---------------------------------------------------------
if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'clean_df' not in st.session_state: st.session_state['clean_df'] = pd.DataFrame()
if 'category_map' not in st.session_state: st.session_state['category_map'] = {}
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}

def compress_data_to_code(data_dict):
    try:
        json_str = json.dumps(data_dict, ensure_ascii=False)
        compressed = gzip.compress(json_str.encode('utf-8'))
        b64_str = base64.b64encode(compressed).decode('utf-8')
        return b64_str
    except: return None

def decompress_code_to_data(b64_str):
    try:
        compressed = base64.b64decode(b64_str)
        json_str = gzip.decompress(compressed).decode('utf-8')
        return json.loads(json_str)
    except: return None

FIXED_CATEGORIES = {
    "国語": ["漢字", "文法", "評論", "古文", "その他"],
    "数学": ["正負の数・文字と式", "一次方程式・連立方程式", "平方根", "式の展開と因数分解", "二次方程式", "比例・反比例", "一次関数", "関数y=ax^2", "平面図形", "空間図形", "図形の性質と証明", "確率・統計", "融合問題", "その他"],
    "英語": ["リスニング", "和訳", "英訳", "英作文", "文法", "読解", "融合問題", "その他"],
    "理科": ["【物理】光・音・力", "【物理】電流と磁界", "【物理】運動とエネルギー", "【化学】物質・気体・水溶液", "【化学】化学変化と原子・分子", "【化学】イオン・電池", "【生物】植物", "【生物】動物", "【生物】遺伝・細胞", "【地学】大地", "【地学】気象", "【地学】宇宙", "融合問題", "その他"],
    "社会": ["【地理】世界", "【地理】日本", "【歴史】古代～中世", "【歴史】近世", "【歴史】近代", "【歴史】現代", "【公民】政治", "【公民】経済", "【公民】国際", "融合問題", "その他"]
}

# ---------------------------------------------------------
# 🛠️ 関数定義
# ---------------------------------------------------------
def ask_gemini_robust(prompt, image_list=None, use_flash=False):
    target_model = model_vision if image_list else (model_flash if use_flash else model_pro)
    try:
        if image_list: response = target_model.generate_content([prompt] + image_list)
        else: response = target_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"エラー: {e}"

def detect_subject(file_name):
    name_str = str(file_name)
    for sub in ['数学', '英語', '理科', '社会', '国語']:
        if sub in name_str: return sub
    return 'その他'

def parse_csv(file):
    try:
        file.seek(0)
        try: df = pd.read_csv(file, header=None)
        except: 
            file.seek(0)
            df = pd.read_csv(file, header=None, encoding='cp932')
        
        header_row_mask = df.apply(lambda r: r.astype(str).str.contains('大問|内容').any(), axis=1)
        if len(df[header_row_mask]) > 0:
            idx = df[header_row_mask].index[0]
            col_idx = 0
            for c in df.columns:
                val = str(df.iloc[idx][c])
                if '大問' in val or '内容' in val:
                    col_idx = c
                    break
            
            subset = df.iloc[idx:, col_idx:].reset_index(drop=True).T
            subset.columns = [str(val).strip() for val in subset.iloc[0]]
            subset = subset[1:]
            
            if '大問' in subset.columns: subset = subset.dropna(subset=['大問'])
            subset['点数'] = pd.to_numeric(subset['点数'], errors='coerce').fillna(0)
            subset['配点'] = pd.to_numeric(subset['配点'], errors='coerce').fillna(0)
            subset['ファイル名'] = str(file.name)
            subset['教科'] = detect_subject(file.name)
            
            if '点数' in subset.columns: return subset
    except: pass
    return None

def process_and_categorize():
    if not st.session_state['data_store']:
        st.session_state['clean_df'] = pd.DataFrame()
        return

    with st.status(f"🚀 データ解析中...", expanded=True) as status:
        raw_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
        unique_pairs = raw_df[['教科', '内容']].drop_duplicates()
        unknown_list = []
        for _, row in unique_pairs.iterrows():
            subj = row['教科']
            topic = str(row['内容']).strip()
            is_perfect = False
            if subj in FIXED_CATEGORIES and topic in FIXED_CATEGORIES[subj]: is_perfect = True
            
            if not is_perfect and (subj, topic) not in st.session_state['category_map']:
                unknown_list.append(f"{subj}: {topic}")
        
        if unknown_list:
            categories_str = json.dumps(FIXED_CATEGORIES, ensure_ascii=False)
            prompt = f"""
            入試分析の専門家として、以下の「教科: 単元名」を【定義済みマスタ】のカテゴリに分類し、
            JSON形式 `{{"教科: 単元名": "定義カテゴリ", ...}}` で出力して。
            マスタ: {categories_str}
            データ: {"\n".join(unknown_list)}
            """
            response = ask_gemini_robust(prompt, use_flash=False)
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    mapping = json.loads(json_match.group())
                    for k, v in mapping.items():
                        if ':' in k:
                            s, t = k.split(':', 1)
                            st.session_state['category_map'][(s.strip(), t.strip())] = v.strip()
            except: pass

        df_clean = raw_df.copy()
        if '詳細' not in df_clean.columns: df_clean['詳細'] = df_clean['内容']
        def apply_mapping(row):
            key = (row['教科'], str(row['内容']).strip())
            mapped = st.session_state['category_map'].get(key, row['内容'])
            return mapped if mapped else row['内容']

        df_clean['内容'] = df_clean.apply(apply_mapping, axis=1)
        st.session_state['clean_df'] = df_clean
        status.update(label="完了", state="complete", expanded=False)

# ---------------------------------------------------------
# 🖥️ サイドバー (設定・同期)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 設定 & 同期")
    
    with st.expander("📲 データ移行 (スマホへ送る)"):
        tab_s1, tab_s2 = st.tabs(["保存", "復元"])
        with tab_s1:
            if st.session_state['data_store']:
                backup = {
                    'textbooks': st.session_state['textbooks'],
                    'data_store': {n: d.to_json(orient='split') for n, d in st.session_state['data_store'].items()},
                    'category_map': {f"{k[0]}:{k[1]}": v for k,v in st.session_state['category_map'].items()}
                }
                code = compress_data_to_code(backup)
                st.code(code, language="text")
                st.caption("このコードをコピーしてスマホで読み込んでください。")
        with tab_s2:
            inp = st.text_area("セーブコード貼付")
            if st.button("復元"):
                d = decompress_code_to_data(inp.strip())
                if d:
                    if 'textbooks' in d: st.session_state['textbooks'] = d['textbooks']
                    if 'data_store' in d:
                        st.session_state['data_store'] = {n: pd.read_json(j, orient='split') for n, j in d['data_store'].items()}
                    if 'category_map' in d:
                        st.session_state['category_map'] = {(k.split(':')[0], k.split(':')[1]): v for k, v in d['category_map'].items()}
                    st.session_state['clean_df'] = pd.DataFrame()
                    st.success("復元完了！")
                    time.sleep(1)
                    st.rerun()

    st.markdown("---")
    st.markdown("##### 📚 使用参考書")
    with st.form("books"):
        for sub in ["数学","英語","理科","社会","国語"]:
            val = st.session_state['textbooks'].get(sub, "")
            st.session_state['textbooks'][sub] = st.text_input(sub, value=val, placeholder="例: 自由自在")
        if st.form_submit_button("保存"): st.rerun()

    st.markdown("---")
    if st.button("🚨 データ全削除", type="primary"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------
# 📂 メインコンテンツ
# ---------------------------------------------------------
st.markdown("## 🏔️ 新潟高校 合格ナビ")

# --- ファイルアップロード (カードUI) ---
with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("**📂 成績データのアップロード (CSV)**")
        uploaded_files = st.file_uploader("", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")
    with c2:
        st.markdown("&nbsp;")
        if st.button("🚀 解析スタート", type="primary"):
            if uploaded_files:
                for f in uploaded_files:
                    df = parse_csv(f)
                    if df is not None: st.session_state['data_store'][f.name] = df
                process_and_categorize()
                st.rerun()
            elif st.session_state['data_store']:
                process_and_categorize()
                st.rerun()
            else:
                st.warning("ファイルを選択してね")

if not st.session_state['clean_df'].empty:
    df_show = st.session_state['clean_df']
    
    # タブメニュー
    tab1, tab2, tab3, tab4 = st.tabs(["📊 全体分析", "📖 弱点攻略", "📅 合格計画", "📷 画像指導"])

    # --- TAB 1: 分析 ---
    with tab1:
        st.markdown("### 📊 現在の実力分析")
        summary = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率'] = (summary['点数'] / summary['配点'] * 100).fillna(0)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container():
                st.markdown("##### ⚠️ 優先して復習すべき単元 (ワースト10)")
                worst_10 = summary.sort_values('得点率').head(10)
                st.dataframe(
                    worst_10[['教科', '内容', '得点率']], 
                    column_config={"得点率": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)},
                    use_container_width=True, hide_index=True
                )
        with col2:
            with st.container():
                st.markdown("##### 教科別平均点")
                sub_sum = df_show.groupby('教科')[['点数', '配点']].sum().reset_index()
                sub_sum['率'] = (sub_sum['点数']/sub_sum['配点']*100)
                st.dataframe(
                    sub_sum[['教科', '率']], 
                    column_config={"率": st.column_config.NumberColumn(format="%.1f%%")},
                    hide_index=True, use_container_width=True
                )

    # --- TAB 2: 復習 (QB風レイアウト) ---
    with tab2:
        st.markdown("### 📖 AI家庭教師の弱点攻略")
        
        # 選択エリア
        with st.container():
            c1, c2 = st.columns(2)
            sel_sub = c1.selectbox("教科を選択", summary['教科'].unique())
            sel_top = c2.selectbox("単元を選択", summary[summary['教科']==sel_sub].sort_values('得点率')['内容'])
        
        # データ取得
        target_rows = df_show[(df_show['教科']==sel_sub) & (df_show['内容']==sel_top)]
        rate = (target_rows['点数'].sum() / target_rows['配点'].sum() * 100).round(1)
        details = "、".join(target_rows['詳細'].unique().tolist())
        book = st.session_state['textbooks'].get(sel_sub, "指定なし")

        # アクションカード
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if st.button("💡 復習ポイントを聞く", use_container_width=True):
                with st.spinner("AIが分析中..."):
                    p = f"新潟高校入試志望。{sel_sub}の「{sel_top}」(詳細:{details})が得点率{rate}%。参考書「{book}」を使ってどう復習すべき？簡潔に3点。"
                    st.session_state['guide'] = ask_gemini_robust(p)
        
        with col_act2:
            if st.button("📝 実践問題に挑戦 (QBモード)", use_container_width=True, type="primary"):
                with st.spinner("問題作成中..."):
                    p2 = f"""
                    新潟高校入試レベル。教科:{sel_sub}, 分野:{sel_top}。
                    実践的な問題を1問作成して。
                    形式: 
                    ## 問題
                    (問題文)
                    ## 解説
                    (正解と詳しい解説)
                    """
                    st.session_state['test'] = ask_gemini_robust(p2)

        # 結果表示エリア
        if 'guide' in st.session_state:
            st.info(st.session_state['guide'], icon="💡")
            
        if 'test' in st.session_state:
            # QB風の問題表示
            content = st.session_state['test']
            try:
                q_part, a_part = content.split("## 解説")
                a_part = "## 解説" + a_part
            except:
                q_part = content
                a_part = "解説が生成されませんでした。"

            st.markdown("---")
            st.markdown(f"""
            <div style="background-color:#e8f4fd; padding:20px; border-radius:10px; border-left:5px solid #007bff; margin-bottom:20px;">
                <h4 style="color:#007bff; margin-top:0;">Q. 実践問題</h4>
                <div style="font-size:1.1em; line-height:1.6;">{q_part.replace('## 問題', '')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("👉 解答・解説を見る"):
                st.markdown(a_part)

    # --- TAB 3: 計画 ---
    with tab3:
        st.markdown("### 📅 合格カレンダー")
        if st.button("スケジュールを再生成", use_container_width=True):
            with st.spinner("スケジュール立案中..."):
                prompt = f"今日({today})から入試({exam_date})までの新潟高校合格スケジュール。弱点教科を中心に。"
                res = ask_gemini_robust(prompt)
                st.markdown(res)

    # --- TAB 4: 画像 ---
    with tab4:
        st.markdown("### 📷 ノート/答案の添削")
        i1 = st.file_uploader("問題", type=['jpg','png'])
        i2 = st.file_uploader("自分の答案", type=['jpg','png'])
        if i1 and i2 and st.button("添削開始", type="primary"):
            with st.spinner("解析中..."):
                res = ask_gemini_robust("新潟高校志望。この答案を採点し、改善点を指導して。", [PIL.Image.open(i1), PIL.Image.open(i2)])
                st.markdown(res)

else:
    # データがない時のガイド
    st.info("👆 上のボックスにCSVファイル（成績データ）を入れて「解析スタート」を押してください。")
