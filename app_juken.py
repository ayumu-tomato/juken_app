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
# 🔐 初期設定
# ==========================================
st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide", page_icon="🏔️")

# --------------------------------------------------------------------------------
# 🎨 UIデザイン & CSS
# --------------------------------------------------------------------------------
exam_date = datetime.date(2026, 3, 4) 
today = datetime.date.today()
days_left = (exam_date - today).days
if days_left < 0: days_left = 0

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Noto Sans JP', sans-serif;
        color: #333;
    }}
    .stApp {{ background-color: #f4f7f6; }}

    /* 固定カウントダウン */
    .fixed-countdown {{
        position: fixed;
        top: 0;
        right: 0;
        z-index: 999999;
        background: rgba(255, 255, 255, 0.95);
        border-bottom-left-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        padding: 8px 16px;
        text-align: right;
        border-left: 5px solid #007bff;
        line-height: 1.2;
    }}
    .count-label {{ font-size: 10px; color: #666; display: block; font-weight: bold; }}
    .count-number {{ font-size: 20px; font-weight: 800; color: #d9534f; }}
    @media (max-width: 640px) {{
        .fixed-countdown {{ top: 40px; padding: 5px 10px; }}
        .count-number {{ font-size: 16px; }}
    }}

    /* QB風カード */
    div[data-testid="stVerticalBlock"] > div:has(div.stDataFrame), 
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {{
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        margin-bottom: 10px;
    }}

    h1 {{
        color: #007bff;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px !important;
    }}
    
    .stButton > button {{
        width: 100%;
        border-radius: 30px;
        font-weight: bold;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.1s;
    }}
    .stButton > button:active {{ transform: scale(0.98); }}
    button[kind="primary"] {{ background-color: #007bff !important; color: white !important; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
    }}
    .stTabs [aria-selected="true"] {{ background-color: #007bff !important; color: white !important; }}
</style>

<div class="fixed-countdown">
    <span class="count-label">新潟高校入試まで</span>
    <span class="count-number">あと {days_left} 日</span>
</div>
""", unsafe_allow_html=True)

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

if not api_key:
    st.warning("⚠️ SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 🤖 モデル自動検出
# ---------------------------------------------------------
def get_available_models():
    try:
        return [m.name.replace("models/", "") for m in genai.list_models()]
    except:
        return []

ALL_MODELS = get_available_models()

def get_best_pro_model(all_models):
    priority_list = [
        "gemini-3-pro", "gemini-3-pro-preview", "gemini-3.0-pro",
        "gemini-2.5-pro", "gemini-2.0-pro-exp",
        "gemini-1.5-pro-002", "gemini-1.5-pro-latest", "gemini-1.5-pro", "gemini-pro"
    ]
    for m in priority_list:
        if m in all_models: return m
    
    pro_models = [m for m in all_models if "pro" in m and "vision" not in m]
    if pro_models:
        pro_models.sort(reverse=True)
        return pro_models[0]
            
    return "gemini-1.5-flash"

def get_best_flash_model(all_models):
    priority_list = [
        "gemini-2.5-flash", "gemini-2.5-flash-001", "gemini-2.0-flash", "gemini-2.0-flash-exp",
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
    st.sidebar.success(f"🚀 Engine: {MODEL_NAME_PRO}")
except Exception as e:
    st.error(f"❌ モデル起動エラー: {e}")
    st.stop()

# ---------------------------------------------------------
# 💾 データ管理 (保存ロジック修正版)
# ---------------------------------------------------------
if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'clean_df' not in st.session_state: st.session_state['clean_df'] = pd.DataFrame()
if 'category_map' not in st.session_state: st.session_state['category_map'] = {}
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}

def compress_data_to_code(data_dict):
    """データを圧縮して文字列化する（エラーハンドリング強化）"""
    try:
        # default=str を追加して、日付データやNumpy型があっても強制的に文字列化する
        json_str = json.dumps(data_dict, ensure_ascii=False, default=str)
        compressed = gzip.compress(json_str.encode('utf-8'))
        b64_str = base64.b64encode(compressed).decode('utf-8')
        return b64_str
    except Exception as e:
        st.error(f"⚠️ データ保存コード生成中にエラーが発生しました: {e}")
        return None

def decompress_code_to_data(b64_str):
    """文字列をデータに戻す"""
    try:
        compressed = base64.b64decode(b64_str)
        json_str = gzip.decompress(compressed).decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        st.error(f"⚠️ データ復元中にエラーが発生しました: {e}")
        return None

FIXED_CATEGORIES = {
    "国語": ["漢字", "文法", "評論", "古文", "その他"],
    "数学": ["正負の数・文字と式", "一次方程式・連立方程式", "平方根", "式の展開と因数分解", "二次方程式", "比例・反比例", "一次関数", "関数y=ax^2", "平面図形（作図・移動・おうぎ形）", "空間図形", "図形の性質と証明（合同・相似・円）", "確率・統計（データの活用・三平方の定理）", "融合問題", "その他"],
    "英語": ["リスニング", "和訳", "英訳", "英作文", "文法", "読解", "融合問題", "その他"],
    "理科": ["【物理】光・音・力", "【物理】電流と磁界", "【物理】運動とエネルギー", "【化学】身の回りの物質・気体・水溶液", "【化学】化学変化と原子・分子", "【化学】化学変化とイオン・電池", "【生物】植物の生活と種類", "【生物】動物の生活と生物の変遷", "【生物】生命の連続性（遺伝・細胞）", "【地学】大地の変化（火山・地震・地層）", "【地学】気象とその変化", "【地学】地球と宇宙", "融合問題", "その他"],
    "社会": ["【地理】世界の姿・気候・生活文化", "【地理】世界の諸地域", "【地理】日本の姿・産業・資源エネルギー", "【地理】日本の諸地域", "【歴史】古代〜中世（文明〜室町）", "【歴史】近世（安土桃山・江戸）", "【歴史】近代①（明治〜第一次大戦）", "【歴史】近代②〜現代（昭和〜現在）", "【公民】現代社会・日本国憲法・人権", "【公民】政治の仕組み", "【公民】経済の仕組み", "【公民】国際社会・環境問題", "融合問題", "その他"]
}

# ---------------------------------------------------------
# 🛠️ 関数定義
# ---------------------------------------------------------
def ask_gemini_robust(prompt, image_list=None, use_flash=False):
    max_retries = 3
    if image_list: target_model = model_vision
    elif use_flash: target_model = model_flash
    else: target_model = model_pro

    for attempt in range(max_retries):
        try:
            if image_list: response = target_model.generate_content([prompt] + image_list)
            else: response = target_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                st.toast(f"⏳ 待機中 ({attempt+1}/3)")
                time.sleep((attempt + 1) * 3)
            else: return f"エラー: {e}"
    return "❌ 応答できませんでした。"

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
            
            raw_cols = [str(val).strip() for val in subset.iloc[0]]
            new_cols = []
            seen = {}
            for c in raw_cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            subset.columns = new_cols
            
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

    model_label = MODEL_NAME_PRO 
    
    with st.status(f"🚀 データを解析中... (Engine: {model_label})", expanded=True) as status:
        st.write("📂 データを結合中...")
        try:
            raw_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
        except Exception as e:
            st.error(f"データ結合エラー: {e}")
            st.warning("一部のファイルの形式が不正な可能性があります。「全データを削除」してやり直してください。")
            status.update(label="⚠️ エラー発生", state="error")
            return

        time.sleep(0.1)
        
        st.write("🔍 未知の単元を検索中...")
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
            st.write(f"🧠 {len(unknown_list)} 件の単元をAIが思考・分類中...")
            categories_str = json.dumps(FIXED_CATEGORIES, ensure_ascii=False, indent=2)
            prompt = f"""
            入試データ分析の専門家として振る舞ってください。
            入力された「教科: 元の単元名」を分析し、以下の【定義済みマスタ】の中で最も適切なカテゴリに分類してください。
            【定義済みマスタ】
            {categories_str}
            【入力データ】
            """ + "\n".join(unknown_list) + """
            【出力形式】
            JSON形式の辞書 `{ "教科: 元の単元名": "定義済みカテゴリ名", ... }` のみを出力してください。
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
            except: st.warning("一部の分類に失敗")

        st.write("💾 保存中...")
        df_clean = raw_df.copy()
        if '詳細' not in df_clean.columns: df_clean['詳細'] = df_clean['内容']
        def apply_mapping(row):
            key = (row['教科'], str(row['内容']).strip())
            mapped = st.session_state['category_map'].get(key, row['内容'])
            return mapped if mapped else row['内容']

        df_clean['内容'] = df_clean.apply(apply_mapping, axis=1)
        st.session_state['clean_df'] = df_clean
        status.update(label="✅ 完了！", state="complete", expanded=False)

def get_status_emoji(rate):
    if rate <= 50: return "🔴"
    elif rate <= 70: return "🟡"
    else: return "🟢"

# ---------------------------------------------------------
# 🖥️ サイドバー設定 (修正: 安全な保存・復元)
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📲 簡単データ移行")
    st.caption("別のデバイスに移る時は「セーブコード」を使うと便利です。")
    
    sync_tab1, sync_tab2 = st.tabs(["📤 保存(コピー)", "📥 復元(貼付)"])
    
    with sync_tab1:
        if st.session_state['data_store'] or st.session_state['textbooks']:
            
            # --- 【修正】Category Mapのキー(タプル)を文字列に安全変換 ---
            safe_category_map = {}
            for k, v in st.session_state['category_map'].items():
                try:
                    if isinstance(k, (list, tuple)) and len(k) >= 2:
                        safe_category_map[f"{k[0]}:{k[1]}"] = v
                    else:
                        safe_category_map[str(k)] = v
                except: continue

            # --- 【修正】DataFrameをJSON化 (日付等に対応) ---
            safe_data_store = {}
            for name, df in st.session_state['data_store'].items():
                try:
                    safe_data_store[name] = df.to_json(orient='split', force_ascii=False, date_format='iso')
                except:
                    pass

            backup_data = {
                'textbooks': st.session_state['textbooks'],
                'data_store': safe_data_store,
                'category_map': safe_category_map
            }
            
            save_code = compress_data_to_code(backup_data)
            
            if save_code:
                st.info("👇 このコードをコピーして、LINEやメモ帳でスマホに送ってください。")
                st.code(save_code, language="text")
            else:
                st.warning("保存コードの生成に失敗しました。（エラー詳細は画面上部）")
        else:
            st.caption("データがありません")

    with sync_tab2:
        input_code = st.text_area("ここにセーブコードを貼り付け:", height=100)
        if st.button("復元を実行"):
            if input_code:
                restored_data = decompress_code_to_data(input_code.strip())
                if restored_data:
                    try:
                        # 教材データの復元
                        if 'textbooks' in restored_data: st.session_state['textbooks'] = restored_data['textbooks']
                        
                        # 成績データの復元
                        if 'data_store' in restored_data:
                            st.session_state['data_store'] = {}
                            for name, df_json in restored_data['data_store'].items():
                                st.session_state['data_store'][name] = pd.read_json(df_json, orient='split')
                        
                        # カテゴリマップの復元（文字列 "数:関" → タプル ("数","関") に戻す）
                        if 'category_map' in restored_data:
                            st.session_state['category_map'] = {}
                            for k, v in restored_data['category_map'].items():
                                if ':' in k:
                                    s, t = k.split(':', 1)
                                    st.session_state['category_map'][(s, t)] = v
                                else:
                                    # 万が一フォーマットが違う場合
                                    st.session_state['category_map'][(k, k)] = v
                        
                        st.session_state['clean_df'] = pd.DataFrame() 
                        st.success("✅ 復元完了！画面を更新します。")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"復元処理中にエラー: {e}")
                else:
                    st.error("コードが間違っているか、壊れています。")

    st.markdown("---")
    st.subheader("📚 登録済み参考書")
    if st.session_state['textbooks']:
        for subj, book in list(st.session_state['textbooks'].items()):
            if book:
                c1, c2 = st.columns([0.8, 0.2])
                c1.write(f"**{subj}**: {book}")
                if c2.button("🗑️", key=f"del_book_{subj}"):
                    del st.session_state['textbooks'][subj]
                    st.rerun()
    
    with st.expander("追加・編集"):
        with st.form("textbook_form"):
            tb_math = st.text_input("数学", value=st.session_state['textbooks'].get('数学', ''))
            tb_eng = st.text_input("英語", value=st.session_state['textbooks'].get('英語', ''))
            tb_sci = st.text_input("理科", value=st.session_state['textbooks'].get('理科', ''))
            tb_soc = st.text_input("社会", value=st.session_state['textbooks'].get('社会', ''))
            tb_jpn = st.text_input("国語", value=st.session_state['textbooks'].get('国語', ''))
            if st.form_submit_button("保存"):
                st.session_state['textbooks'] = {'数学': tb_math, '英語': tb_eng, '理科': tb_sci, '社会': tb_soc, '国語': tb_jpn}
                st.rerun()

    st.markdown("---")
    st.subheader("💾 登録済みファイル")
    if st.session_state['data_store']:
        for file_name in list(st.session_state['data_store'].keys()):
            c1, c2 = st.columns([0.85, 0.15])
            c1.text(file_name)
            if c2.button("🗑️", key=f"del_file_{file_name}"):
                del st.session_state['data_store'][file_name]
                st.session_state['clean_df'] = pd.DataFrame()
                st.rerun()
        
        if st.button("🚨 全データを削除", type="primary"):
            st.session_state['data_store'] = {}
            st.session_state['clean_df'] = pd.DataFrame()
            st.session_state['category_map'] = {}
            st.rerun()
    else:
        st.info("ファイルなし")

# ---------------------------------------------------------
# 📂 メイン画面
# ---------------------------------------------------------
st.markdown("### 1️⃣ データのアップロード & 解析")
col_up, col_btn = st.columns([3, 1])

with col_up:
    uploaded_files = st.file_uploader("CSVファイル", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")

with col_btn:
    if st.button("🚀 AI解析を実行", type="primary", use_container_width=True):
        if uploaded_files:
            new_count = 0
            for file in uploaded_files:
                df = parse_csv(file)
                if df is not None:
                    st.session_state['data_store'][file.name] = df
                    new_count += 1
            if new_count > 0:
                process_and_categorize()
            else:
                st.warning("有効なCSVがありません")
        elif st.session_state['data_store']:
            process_and_categorize()
        else:
            st.warning("ファイルを選択してください")

if not st.session_state['clean_df'].empty:
    df_show = st.session_state['clean_df']
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📅 合格計画", "📷 画像採点"])

    with tab1:
        # データ集計
        summary = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).fillna(0).round(1)
        
        # 判定カラムの追加 (🔴/🟡/🟢)
        summary['判定'] = summary['得点率(%)'].apply(get_status_emoji)
        
        # 表示用データの整理
        summary_clean = summary[['教科', '内容', '判定', '得点率(%)']].copy()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("⚠️ 全体：優先復習単元")
            st.dataframe(
                summary_clean.sort_values('得点率(%)').head(10), 
                column_config={
                    "得点率(%)": st.column_config.ProgressColumn(
                        "得点率", 
                        format="%.1f%%", 
                        min_value=0, 
                        max_value=100
                    ),
                    "判定": st.column_config.TextColumn("状態", width="small")
                },
                use_container_width=True, 
                hide_index=True
            )
        with col2:
            st.subheader("教科別平均")
            sub_sum = df_show.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率(%)'] = (sub_sum['点数']/sub_sum['配点']*100).fillna(0).round(1)
            st.dataframe(
                sub_sum[['教科', '得点率(%)']], 
                column_config={
                    "得点率(%)": st.column_config.ProgressColumn(
                        "平均点率",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    )
                },
                hide_index=True
            )
            
        st.markdown("---")
        st.subheader("📚 教科ごとの弱点")
        subjects = df_show['教科'].unique()
        cols = st.columns(len(subjects)) if len(subjects) > 0 else [st.container()]
        
        for i, sub in enumerate(subjects):
            with cols[i]:
                st.markdown(f"**{sub}**")
                sub_df = summary_clean[summary_clean['教科'] == sub].sort_values('得点率(%)').head(5)
                if not sub_df.empty:
                    st.dataframe(
                        sub_df[['内容', '判定', '得点率(%)']], 
                        column_config={
                            "得点率(%)": st.column_config.ProgressColumn(
                                format="%.0f%%", min_value=0, max_value=100
                            ),
                            "判定": st.column_config.TextColumn(width="small")
                        },
                        use_container_width=True, 
                        hide_index=True
                    )
                else: st.caption("データなし")

    with tab2:
        st.subheader("AI家庭教師による指導")
        
        # データ準備
        summary_t2 = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary_t2['得点率(%)'] = (summary_t2['点数'] / summary_t2['配点'] * 100).fillna(0).round(1)

        c1, c2 = st.columns(2)
        with c1: 
            sel_sub = st.selectbox("教科", summary_t2['教科'].unique())
        
        with c2:
            sub_topics = summary_t2[summary_t2['教科']==sel_sub].sort_values('得点率(%)')
            
            topic_map = {}
            for _, row in sub_topics.iterrows():
                icon = get_status_emoji(row['得点率(%)'])
                display_name = f"{icon} {row['内容']} ({row['得点率(%)']}%)"
                topic_map[display_name] = row['内容']
            
            sel_top_display = st.selectbox("単元 (🔴苦手 / 🟡注意 / 🟢定着)", options=list(topic_map.keys()))
            sel_top = topic_map[sel_top_display]
        
        target_rows = df_show[(df_show['教科']==sel_sub) & (df_show['内容']==sel_top)]
        rate = (target_rows['点数'].sum() / target_rows['配点'].sum() * 100).round(1)
        original_topics = target_rows['詳細'].unique().tolist()
        original_topics_str = "、".join([str(t) for t in original_topics])
        
        st.info(f"選択単元: **{sel_top}** (得点率: {rate}%)")
        st.caption(f"詳細: {original_topics_str}")
        book = st.session_state['textbooks'].get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.status(f"🤖 AI({MODEL_NAME_PRO})が思考中...", expanded=True) as status:
                st.write("1. 分析中...")
                p = f"新潟高校志望。教科: {sel_sub}, 苦手カテゴリ: {sel_top}（詳細: {original_topics_str}）, 得点率: {rate}%, 参考書: {book}。復習ポイントとチェック項目3つを教えて。"
                res = ask_gemini_robust(p, use_flash=False)
                st.session_state['guide'] = res
                status.update(label="✅ 完了！", state="complete", expanded=False)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストを作成"):
                with st.status("📝 問題作成中...", expanded=True) as status:
                    p2 = f"新潟高校入試レベル。{sel_sub}の「{sel_top}」（詳細: {original_topics_str}）の実践問題1問。解答解説付き。"
                    res = ask_gemini_robust(p2, use_flash=False)
                    st.session_state['test'] = res
                    status.update(label="✅ 完了！", state="complete", expanded=False)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])

    with tab3:
        if st.button("合格スケジュール作成"):
            with st.status("📅 スケジュールを立案中...", expanded=True) as status:
                st.write("1. 弱点を抽出中...")
                summary = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
                summary['得点率'] = (summary['点数'] / summary['配点'] * 100)
                weak_points = summary.sort_values('得点率').head(5)
                weak_str = ""
                for _, row in weak_points.iterrows():
                    weak_str += f"- {row['教科']}: {row['内容']} (得点率{row['得点率']:.1f}%)\n"
                
                st.write("2. カリキュラム構築中...")
                prompt = f"""
                今日({datetime.date.today()})から入試({datetime.date(2026, 3, 4)})までの新潟高校合格スケジュール。
                【特に苦手な分野】
                {weak_str}
                具体的な対策を含めて作成してください。
                """
                res = ask_gemini_robust(prompt, use_flash=False)
                st.markdown(res)
                status.update(label="✅ 完成！", state="complete", expanded=False)

    with tab4:
        st.subheader("📷 画像採点＆指導")
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1: img_prob = st.file_uploader("① 問題画像", type=['png', 'jpg', 'jpeg'])
        with col_img2: img_user = st.file_uploader("② 解答画像", type=['png', 'jpg', 'jpeg'])
        with col_img3: img_ans = st.file_uploader("③ 模範解答画像", type=['png', 'jpg', 'jpeg'])
        
        if img_prob and img_user and img_ans:
            if st.button(f"🚀 採点実行 ({MODEL_NAME_PRO})"):
                with st.status("👀 解析中...", expanded=True) as status:
                    images = [PIL.Image.open(img_prob), PIL.Image.open(img_user), PIL.Image.open(img_ans)]
                    prompt_v = "新潟高校志望。3枚の画像から、厳密な採点、添削、弱点分析、類題の提示を行ってください。"
                    res = ask_gemini_robust(prompt_v, images)
                    st.markdown(res)
                    status.update(label="✅ 完了！", state="complete", expanded=False)
else:
    st.info("👆 上記からCSVをアップロードし、「AI解析を実行」ボタンを押してください。")
