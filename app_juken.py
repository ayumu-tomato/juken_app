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
# 音声生成用ライブラリ
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

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

    /* カードデザイン */
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
    
    /* 反省コメント */
    .reflection-box {{
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-size: 0.9em;
    }}
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
# 🤖 モデル設定
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
except: pass

# ---------------------------------------------------------
# 💾 データ管理
# ---------------------------------------------------------
if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'clean_df' not in st.session_state: st.session_state['clean_df'] = pd.DataFrame()
if 'category_map' not in st.session_state: st.session_state['category_map'] = {}
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}
# 特訓モード用セッション
if 'practice_q' not in st.session_state: st.session_state['practice_q'] = None
if 'practice_a' not in st.session_state: st.session_state['practice_a'] = None
if 'practice_script' not in st.session_state: st.session_state['practice_script'] = None

def compress_data_to_code(data_dict):
    try:
        json_str = json.dumps(data_dict, ensure_ascii=False, default=str)
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')
    except: return None

def decompress_code_to_data(b64_str):
    try:
        compressed = base64.b64decode(b64_str)
        return json.loads(gzip.decompress(compressed).decode('utf-8'))
    except: return None

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
                time.sleep((attempt + 1) * 3)
            else: return f"エラー: {e}"
    return "❌ 応答できませんでした。"

def text_to_speech(text, lang='en'):
    """gTTSで音声を生成"""
    if gTTS is None: return None
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except: return None

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
                if c in seen: seen[c]+=1; new_cols.append(f"{c}_{seen[c]}")
                else: seen[c]=0; new_cols.append(c)
            subset.columns = new_cols
            subset = subset[1:]
            if '大問' in subset.columns: subset = subset.dropna(subset=['大問'])
            subset['点数'] = pd.to_numeric(subset['点数'], errors='coerce').fillna(0)
            subset['配点'] = pd.to_numeric(subset['配点'], errors='coerce').fillna(0)
            subset['ファイル名'] = str(file.name)
            # 教科判定
            name_str = str(file.name)
            subj = 'その他'
            for s in ['数学','英語','理科','社会','国語']:
                if s in name_str: subj=s
            subset['教科'] = subj
            if '反省' in subset.columns: subset['反省'] = subset['反省'].fillna("").astype(str)
            if '点数' in subset.columns: return subset
    except: pass
    return None

def process_and_categorize():
    if not st.session_state['data_store']:
        st.session_state['clean_df'] = pd.DataFrame()
        return
    with st.status(f"🚀 解析中... (Engine: {MODEL_NAME_PRO})", expanded=True) as status:
        try:
            raw_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
        except: return
        
        unique_pairs = raw_df[['教科', '内容']].drop_duplicates()
        unknown_list = []
        for _, row in unique_pairs.iterrows():
            subj = row['教科']
            topic = str(row['内容']).strip()
            if subj in FIXED_CATEGORIES and topic in FIXED_CATEGORIES[subj]: continue
            if (subj, topic) not in st.session_state['category_map']:
                unknown_list.append(f"{subj}: {topic}")
        
        if unknown_list:
            categories_str = json.dumps(FIXED_CATEGORIES, ensure_ascii=False)
            prompt = f"「教科:単元」を分析し、最も適切なカテゴリをJSON辞書で出力せよ。\nマスタ: {categories_str}\n入力: {unknown_list}"
            response = ask_gemini_robust(prompt, use_flash=False)
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    mapping = json.loads(json_match.group())
                    for k, v in mapping.items():
                        if ':' in k: s, t = k.split(':', 1); st.session_state['category_map'][(s.strip(), t.strip())] = v.strip()
            except: pass

        df_clean = raw_df.copy()
        if '詳細' not in df_clean.columns: df_clean['詳細'] = df_clean['内容']
        if '反省' not in df_clean.columns: df_clean['反省'] = ""
        def apply_mapping(row):
            key = (row['教科'], str(row['内容']).strip())
            return st.session_state['category_map'].get(key, row['内容'])
        df_clean['内容'] = df_clean.apply(apply_mapping, axis=1)
        st.session_state['clean_df'] = df_clean
        status.update(label="✅ 完了", state="complete", expanded=False)

def get_status_emoji(rate):
    if rate <= 50: return "🔴"
    elif rate <= 70: return "🟡"
    else: return "🟢"

# ---------------------------------------------------------
# 🖥️ サイドバー
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📲 データ管理")
    sync_tab1, sync_tab2 = st.tabs(["📤 保存", "📥 復元"])
    with sync_tab1:
        if st.session_state['data_store'] or st.session_state['textbooks']:
            safe_cat = {f"{k[0]}:{k[1]}" if isinstance(k, tuple) else str(k): v for k, v in st.session_state['category_map'].items()}
            safe_ds = {n: df.to_json(orient='split', force_ascii=False, date_format='iso') for n, df in st.session_state['data_store'].items()}
            backup = {'textbooks': st.session_state['textbooks'], 'data_store': safe_ds, 'category_map': safe_cat}
            code = compress_data_to_code(backup)
            if code:
                st.download_button("💾 ファイル保存", code, f"niigata_bk_{today}.txt", "text/plain", type="primary")
                with st.expander("コード表示"): st.code(code)
    with sync_tab2:
        up_file = st.file_uploader("ファイル", type=['txt'])
        up_text = st.text_area("コード")
        if st.button("復元"):
            target = up_file.read().decode() if up_file else up_text.strip()
            data = decompress_code_to_data(target)
            if data:
                try:
                    st.session_state['textbooks'] = data.get('textbooks', {})
                    st.session_state['category_map'] = {(k.split(':',1)[0], k.split(':',1)[1]) if ':' in k else (k,k): v for k, v in data.get('category_map', {}).items()}
                    st.session_state['data_store'] = {n: pd.read_json(j, orient='split') for n, j in data.get('data_store', {}).items()}
                    st.session_state['clean_df'] = pd.DataFrame()
                    st.rerun()
                except: st.error("復元失敗")
    
    st.divider()
    st.subheader("📚 参考書")
    for s, b in st.session_state['textbooks'].items():
        c1,c2=st.columns([8,2])
        c1.write(f"**{s}**: {b}"); 
        if c2.button("🗑️", key=f"d_{s}"): del st.session_state['textbooks'][s]; st.rerun()
    with st.expander("編集"):
        with st.form("tb"):
            tm=st.text_input("数学", st.session_state['textbooks'].get('数学',''))
            te=st.text_input("英語", st.session_state['textbooks'].get('英語',''))
            ts=st.text_input("理科", st.session_state['textbooks'].get('理科',''))
            tc=st.text_input("社会", st.session_state['textbooks'].get('社会',''))
            tj=st.text_input("国語", st.session_state['textbooks'].get('国語',''))
            if st.form_submit_button("保存"):
                st.session_state['textbooks']={'数学':tm,'英語':te,'理科':ts,'社会':tc,'国語':tj}
                st.rerun()
    
    st.divider()
    if st.button("🚨 全データ削除"):
        st.session_state['data_store']={}; st.session_state['clean_df']=pd.DataFrame(); st.session_state['practice_q']=None
        st.rerun()

# ---------------------------------------------------------
# 📂 メイン画面
# ---------------------------------------------------------
st.markdown("### 1️⃣ データのアップロード & 解析")
col_up, col_btn = st.columns([3, 1])
with col_up:
    uploaded_files = st.file_uploader("CSVファイル", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")
with col_btn:
    if st.button("🚀 AI解析", type="primary", use_container_width=True):
        if uploaded_files:
            for file in uploaded_files:
                df = parse_csv(file)
                if df is not None: st.session_state['data_store'][file.name] = df
            process_and_categorize()
        elif st.session_state['data_store']: process_and_categorize()
        else: st.warning("ファイルを選択してください")

if not st.session_state['clean_df'].empty:
    df_show = st.session_state['clean_df']
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📷 画像採点", "🧩 その他特訓"])

    # ------------------
    # TAB 1: 分析
    # ------------------
    with tab1:
        summary = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).fillna(0).round(1)
        summary['判定'] = summary['得点率(%)'].apply(get_status_emoji)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("⚠️ 優先復習単元")
            st.dataframe(summary.sort_values('得点率(%)').head(10)[['教科','内容','判定','得点率(%)']], use_container_width=True, hide_index=True)
        with c2:
            st.subheader("教科別平均")
            sub_sum = df_show.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率(%)'] = (sub_sum['点数']/sub_sum['配点']*100).fillna(0).round(1)
            st.dataframe(sub_sum[['教科','得点率(%)']], use_container_width=True, hide_index=True)

    # ------------------
    # TAB 2: 復習
    # ------------------
    with tab2:
        st.subheader("AI家庭教師")
        summary_t2 = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary_t2['得点率(%)'] = (summary_t2['点数']/summary_t2['配点']*100).fillna(0).round(1)
        
        c1, c2 = st.columns(2)
        sel_sub = c1.selectbox("教科", summary_t2['教科'].unique())
        
        sub_topics = summary_t2[summary_t2['教科']==sel_sub].sort_values('得点率(%)')
        topic_map = {f"{get_status_emoji(row['得点率(%)'])} {row['内容']} ({row['得点率(%)']}%)": row['内容'] for _, row in sub_topics.iterrows()}
        sel_top_d = c2.selectbox("単元", list(topic_map.keys()))
        sel_top = topic_map[sel_top_d]
        
        target_rows = df_show[(df_show['教科']==sel_sub) & (df_show['内容']==sel_top)]
        rate = (target_rows['点数'].sum() / target_rows['配点'].sum() * 100).round(1)
        reflections = [str(r) for r in target_rows['反省'].unique() if r and r!="nan"]
        ref_text = "\n".join([f"- {r}" for r in reflections]) if reflections else "特になし"
        
        st.info(f"得点率: {rate}%")
        if reflections: st.info(f"📝 反省メモ:\n{ref_text}")
        
        if st.button("① アドバイスを聞く"):
            prompt = f"新潟高校志望。{sel_sub}の「{sel_top}」について、得点率{rate}%。反省:{ref_text}。具体的な復習法とチェック項目を教えて。"
            st.session_state['guide'] = ask_gemini_robust(prompt)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テスト作成"):
                prompt = f"新潟高校入試レベル。{sel_sub}「{sel_top}」の実践問題1問。反省「{ref_text}」を踏まえて作成せよ。解答解説付き。"
                st.session_state['test'] = ask_gemini_robust(prompt)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])

    # ------------------
    # TAB 3: 画像採点
    # ------------------
    with tab3:
        st.subheader("📷 自由画像採点")
        c1,c2,c3 = st.columns(3)
        img_p = c1.file_uploader("問題", type=['jpg','png'])
        img_u = c2.file_uploader("解答", type=['jpg','png'])
        img_a = c3.file_uploader("正解", type=['jpg','png'])
        if img_p and img_u and img_a and st.button("採点開始"):
            with st.spinner("採点中..."):
                imgs = [PIL.Image.open(i) for i in [img_p, img_u, img_a]]
                res = ask_gemini_robust("厳密に採点し、添削とアドバイスを行ってください。", imgs)
                st.markdown(res)

    # ------------------
    # TAB 4: その他特訓
    # ------------------
    with tab4:
        st.subheader("🧩 その他特訓（ランダム出題）")
        st.caption("公立高校入試レベルの問題をランダムに出題します。解答を紙に書いて撮影してください。")
        
        train_menu = st.radio("メニューを選択", ["理科記述", "社会記述", "漢字", "リスニング", "証明問題"], horizontal=True)
        
        if st.button("🎲 問題を作成する"):
            st.session_state['practice_a'] = None # 解答クリア
            st.session_state['practice_script'] = None
            
            with st.spinner("AIが出題中..."):
                if train_menu == "リスニング":
                    # リスニング用の特別なプロンプト
                    p_lis = """
                    公立高校入試レベルの英語リスニング問題を作成してください。
                    出力フォーマット:
                    【スクリプト】
                    (ここに読み上げ用の英文のみを書く)
                    【設問】
                    (ここに設問文と選択肢などを書く)
                    【正解】
                    (ここに正解と解説を書く)
                    """
                    res = ask_gemini_robust(p_lis)
                    st.session_state['practice_q'] = res # 生データ保持
                    
                    # スクリプト抽出
                    try:
                        parts = res.split("【設問】")
                        script_part = parts[0].replace("【スクリプト】", "").strip()
                        question_part = "【設問】" + parts[1] if len(parts) > 1 else res
                        
                        st.session_state['practice_script'] = script_part
                        st.session_state['practice_q_display'] = question_part
                    except:
                        st.session_state['practice_q_display'] = res
                
                else:
                    # その他の科目
                    p_normal = f"""
                    公立高校入試レベルの「{train_menu}」の問題を1問作成してください。
                    新潟高校志望の生徒向けです。
                    
                    出力形式:
                    【問題】
                    (問題文のみを表示)
                    
                    【正解と解説】
                    (模範解答と解説、採点基準)
                    """
                    res = ask_gemini_robust(p_normal)
                    st.session_state['practice_q'] = res
                    # 表示用に分割（正解は隠す）
                    if "【正解と解説】" in res:
                        st.session_state['practice_q_display'] = res.split("【正解と解説】")[0]
                    else:
                        st.session_state['practice_q_display'] = res

        # 問題表示エリア
        if st.session_state['practice_q']:
            st.markdown("---")
            st.markdown("#### 📝 問題")
            
            # リスニングの場合の音声プレイヤー
            if train_menu == "リスニング" and st.session_state['practice_script']:
                if gTTS is None:
                    st.error("⚠️ `gTTS` ライブラリがインストールされていません。")
                else:
                    st.write("🔈 **音声を再生して解答してください**")
                    audio_data = text_to_speech(st.session_state['practice_script'])
                    if audio_data:
                        st.audio(audio_data, format='audio/mp3')
            
            st.markdown(st.session_state.get('practice_q_display', ''))
            
            st.markdown("---")
            st.write("📷 **解答をアップロードして採点**")
            user_ans_img = st.file_uploader("解答の写真をアップロード", type=['jpg', 'png', 'jpeg'], key="practice_up")
            
            if user_ans_img and st.button("💯 採点・フィードバック"):
                with st.spinner("AI先生が採点中..."):
                    # 全体の情報（正解含む）とユーザー画像を渡す
                    prompt_check = f"""
                    以下の問題データに基づき、生徒の解答画像を採点してください。
                    
                    【問題データ（正解含む）】
                    {st.session_state['practice_q']}
                    
                    採点結果、添削、改善アドバイスをわかりやすく出力してください。
                    """
                    img = PIL.Image.open(user_ans_img)
                    res_check = ask_gemini_robust(prompt_check, [img])
                    st.session_state['practice_a'] = res_check
            
            if st.session_state['practice_a']:
                st.success("✅ 採点完了！")
                st.markdown(st.session_state['practice_a'])
                
                # 正解データの表示（トグル）
                with st.expander("模範解答を表示する"):
                    st.markdown(st.session_state['practice_q'])

else:
    st.info("👆 サイドバーからCSVを読み込むか、ファイルをアップロードしてください。")
