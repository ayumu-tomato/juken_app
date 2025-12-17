import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import PIL.Image
import json
import re
import time
import io

# ==========================================
# 🔐 初期設定
# ==========================================
st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide")
st.title("🏔️ 新潟高校 合格ストラテジー & 徹底復習")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

if not api_key:
    st.warning("⚠️ SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 🤖 モデル自動検出ロジック (ハイブリッド構成)
# ---------------------------------------------------------
def get_available_models():
    try:
        return [m.name.replace("models/", "") for m in genai.list_models()]
    except:
        return []

ALL_MODELS = get_available_models()

def get_best_pro_model(all_models):
    """指導・採点用：最新のProモデルを探す"""
    priority_list = [
        "gemini-3-pro",
        "gemini-3-pro-preview",
        "gemini-3.0-pro",
        "gemini-2.5-pro",
        "gemini-2.0-pro-exp",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    for m in priority_list:
        if m in all_models: return m
    
    # リストになくてもProがあれば使う
    pro_models = [m for m in all_models if "pro" in m and "vision" not in m]
    if pro_models:
        pro_models.sort(reverse=True)
        return pro_models[0]
            
    return "gemini-1.5-flash"

def get_best_flash_model(all_models):
    """単元整理用：最新のFlashモデルを探す"""
    priority_list = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-001",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b"
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
    
    st.sidebar.success(f"🚀 Guidance: {MODEL_NAME_PRO}")
    st.sidebar.info(f"⚡ Categorize: {MODEL_NAME_FLASH}")
except Exception as e:
    st.error(f"❌ モデルの起動に失敗しました: {e}")
    st.stop()


# ---------------------------------------------------------
# 💾 データ管理
# ---------------------------------------------------------
if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'clean_df' not in st.session_state: st.session_state['clean_df'] = pd.DataFrame()
if 'category_map' not in st.session_state: st.session_state['category_map'] = {}
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}

# 定義済みカテゴリ
FIXED_CATEGORIES = {
    "国語": [
        "漢字", "文法", "評論", "古文", "その他"
    ],
    "数学": [
        "正負の数・文字と式", "一次方程式・連立方程式", "平方根", "式の展開と因数分解", 
        "二次方程式", "比例・反比例", "一次関数", "関数y=ax^2", 
        "平面図形（作図・移動・おうぎ形）", "空間図形", "図形の性質と証明（合同・相似・円）", 
        "確率・統計（データの活用・三平方の定理）", "融合問題", "その他"
    ],
    "英語": [
        "be動詞・一般動詞・命令文", "代名詞・疑問詞・現在進行形", "過去形・過去進行形・未来表現", 
        "助動詞", "不定詞・動名詞", "比較", "受動態", "現在完了形", 
        "分詞・分詞構文", "関係代名詞", "文構造・接続詞（SVOO/SVOC・that節）", 
        "間接疑問文・仮定法・付加疑問文", "融合問題", "その他"
    ],
    "理科": [
        "【物理】光・音・力", "【物理】電流と磁界", "【物理】運動とエネルギー", 
        "【化学】身の回りの物質・気体・水溶液", "【化学】化学変化と原子・分子", 
        "【化学】化学変化とイオン・電池", "【生物】植物の生活と種類", 
        "【生物】動物の生活と生物の変遷", "【生物】生命の連続性（遺伝・細胞）", 
        "【地学】大地の変化（火山・地震・地層）", "【地学】気象とその変化", 
        "【地学】地球と宇宙", "融合問題", "その他"
    ],
    "社会": [
        "【地理】世界の姿・気候・生活文化", "【地理】世界の諸地域", 
        "【地理】日本の姿・産業・資源エネルギー", "【地理】日本の諸地域", 
        "【歴史】古代〜中世（文明〜室町）", "【歴史】近世（安土桃山・江戸）", 
        "【歴史】近代①（明治〜第一次大戦）", "【歴史】近代②〜現代（昭和〜現在）", 
        "【公民】現代社会・日本国憲法・人権", "【公民】政治の仕組み", 
        "【公民】経済の仕組み", "【公民】国際社会・環境問題", "融合問題", "その他"
    ]
}

# ---------------------------------------------------------
# 🛠️ 関数定義
# ---------------------------------------------------------
def ask_gemini_robust(prompt, image_list=None, use_flash=False):
    max_retries = 3
    if image_list:
        target_model = model_vision
    elif use_flash:
        target_model = model_flash
    else:
        target_model = model_pro

    for attempt in range(max_retries):
        try:
            if image_list:
                response = target_model.generate_content([prompt] + image_list)
            else:
                response = target_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                st.toast(f"⏳ アクセス集中...待機中 ({attempt+1}/3)")
                time.sleep((attempt + 1) * 3)
            else:
                return f"エラー: {e}"
    return "❌ 混雑のため応答できませんでした。"

def detect_subject(file_name):
    """
    ファイル名から教科を判定します。
    ユーザー指摘により、ファイル名（例：社会_岩手県.csv）から直接判定します。
    """
    name_str = str(file_name)
    
    # 教科リスト
    subjects = ['数学', '英語', '理科', '社会', '国語']
    
    for sub in subjects:
        # ファイル名に教科名が含まれていればそれを採用
        if sub in name_str:
            return sub
            
    return 'その他'

def parse_csv(file):
    try:
        file.seek(0)
        try:
            df = pd.read_csv(file, header=None)
        except:
            file.seek(0)
            df = pd.read_csv(file, header=None, encoding='cp932')
        
        header_row_mask = df.apply(lambda r: r.astype(str).str.contains('大問|内容').any(), axis=1)
        if len(df[header_row_mask]) > 0:
            idx = df[header_row_mask].index[0]
            target_row = df.iloc[idx]
            col_idx = 0
            for c in df.columns:
                val = str(target_row[c])
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
            
            # 【修正】ファイル名から教科を判定
            subset['教科'] = detect_subject(file.name)
            
            if '点数' in subset.columns:
                return subset
    except:
        pass
    return None

def process_and_categorize():
    if not st.session_state['data_store']:
        st.session_state['clean_df'] = pd.DataFrame()
        return

    # 単元整理はFlashで高速化
    model_label = MODEL_NAME_FLASH
    
    with st.status(f"🚀 データを解析しています... (Engine: {model_label})", expanded=True) as status:
        st.write("📂 1. データを結合中...")
        raw_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
        time.sleep(0.2)
        
        st.write("🔍 2. 未知の単元を検索中...")
        unique_pairs = raw_df[['教科', '内容']].drop_duplicates()
        unknown_list = []
        
        # 教科ごとの単元分類プロンプト作成用
        tasks = []
        
        for _, row in unique_pairs.iterrows():
            subj = row['教科']
            topic = str(row['内容']).strip()
            
            # 既に完全一致するカテゴリがあるか確認
            is_perfect_match = False
            if subj in FIXED_CATEGORIES:
                if topic in FIXED_CATEGORIES[subj]:
                    is_perfect_match = True
            
            if not is_perfect_match and (subj, topic) not in st.session_state['category_map']:
                # まだマップになく、リストにもないものをリストアップ
                unknown_list.append(f"{subj}: {topic}")
        
        if unknown_list:
            st.write(f"⚡ 3. 新しい単元 {len(unknown_list)} 件を高速分類中...")
            categories_str = json.dumps(FIXED_CATEGORIES, ensure_ascii=False, indent=2)
            
            prompt = f"""
            あなたはデータ分類システムです。
            入力された「教科: 単元名」のリストを、以下の【定義済みマスタ】にあるカテゴリ名のどれかに分類してください。
            
            【重要ルール】
            1. **必ず**【定義済みマスタ】に記載されているカテゴリ名と**完全に一致する文字列**を出力してください。一言一句変えてはいけません。
            2. 教科も考慮してください。数学の単元を社会のカテゴリに入れないでください。
            3. どうしても当てはまらない場合は、その教科内の「その他」または「融合問題」を選択してください。
            4. 出力はJSON形式の辞書 `{{"教科: 元の単元名": "定義済みカテゴリ名", ...}}` のみにしてください。

            【定義済みマスタ】
            {categories_str}
            
            【入力データ】
            """ + "\n".join(unknown_list)
            
            response = ask_gemini_robust(prompt, use_flash=True)
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    mapping = json.loads(json_match.group())
                    for k, v in mapping.items():
                        if ':' in k:
                            s, t = k.split(':', 1)
                            st.session_state['category_map'][(s.strip(), t.strip())] = v.strip()
            except:
                st.warning("⚠️ 分類に一部失敗しましたが続行します")
        else:
            st.write("✨ 全ての単元は分類済みです")

        st.write("💾 4. データを保存中...")
        df_clean = raw_df.copy()
        if '詳細' not in df_clean.columns: df_clean['詳細'] = df_clean['内容']
        
        def apply_mapping(row):
            key = (row['教科'], str(row['内容']).strip())
            mapped_val = st.session_state['category_map'].get(key, row['内容'])
            # マッピングがない場合は元の値を使うが、本来はすべてマッピングされるはず
            return mapped_val if mapped_val else row['内容']

        df_clean['内容'] = df_clean.apply(apply_mapping, axis=1)
        st.session_state['clean_df'] = df_clean
        
        status.update(label="✅ 解析完了！", state="complete", expanded=False)

# ---------------------------------------------------------
# 🖥️ サイドバー設定
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📲 データ同期")
    if st.session_state['data_store'] or st.session_state['textbooks']:
        backup_data = {
            'textbooks': st.session_state['textbooks'],
            'data_store': {name: df.to_json(orient='split') for name, df in st.session_state['data_store'].items()}
        }
        json_str = json.dumps(backup_data, ensure_ascii=False)
        st.download_button(
            label="📤 データを保存",
            data=json_str,
            file_name=f"juken_backup_{datetime.date.today()}.json",
            mime="application/json"
        )
    
    uploaded_backup = st.file_uploader("📥 データを復元", type=['json'], key="backup_uploader")
    if uploaded_backup:
        try:
            data = json.load(uploaded_backup)
            if 'textbooks' in data:
                st.session_state['textbooks'] = data['textbooks']
            if 'data_store' in data:
                st.session_state['data_store'] = {}
                for name, df_json in data['data_store'].items():
                    st.session_state['data_store'][name] = pd.read_json(df_json, orient='split')
                st.session_state['clean_df'] = pd.DataFrame()
                st.session_state['category_map'] = {}
                st.success("✅ 復元成功！")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"復元エラー: {e}")

    st.markdown("---")
    st.subheader("📚 参考書")
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
    st.subheader("💾 ファイル")

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
        # 全体ワースト
        summary = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).round(1)
        summary_clean = pd.DataFrame(summary.to_dict('list'))
        summary_clean.columns = [str(c) for c in summary_clean.columns]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("⚠️ 全体：優先復習単元")
            st.dataframe(summary_clean.sort_values('得点率(%)').head(10), column_config={"得点率(%)": st.column_config.NumberColumn(format="%.1f%%")}, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("教科別平均")
            sub_sum = df_show.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率'] = (sub_sum['点数']/sub_sum['配点']*100).round(1)
            sub_sum_clean = pd.DataFrame(sub_sum.to_dict('list'))
            sub_sum_clean.columns = [str(c) for c in sub_sum_clean.columns]
            st.dataframe(sub_sum_clean, hide_index=True)
            
        # 教科ごとのワースト表示
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
                        sub_df[['内容', '得点率(%)']], 
                        column_config={"得点率(%)": st.column_config.NumberColumn(format="%.1f%%")},
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.caption("データなし")

    with tab2:
        st.subheader("AI家庭教師による指導")
        c1, c2 = st.columns(2)
        with c1: sel_sub = st.selectbox("教科", summary['教科'].unique())
        with c2: sel_top = st.selectbox("単元", summary[summary['教科']==sel_sub].sort_values('得点率(%)')['内容'])
        
        target_rows = df_show[(df_show['教科']==sel_sub) & (df_show['内容']==sel_top)]
        rate = (target_rows['点数'].sum() / target_rows['配点'].sum() * 100).round(1)
        original_topics = target_rows['詳細'].unique().tolist()
        original_topics_str = "、".join([str(t) for t in original_topics])
        
        st.info(f"単元: **{sel_top}** (得点率: {rate}%)")
        st.caption(f"詳細: {original_topics_str}")
        
        book = st.session_state['textbooks'].get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.status(f"🤖 AI({MODEL_NAME_PRO})が指導内容を作成中...", expanded=True) as status:
                st.write("1. 成績データを分析中...")
                time.sleep(0.5)
                p = f"""
                新潟高校志望の生徒への指導。教科: {sel_sub}, 苦手カテゴリ: {sel_top}（詳細: {original_topics_str}）, 得点率: {rate}%, 参考書: {book}。
                新潟高校合格レベルに引き上げるための復習ポイント、理解度チェック項目3つを教えて。
                """
                # 指導はPro (use_flash=False)
                res = ask_gemini_robust(p, use_flash=False)
                st.session_state['guide'] = res
                status.update(label="✅ アドバイス作成完了！", state="complete", expanded=False)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストを作成"):
                with st.status("📝 入試レベル問題を作成中...", expanded=True) as status:
                    p2 = f"新潟高校入試レベル。{sel_sub}の「{sel_top}」（詳細: {original_topics_str}）に関する実践問題を1問作成し、解答と解説を付けて。"
                    # 問題作成はPro (use_flash=False)
                    res = ask_gemini_robust(p2, use_flash=False)
                    st.session_state['test'] = res
                    status.update(label="✅ 問題作成完了！", state="complete", expanded=False)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])

    with tab3:
        if st.button("合格スケジュール作成"):
            with st.status("📅 スケジュールを立案中...", expanded=True) as status:
                prompt = f"今日({datetime.date.today()})から入試({datetime.date(2026, 3, 4)})までの新潟高校合格に向けた学習スケジュールを作成してください。"
                res = ask_gemini_robust(prompt, use_flash=False)
                st.markdown(res)
                status.update(label="✅ スケジュール完成！", state="complete", expanded=False)

    with tab4:
        st.subheader("📷 画像採点＆指導")
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1: img_prob = st.file_uploader("① 問題画像", type=['png', 'jpg', 'jpeg'])
        with col_img2: img_user = st.file_uploader("② 解答画像", type=['png', 'jpg', 'jpeg'])
        with col_img3: img_ans = st.file_uploader("③ 模範解答画像", type=['png', 'jpg', 'jpeg'])
        
        if img_prob and img_user and img_ans:
            if st.button(f"🚀 採点実行 ({MODEL_NAME_PRO})"):
                with st.status("👀 画像を解析中...", expanded=True) as status:
                    images = [PIL.Image.open(img_prob), PIL.Image.open(img_user), PIL.Image.open(img_ans)]
                    prompt_v = "新潟高校志望。3枚の画像（問題、生徒解答、模範解答）から、厳密な採点、添削、弱点分析、類題の提示を行ってください。"
                    res = ask_gemini_robust(prompt_v, images)
                    st.markdown(res)
                    status.update(label="✅ 採点完了！", state="complete", expanded=False)
else:
    st.info("👆 上記からCSVをアップロードし、「AI解析を実行」ボタンを押してください。")
