import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import PIL.Image
import json
import re
import time

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
# 🤖 AIモデル設定（Gemini 1.5 Pro 固定）
# ---------------------------------------------------------
# ユーザー様は「使い放題」とのことですので、最高精度のProを固定で使用します。
MODEL_NAME = 'gemini-1.5-pro'

try:
    model_main = genai.GenerativeModel(MODEL_NAME)
    model_vision = genai.GenerativeModel(MODEL_NAME)
    # 接続確認
    # model_main.generate_content("test")
except Exception as e:
    st.error(f"❌ モデル『{MODEL_NAME}』の読み込みに失敗しました。")
    st.info("requirements.txt に `google-generativeai>=0.8.3` が指定されているか確認してください。")
    st.stop()

st.sidebar.caption(f"🚀 AI Model: {MODEL_NAME} (Active)")

# ---------------------------------------------------------
# 💾 データ管理（セッションステート）
# ---------------------------------------------------------
if 'data_store' not in st.session_state: st.session_state['data_store'] = {} # 生データ
if 'clean_df' not in st.session_state: st.session_state['clean_df'] = pd.DataFrame() # 整理済みデータ
if 'category_map' not in st.session_state: st.session_state['category_map'] = {} # 単元変換辞書
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}

# 12分類定義
FIXED_CATEGORIES = {
    "国語": ["漢字", "文法", "評論", "古文", "その他"],
    "数学": ["数と式", "方程式・不等式", "関数(比例・1次)", "関数(2次・その他)", "平面図形", "空間図形", "図形の証明", "確率", "データの活用", "整数・規則性", "作図", "融合問題・その他"],
    "英語": ["単語・語彙", "文法(時制・動詞)", "文法(準動詞・関係詞)", "文法(その他)", "長文読解(物語)", "長文読解(説明文)", "英作文", "リスニング", "会話文", "語順整序", "適語補充", "その他"],
    "理科": ["物理(光・音・力)", "物理(電気・磁界)", "物理(運動・エネルギー)", "化学(物質・気体)", "化学(変化・原子)", "化学(イオン・電池)", "生物(植物)", "生物(動物・人体)", "生物(遺伝・進化)", "地学(火山・地層)", "地学(天気・気象)", "地学(天体)"],
    "社会": ["地理(世界)", "地理(日本)", "地理(資料読取)", "歴史(古代～中世)", "歴史(近世)", "歴史(近現代)", "公民(現代社会・人権)", "公民(政治)", "公民(経済)", "公民(国際)", "融合問題", "その他"]
}

# ---------------------------------------------------------
# 🛠️ 関数定義
# ---------------------------------------------------------
def ask_gemini_robust(prompt, image_list=None):
    """リトライ機能付きAI呼び出し"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if image_list:
                response = model_vision.generate_content([prompt] + image_list)
            else:
                response = model_main.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                time.sleep((attempt + 1) * 2)
            else:
                return f"エラー: {e}"
    return "❌ 混雑のため応答できませんでした。"

def parse_csv(file):
    """CSV読み込み・クリーニング"""
    try:
        file.seek(0)
        try:
            df = pd.read_csv(file, header=None)
        except:
            file.seek(0)
            df = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行を探す
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
            
            for sub in ['数学','英語','理科','社会','国語']:
                if sub in file.name:
                    subset['教科'] = sub
                    break
            else:
                subset['教科'] = 'その他'
            
            if '点数' in subset.columns:
                return subset
    except:
        pass
    return None

def process_and_categorize():
    """【重要】データを結合し、AIで単元を整理して保存する一括処理"""
    if not st.session_state['data_store']:
        return

    # 1. 全データ結合
    raw_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
    
    # 2. 未知の単元を探す
    unique_pairs = raw_df[['教科', '内容']].drop_duplicates()
    unknown_list = []
    for _, row in unique_pairs.iterrows():
        subj = row['教科']
        topic = str(row['内容']).strip()
        if (subj, topic) not in st.session_state['category_map']:
            unknown_list.append(f"{subj}: {topic}")
    
    # 3. AIに分類させる（未知がある場合のみ）
    if unknown_list:
        status_text = st.empty()
        status_text.info(f"🤖 AIが {len(unknown_list)} 件の新しい単元を分析・整理しています...（Gemini 1.5 Pro）")
        
        categories_str = json.dumps(FIXED_CATEGORIES, ensure_ascii=False, indent=2)
        prompt = f"""
        学習塾の教務システムとして振る舞ってください。
        入力された「教科: 単元名」を、以下の【定義済みカテゴリリスト】から最も適切なものに分類し、JSON形式で出力してください。
        
        【定義済みカテゴリリスト】
        {categories_str}
        
        【入力データ】
        """ + "\n".join(unknown_list)
        
        response = ask_gemini_robust(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                mapping = json.loads(json_match.group())
                for k, v in mapping.items():
                    if ':' in k:
                        s, t = k.split(':', 1)
                        st.session_state['category_map'][(s.strip(), t.strip())] = v.strip()
                status_text.success("✅ 整理完了！")
                time.sleep(1)
                status_text.empty()
        except:
            status_text.warning("⚠️ 一部の単元整理に失敗しましたが、処理を続行します。")

    # 4. マッピングを適用して保存
    df_clean = raw_df.copy()
    if '詳細' not in df_clean.columns: df_clean['詳細'] = df_clean['内容']
    
    def apply_mapping(row):
        key = (row['教科'], str(row['内容']).strip())
        return st.session_state['category_map'].get(key, row['内容']) # マップになければそのまま

    df_clean['内容'] = df_clean.apply(apply_mapping, axis=1)
    
    # 5. 結果をセッションステートに保存（これで再計算不要！）
    st.session_state['clean_df'] = df_clean

# ---------------------------------------------------------
# 🖥️ サイドバー設定
# ---------------------------------------------------------
with st.sidebar.form("textbook_form"):
    st.subheader("📚 参考書設定")
    tb_math = st.text_input("数学", value=st.session_state['textbooks'].get('数学', ''), placeholder="例: チャート式")
    tb_eng = st.text_input("英語", value=st.session_state['textbooks'].get('英語', ''), placeholder="例: 教科書")
    tb_sci = st.text_input("理科", value=st.session_state['textbooks'].get('理科', ''), placeholder="例: 自由自在")
    tb_soc = st.text_input("社会", value=st.session_state['textbooks'].get('社会', ''), placeholder="例: 用語集")
    tb_jpn = st.text_input("国語", value=st.session_state['textbooks'].get('国語', ''), placeholder="例: 便覧")
    if st.form_submit_button("保存"):
        st.session_state['textbooks'] = {'数学': tb_math, '英語': tb_eng, '理科': tb_sci, '社会': tb_soc, '国語': tb_jpn}
        st.sidebar.success("設定保存完了")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ データを全消去"):
    st.session_state['data_store'] = {}
    st.session_state['clean_df'] = pd.DataFrame()
    st.session_state['category_map'] = {}
    st.rerun()

# ---------------------------------------------------------
# 📂 メイン画面：アップロード & 解析実行
# ---------------------------------------------------------
st.markdown("### 1️⃣ データのアップロード & 解析")
st.caption("CSVファイルをアップロードし、**「AI解析を実行」ボタン**を押してください。整理結果が保存され、タブ切り替えが高速になります。")

col_up, col_btn = st.columns([3, 1])

with col_up:
    uploaded_files = st.file_uploader("CSVファイル", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")

with col_btn:
    # 解析実行ボタン（ここを押した時だけ重い処理が走る）
    if st.button("🚀 AI解析を実行", type="primary", use_container_width=True):
        if uploaded_files:
            # 1. 読み込み
            new_count = 0
            for file in uploaded_files:
                df = parse_csv(file)
                if df is not None:
                    st.session_state['data_store'][file.name] = df
                    new_count += 1
            
            if new_count > 0:
                # 2. 整理処理（ここでGemini Proが動く）
                process_and_categorize()
                st.success(f"✅ {new_count}件のファイルを読み込み、単元を整理しました！")
            else:
                st.warning("有効なCSVファイルが見つかりませんでした。")
        elif st.session_state['data_store']:
            # ファイルアップロードなしでボタンだけ押した場合（再整理）
            process_and_categorize()
            st.success("✅ 既存データの再整理が完了しました！")
        else:
            st.warning("ファイルを選択してください。")

# データがある場合のみ詳細を表示
if not st.session_state['clean_df'].empty:
    df_show = st.session_state['clean_df']
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📅 合格計画", "📷 画像採点"])

    # --- Tab 1: 分析 ---
    with tab1:
        summary = df_show.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).round(1)
        
        # 表示用クリーンアップ
        summary_clean = pd.DataFrame(summary.to_dict('list'))
        summary_clean.columns = [str(c) for c in summary_clean.columns]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("⚠️ 優先復習単元")
            st.dataframe(
                summary_clean.sort_values('得点率(%)').head(10),
                column_config={"得点率(%)": st.column_config.NumberColumn(format="%.1f%%")},
                use_container_width=True,
                hide_index=True
            )
        with col2:
            st.subheader("教科別平均")
            sub_sum = df_show.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率'] = (sub_sum['点数']/sub_sum['配点']*100).round(1)
            
            sub_sum_clean = pd.DataFrame(sub_sum.to_dict('list'))
            sub_sum_clean.columns = [str(c) for c in sub_sum_clean.columns]
            st.dataframe(sub_sum_clean, hide_index=True)

    # --- Tab 2: 復習 ---
    with tab2:
        st.subheader("AI家庭教師による指導")
        c1, c2 = st.columns(2)
        with c1: 
            sel_sub = st.selectbox("教科", summary['教科'].unique())
        with c2: 
            sel_top = st.selectbox("単元", summary[summary['教科']==sel_sub].sort_values('得点率(%)')['内容'])
        
        # ターゲットデータ抽出
        target_rows = df_show[(df_show['教科']==sel_sub) & (df_show['内容']==sel_top)]
        rate = (target_rows['点数'].sum() / target_rows['配点'].sum() * 100).round(1)
        original_topics = target_rows['詳細'].unique().tolist()
        original_topics_str = "、".join([str(t) for t in original_topics])
        
        st.info(f"単元: **{sel_top}** (得点率: {rate}%)")
        st.caption(f"詳細: {original_topics_str}")
        
        book = st.session_state['textbooks'].get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.spinner("Gemini 1.5 Pro が思考中..."):
                p = f"""
                新潟高校志望の生徒への指導。
                教科: {sel_sub}
                苦手カテゴリ: {sel_top}（元の単元名: {original_topics_str}）
                得点率: {rate}%
                使用参考書: {book}
                
                上記に基づき、新潟高校合格レベルに引き上げるための具体的な復習ポイント、理解の深さの目安、チェック項目3つを教えてください。
                """
                st.session_state['guide'] = ask_gemini_robust(p)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストを作成"):
                with st.spinner("問題作成中..."):
                    p2 = f"新潟高校入試レベル。{sel_sub}の「{sel_top}」（詳細: {original_topics_str}）に関する実践問題を1問作成し、解答と解説を付けてください。"
                    st.session_state['test'] = ask_gemini_robust(p2)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])

    # --- Tab 3: 計画 ---
    with tab3:
        if st.button("合格スケジュール作成"):
            with st.spinner("計画立案中..."):
                prompt = f"今日({datetime.date.today()})から入試({EXAM_DATE})までの新潟高校合格に向けた学習スケジュールを作成してください。"
                st.markdown(ask_gemini_robust(prompt))

    # --- Tab 4: 画像 ---
    with tab4:
        st.subheader("📷 画像採点＆指導")
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            img_prob = st.file_uploader("① 問題画像", type=['png', 'jpg', 'jpeg'])
        with col_img2:
            img_user = st.file_uploader("② 自分の解答画像", type=['png', 'jpg', 'jpeg'])
        with col_img3:
            img_ans = st.file_uploader("③ 模範解答画像", type=['png', 'jpg', 'jpeg'])
        
        if img_prob and img_user and img_ans:
            if st.button("🚀 採点実行 (Gemini Pro)"):
                with st.spinner("画像を分析中..."):
                    images = [PIL.Image.open(img_prob), PIL.Image.open(img_user), PIL.Image.open(img_ans)]
                    prompt_v = "新潟高校志望。3枚の画像（問題、生徒解答、模範解答）から、厳密な採点、添削、弱点分析、類題の提示を行ってください。"
                    st.markdown(ask_gemini_robust(prompt_v, images))
else:
    st.info("👆 上記からCSVをアップロードし、「AI解析を実行」ボタンを押してください。")
