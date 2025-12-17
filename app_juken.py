import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import PIL.Image
import json
import re

# ==========================================
# 🔐 1. セキュリティ設定 & 接続診断
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide")
st.title("🏔️ 新潟高校 合格ストラテジー & 徹底復習")

if not api_key:
    st.warning("⚠️ アプリの設定(Secrets)にAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 🚑 モデル自動検出 & 選択機能（ここが修復の肝です）
# ---------------------------------------------------------
st.sidebar.header("⚙️ システム設定")

# 利用可能なモデルを取得してみる
try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    if not available_models:
        st.error("❌ 利用可能なモデルが見つかりません。APIキーが無効か、Google側の障害の可能性があります。")
        st.stop()

    # 優先順位: 1.5-pro -> 1.5-flash -> gemini-pro
    default_index = 0
    for i, m_name in enumerate(available_models):
        if "gemini-1.5-pro" in m_name:
            default_index = i
            break
        elif "gemini-1.5-flash" in m_name: # proがない場合の第2候補
            default_index = i
            
    # ユーザーがモデルを選べるようにする（これで404回避）
    selected_model_name = st.sidebar.selectbox(
        "使用するAIモデル",
        available_models,
        index=default_index,
        help="エラーが出る場合は別のモデルに切り替えてください"
    )
    
    # 選択されたモデルで初期化
    model_text = genai.GenerativeModel(selected_model_name)
    model_vision = genai.GenerativeModel(selected_model_name)
    
    st.sidebar.success(f"✅ {selected_model_name} に接続中")
    st.sidebar.caption(f"Lib Version: {genai.__version__}")

except Exception as e:
    st.error(f"接続エラー: {e}")
    st.stop()


# ---------------------------------------------------------
# 2. アプリの共通設定
# ---------------------------------------------------------
TARGET_SCHOOL = "新潟高校（普通科）"
EXAM_DATE = datetime.date(2026, 3, 4)

if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}
if 'confirm_delete' not in st.session_state: st.session_state['confirm_delete'] = False
if 'category_map' not in st.session_state: st.session_state['category_map'] = {}

FIXED_CATEGORIES = {
    "国語": ["漢字", "文法", "評論", "古文", "その他"],
    "数学": ["数と式", "方程式・不等式", "関数(比例・1次)", "関数(2次・その他)", "平面図形", "空間図形", "図形の証明", "確率", "データの活用", "整数・規則性", "作図", "融合問題・その他"],
    "英語": ["単語・語彙", "文法(時制・動詞)", "文法(準動詞・関係詞)", "文法(その他)", "長文読解(物語)", "長文読解(説明文)", "英作文", "リスニング", "会話文", "語順整序", "適語補充", "その他"],
    "理科": ["物理(光・音・力)", "物理(電気・磁界)", "物理(運動・エネルギー)", "化学(物質・気体)", "化学(変化・原子)", "化学(イオン・電池)", "生物(植物)", "生物(動物・人体)", "生物(遺伝・進化)", "地学(火山・地層)", "地学(天気・気象)", "地学(天体)"],
    "社会": ["地理(世界)", "地理(日本)", "地理(資料読取)", "歴史(古代～中世)", "歴史(近世)", "歴史(近現代)", "公民(現代社会・人権)", "公民(政治)", "公民(経済)", "公民(国際)", "融合問題", "その他"]
}

# --- サイドバー ---
st.sidebar.subheader("📚 参考書設定")
with st.sidebar.form("textbook_form"):
    tb_math = st.text_input("数学", value=st.session_state['textbooks'].get('数学', ''), placeholder="例: チャート式")
    tb_eng = st.text_input("英語", value=st.session_state['textbooks'].get('英語', ''), placeholder="例: 教科書")
    tb_sci = st.text_input("理科", value=st.session_state['textbooks'].get('理科', ''), placeholder="例: 自由自在")
    tb_soc = st.text_input("社会", value=st.session_state['textbooks'].get('社会', ''), placeholder="例: 用語集")
    tb_jpn = st.text_input("国語", value=st.session_state['textbooks'].get('国語', ''), placeholder="例: 便覧")
    if st.form_submit_button("設定を保存"):
        st.session_state['textbooks'] = {'数学': tb_math, '英語': tb_eng, '理科': tb_sci, '社会': tb_soc, '国語': tb_jpn}
        st.sidebar.success("保存完了")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 データ管理")

if st.session_state['data_store']:
    st.sidebar.success(f"{len(st.session_state['data_store'])} 件記憶中")
    if not st.session_state['confirm_delete']:
        if st.sidebar.button("🗑️ データを全消去"):
            st.session_state['confirm_delete'] = True
            st.rerun()
    else:
        col_yes, col_no = st.sidebar.columns(2)
        if col_yes.button("はい、削除", type="primary"):
            st.session_state['data_store'] = {}
            st.session_state['category_map'] = {}
            st.session_state['confirm_delete'] = False
            st.rerun()
        if col_no.button("キャンセル"):
            st.session_state['confirm_delete'] = False
            st.rerun()
else:
    st.sidebar.info("データなし")

# ---------------------------------------------------------
# 3. 関数定義
# ---------------------------------------------------------
def parse_csv(file):
    try:
        file.seek(0)
        try:
            df = pd.read_csv(file, header=None)
        except UnicodeDecodeError:
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
            new_cols = [str(val).strip() for val in subset.iloc[0]]
            subset.columns = new_cols
            subset.columns.name = None
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
            return None
        return None
    except Exception: return None

def ask_gemini_text(prompt):
    try:
        return model_text.generate_content(prompt).text
    except Exception as e: return f"エラー: {e}"

def ask_gemini_vision(prompt, image_list):
    try:
        content = [prompt] + image_list
        response = model_vision.generate_content(content)
        return response.text
    except Exception as e: return f"エラー: {e}"

def categorize_topics_with_ai(df_all):
    unique_pairs = df_all[['教科', '内容']].drop_duplicates()
    unknown_list = []
    
    for _, row in unique_pairs.iterrows():
        subj = row['教科']
        topic = str(row['内容']).strip()
        if (subj, topic) not in st.session_state['category_map']:
            unknown_list.append(f"{subj}: {topic}")
    
    if unknown_list:
        with st.spinner(f"AIが {len(unknown_list)} 件の単元を整理中..."):
            categories_str = json.dumps(FIXED_CATEGORIES, ensure_ascii=False, indent=2)
            prompt = f"""
            入力された「教科: 単元名」を、以下の【定義済みカテゴリリスト】の中から最も適切なものに分類し、JSON形式で出力してください。
            【定義済みカテゴリリスト】
            {categories_str}
            【入力データ】
            """ + "\n".join(unknown_list)
            
            try:
                response = ask_gemini_text(prompt)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    mapping = json.loads(json_match.group())
                    for k, v in mapping.items():
                        if ':' in k:
                            s, t = k.split(':', 1)
                            st.session_state['category_map'][(s.strip(), t.strip())] = v.strip()
            except Exception as e:
                st.error(f"分類エラー: {e}")

    df_clean = df_all.copy()
    if '詳細' not in df_clean.columns:
        df_clean['詳細'] = df_clean['内容']
    
    def apply_mapping(row):
        key = (row['教科'], str(row['内容']).strip())
        return st.session_state['category_map'].get(key, row['内容'])

    df_clean['内容'] = df_clean.apply(apply_mapping, axis=1)
    return df_clean

# ---------------------------------------------------------
# 4. メイン画面
# ---------------------------------------------------------
st.markdown("##### 📂 学習データのアップロード（CSV）")

with st.form("upload_form", clear_on_submit=True):
    uploaded_files = st.file_uploader("CSVファイルを選択", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")
    submit_upload = st.form_submit_button("📥 読み込んで保存")
    
    if submit_upload and uploaded_files:
        new_c, over_c = 0, 0
        error_files = []
        for file in uploaded_files:
            df = parse_csv(file)
            if df is not None:
                if file.name in st.session_state['data_store']: over_c += 1
                else: new_c += 1
                st.session_state['data_store'][file.name] = df
            else:
                error_files.append(file.name)
        
        if new_c > 0 or over_c > 0:
            st.success(f"✅ 新規:{new_c}件 / 上書き:{over_c}件 保存完了")
            st.rerun()
        if error_files:
            st.error(f"⚠️ 読み込めなかった: {', '.join(error_files)}")

# ---------------------------------------------------------
# 5. 機能タブ
# ---------------------------------------------------------
if st.session_state['data_store']:
    raw_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
    all_df = categorize_topics_with_ai(raw_df)
else:
    all_df = pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📅 計画", "📷 画像採点"])

with tab1:
    if not all_df.empty:
        summary = all_df.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).round(1)
        
        summary_clean = pd.DataFrame(summary.to_dict('list'))
        summary_clean.columns = [str(c) for c in summary_clean.columns]
        
        st.subheader("データ分析")
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("⚠️ 優先復習単元")
            st.dataframe(
                summary_clean.sort_values('得点率(%)').head(10),
                column_config={"得点率(%)": st.column_config.NumberColumn(format="%.1f%%")},
                use_container_width=True,
                hide_index=True
            )
        with col2:
            st.write("教科別平均")
            sub_sum = all_df.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率'] = (sub_sum['点数']/sub_sum['配点']*100).round(1)
            
            sub_sum_clean = pd.DataFrame(sub_sum.to_dict('list'))
            sub_sum_clean.columns = [str(c) for c in sub_sum_clean.columns]
            
            st.dataframe(sub_sum_clean, hide_index=True)
    else:
        st.info("CSVをアップロードしてください。")

with tab2:
    if not all_df.empty and 'summary' in locals():
        st.subheader("弱点克服")
        c1, c2 = st.columns(2)
        with c1: sel_sub = st.selectbox("教科", summary['教科'].unique())
        with c2: sel_top = st.selectbox("単元", summary[summary['教科']==sel_sub].sort_values('得点率(%)')['内容'])
        
        target_rows = all_df[(all_df['教科']==sel_sub) & (all_df['内容']==sel_top)]
        rate = (target_rows['点数'].sum() / target_rows['配点'].sum() * 100).round(1)
        
        st.info(f"単元「{sel_top}」の得点率: **{rate}%**")
        original_topics = target_rows['詳細'].unique().tolist() if '詳細' in target_rows.columns else []
        original_topics_str = "、".join([str(t) for t in original_topics])
        st.caption(f"含まれる元の単元: {original_topics_str}")
        
        book = st.session_state['textbooks'].get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.spinner("思考中..."):
                p = f"新潟高校志望。教科「{sel_sub}」、カテゴリ「{sel_top}」（詳細は{original_topics_str}など）が苦手（得点率{rate}%）。参考書『{book}』のどこを見るべきか、新潟高校レベルの理解の深さ、チェック項目3つを教えて。"
                st.session_state['guide'] = ask_gemini_text(p)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストをする"):
                with st.spinner("作成中..."):
                    p2 = f"新潟高校レベル。{sel_sub}の「{sel_top}」（詳細: {original_topics_str}）の実践問題1問作成。解答解説付き。"
                    st.session_state['test'] = ask_gemini_text(p2)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])
    else:
        st.info("データをアップロードしてください。")

with tab3:
    if st.button("計画作成"):
        with st.spinner("作成中..."):
            st.markdown(ask_gemini_text(f"今日{datetime.date.today()}から入試{EXAM_DATE}までの新潟高校合格スケジュール。"))

with tab4:
    st.subheader("📷 画像採点")
    st.info("問題、解答、模範解答をセットしてください。")
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.markdown("**① 問題**")
        img_prob_cam = st.camera_input("問題を撮影", key="cam1")
        img_prob_file = st.file_uploader("または画像を選択", type=['png', 'jpg', 'jpeg'], key="file1", label_visibility="collapsed")
        img_prob = img_prob_cam if img_prob_cam else img_prob_file
    with col_img2:
        st.markdown("**② 自分の解答**")
        img_user_cam = st.camera_input("解答を撮影", key="cam2")
        img_user_file = st.file_uploader("または画像を選択", type=['png', 'jpg', 'jpeg'], key="file2", label_visibility="collapsed")
        img_user = img_user_cam if img_user_cam else img_user_file
    with col_img3:
        st.markdown("**③ 模範解答**")
        img_ans_cam = st.camera_input("模範解答を撮影", key="cam3")
        img_ans_file = st.file_uploader("または画像を選択", type=['png', 'jpg', 'jpeg'], key="file3", label_visibility="collapsed")
        img_ans = img_ans_cam if img_ans_cam else img_ans_file

    st.markdown("---")
    if img_prob and img_user and img_ans:
        if st.button("🚀 採点実行"):
            with st.spinner("分析中..."):
                try:
                    images = [PIL.Image.open(img_prob), PIL.Image.open(img_user), PIL.Image.open(img_ans)]
                    prompt_vision = f"新潟高校志望。3枚の画像から、採点結果(正誤)、添削コメント、原因分析と対策、類題作成を行って。"
                    st.markdown(ask_gemini_vision(prompt_vision, images))
                except Exception as e:
                    st.error(f"エラー: {e}")
    else:
        st.warning("☝️ 3枚全ての画像をセットしてください。")
