import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import PIL.Image

# ==========================================
# 🔐 セキュリティ設定
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

if not api_key:
    st.warning("⚠️ アプリの設定(Secrets)にAPIキーが設定されていません。")
    st.stop()
else:
    genai.configure(api_key=api_key)
    model_vision = genai.GenerativeModel('gemini-1.5-flash')
    model_text = genai.GenerativeModel('gemini-1.5-flash')

# ---------------------------------------------------------
# 1. 設定 & UI初期化
# ---------------------------------------------------------
TARGET_SCHOOL = "新潟高校（普通科）"
EXAM_DATE = datetime.date(2026, 3, 4)

st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide")
st.title("🏔️ 新潟高校 合格ストラテジー & 徹底復習")

if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'textbooks' not in st.session_state: st.session_state['textbooks'] = {}
if 'confirm_delete' not in st.session_state: st.session_state['confirm_delete'] = False

# --- サイドバー ---
st.sidebar.header("📚 使用教材の設定")
with st.sidebar.form("textbook_form"):
    st.markdown("使用している参考書を入力して保存してください。")
    tb_math = st.text_input("数学", value=st.session_state['textbooks'].get('数学', ''), placeholder="例: チャート式")
    tb_eng = st.text_input("英語", value=st.session_state['textbooks'].get('英語', ''), placeholder="例: 教科書")
    tb_sci = st.text_input("理科", value=st.session_state['textbooks'].get('理科', ''), placeholder="例: 自由自在")
    tb_soc = st.text_input("社会", value=st.session_state['textbooks'].get('社会', ''), placeholder="例: 用語集")
    tb_jpn = st.text_input("国語", value=st.session_state['textbooks'].get('国語', ''), placeholder="例: 便覧")
    if st.form_submit_button("参考書設定を保存する"):
        st.session_state['textbooks'] = {'数学': tb_math, '英語': tb_eng, '理科': tb_sci, '社会': tb_soc, '国語': tb_jpn}
        st.sidebar.success("✅ 設定を保存しました！")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 保存されたデータ")

if st.session_state['data_store']:
    st.sidebar.success(f"{len(st.session_state['data_store'])} 件のファイルを記憶中")
    if not st.session_state['confirm_delete']:
        if st.sidebar.button("🗑️ データを全消去"):
            st.session_state['confirm_delete'] = True
            st.rerun()
    else:
        st.sidebar.warning("⚠️ 本当に全てのデータを削除しますか？")
        col_yes, col_no = st.sidebar.columns(2)
        if col_yes.button("はい、削除", type="primary"):
            st.session_state['data_store'] = {}
            st.session_state['confirm_delete'] = False
            st.rerun()
        if col_no.button("キャンセル"):
            st.session_state['confirm_delete'] = False
            st.rerun()
else:
    st.sidebar.info("データはまだありません")

# ---------------------------------------------------------
# 2. 関数定義
# ---------------------------------------------------------
def parse_csv(file):
    """CSVを読み込む関数（列ズレ・文字コード自動対応版）"""
    try:
        file.seek(0)
        try:
            df = pd.read_csv(file, header=None)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, header=None, encoding='cp932')
        
        # '大問' や '内容' が含まれる行を探す
        header_row_mask = df.apply(lambda r: r.astype(str).str.contains('大問|内容').any(), axis=1)
        
        if len(df[header_row_mask]) > 0:
            idx = df[header_row_mask].index[0] # ヘッダーがある行番号
            
            # その行の中で、'大問'などが実際に始まる列番号を探す
            target_row = df.iloc[idx]
            col_idx = 0
            for c in df.columns:
                val = str(target_row[c])
                if '大問' in val or '内容' in val:
                    col_idx = c
                    break
            
            # ヘッダー行以降、かつ有効な列以降を切り出す
            subset = df.iloc[idx:, col_idx:].reset_index(drop=True).T
            
            # 1行目をヘッダーにする
            subset.columns = subset.iloc[0]
            subset = subset[1:]
            
            # 不要な行削除
            if '大問' in subset.columns:
                subset = subset.dropna(subset=['大問'])
            
            # 数値変換（エラーは0に）
            subset['点数'] = pd.to_numeric(subset['点数'], errors='coerce').fillna(0)
            subset['配点'] = pd.to_numeric(subset['配点'], errors='coerce').fillna(0)
            subset['ファイル名'] = file.name
            
            for sub in ['数学','英語','理科','社会','国語']:
                if sub in file.name:
                    subset['教科'] = sub
                    break
            else:
                subset['教科'] = 'その他'
            
            # 必須項目があるか最終チェック
            if '点数' in subset.columns:
                return subset
            return None
        else:
            return None
    except Exception:
        return None

def ask_gemini_text(prompt):
    try:
        return model_text.generate_content(prompt).text
    except Exception as e: return f"エラー: {e}"

def ask_gemini_vision(prompt, image_list):
    try:
        content = [prompt] + image_list
        response = model_vision.generate_content(content)
        return response.text
    except Exception as e:
        return f"エラー: {e}"

# ---------------------------------------------------------
# 3. メイン画面
# ---------------------------------------------------------
st.markdown("##### 📂 学習データのアップロード（CSV）")
st.caption("Excel等で作成したCSVも読み込めます。")

with st.form("upload_form", clear_on_submit=True):
    uploaded_files = st.file_uploader("CSVファイルを選択（複数可）", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")
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
        
        if error_files:
            st.error(f"⚠️ 以下のファイルは形式が読み取れませんでした:\n{', '.join(error_files)}")

# ---------------------------------------------------------
# 4. 機能タブ
# ---------------------------------------------------------
if st.session_state['data_store']:
    all_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
else:
    all_df = pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📅 計画", "📷 画像採点"])

# --- Tab 1: 全体分析 ---
with tab1:
    if not all_df.empty:
        summary = all_df.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
        summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).round(1)
        st.subheader("データ分析")
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("⚠️ 優先復習単元")
            st.dataframe(summary.sort_values('得点率(%)').head(10).style.format({'得点率(%)': '{:.1f}%'}).background_gradient(subset=['得点率(%)'], cmap='RdYlGn'))
        with col2:
            st.write("教科別平均")
            sub_sum = all_df.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率'] = (sub_sum['点数']/sub_sum['配点']*100).round(1)
            st.dataframe(sub_sum)
    else:
        st.info("CSVデータをアップロードすると分析結果が表示されます。")

# --- Tab 2: 復習＆テスト ---
with tab2:
    if not all_df.empty and 'summary' in locals():
        st.subheader("弱点克服")
        c1, c2 = st.columns(2)
        with c1: sel_sub = st.selectbox("教科", summary['教科'].unique())
        with c2: sel_top = st.selectbox("単元", summary[summary['教科']==sel_sub].sort_values('得点率(%)')['内容'])
        
        rate = summary[(summary['教科']==sel_sub)&(summary['内容']==sel_top)]['得点率(%)'].values[0]
        st.info(f"得点率: **{rate}%**")
        book = st.session_state['textbooks'].get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.spinner("AI思考中..."):
                p = f"新潟高校志望。教科{sel_sub}、単元{sel_top}、得点率{rate}%。参考書『{book}』のどこを見るべきか、新潟高校レベルの理解の深さ、チェック項目3つを教えて。"
                st.session_state['guide'] = ask_gemini_text(p)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストをする"):
                with st.spinner("作成中..."):
                    p2 = f"新潟高校レベル。{sel_sub}の{sel_top}の実践問題1問作成。解答解説付き。"
                    st.session_state['test'] = ask_gemini_text(p2)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])
    else:
        st.info("CSVデータをアップロードすると利用できます。")

# --- Tab 3: 計画 ---
with tab3:
    if st.button("計画作成"):
        with st.spinner("作成中..."):
            st.markdown(ask_gemini_text(f"今日{datetime.date.today()}から入試{EXAM_DATE}までの新潟高校合格スケジュール。"))

# --- Tab 4: 画像採点 ---
with tab4:
    st.subheader("📷 カメラでパシャっと採点＆指導")
    st.info("「①問題」「②自分の解答」「③模範解答」を順番に撮影（またはアップロード）してください。AI先生が採点します。")

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
                    prompt_vision = f"新潟高校志望。3枚の画像（問題、生徒解答、模範解答）から、採点結果(正誤)、詳細な添削コメント、原因分析と対策、類題作成を行って。"
                    st.markdown(ask_gemini_vision(prompt_vision, images))
                except Exception as e:
                    st.error(f"エラー: {e}")
    else:
        st.warning("☝️ 3枚全ての画像をセットしてください。")
