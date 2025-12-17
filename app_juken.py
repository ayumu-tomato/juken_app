import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime

# ==========================================
# 🔐 セキュリティ設定
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

if not api_key:
    st.warning("⚠️ アプリの設定(Secrets)にAPIキーが設定されていません。Streamlit CloudのSettings > Secrets に GEMINI_API_KEY を設定してください。")
    st.stop()
else:
    genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 1. 設定 & UI初期化
# ---------------------------------------------------------
TARGET_SCHOOL = "新潟高校（普通科）"
EXAM_DATE = datetime.date(2026, 3, 4)

st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide")
st.title("🏔️ 新潟高校 合格ストラテジー & 徹底復習")

# データ保存用の領域（セッションステート）を作成
if 'data_store' not in st.session_state:
    st.session_state['data_store'] = {}  # {ファイル名: データフレーム} の辞書

# サイドバー：参考書設定
st.sidebar.header("📚 使用教材の設定")
textbooks = {}
with st.sidebar.expander("参考書名を登録する"):
    textbooks['数学'] = st.text_input("数学", placeholder="例: チャート式")
    textbooks['英語'] = st.text_input("英語", placeholder="例: 教科書")
    textbooks['理科'] = st.text_input("理科", placeholder="例: 自由自在")
    textbooks['社会'] = st.text_input("社会", placeholder="例: 用語集")
    textbooks['国語'] = st.text_input("国語", placeholder="例: 便覧")

# サイドバー：保存データの管理
st.sidebar.markdown("---")
st.sidebar.subheader("💾 保存されたデータ")
if st.session_state['data_store']:
    st.sidebar.success(f"現在 {len(st.session_state['data_store'])} 件のファイルを記憶中")
    if st.sidebar.button("🗑️ データを全消去"):
        st.session_state['data_store'] = {}
        st.rerun()
else:
    st.sidebar.info("データはまだありません")

# ---------------------------------------------------------
# 2. データ処理ロジック
# ---------------------------------------------------------
def parse_csv(file):
    """CSVファイルを読み込んでDataFrameに変換する関数"""
    try:
        df = pd.read_csv(file, header=None)
        
        # ヘッダー行を探す（'大問'や'内容'が含まれる行）
        header_idx = df[df.apply(lambda r: r.astype(str).str.contains('大問|内容').any(), axis=1)].index
        if len(header_idx) > 0:
            idx = header_idx[0]
            subset = df.iloc[idx:].reset_index(drop=True).T
            subset.columns = subset.iloc[0]
            subset = subset[1:]
            
            if '大問' in subset.columns:
                subset = subset.dropna(subset=['大問'])
            
            # 数値化
            subset['点数'] = pd.to_numeric(subset['点数'], errors='coerce').fillna(0)
            subset['配点'] = pd.to_numeric(subset['配点'], errors='coerce').fillna(0)
            
            # ファイル名と教科の判定
            subset['ファイル名'] = file.name
            for sub in ['数学','英語','理科','社会','国語']:
                if sub in file.name:
                    subset['教科'] = sub
                    break
            else:
                subset['教科'] = 'その他'
            
            return subset
        return None
    except Exception:
        return None

def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content(prompt).text
    except Exception as e: return f"エラー: {e}"

# ---------------------------------------------------------
# 3. メイン画面（ファイルアップロード）
# ---------------------------------------------------------
st.markdown("##### 📂 学習データのアップロード")
st.caption("※同じ名前のファイルをアップロードすると、新しいデータで上書き保存されます。")

uploaded_files = st.file_uploader("CSVファイルをドラッグ＆ドロップ（複数可）", accept_multiple_files=True, type=['csv'])

# アップロードされたら即座に解析して保存
if uploaded_files:
    new_count = 0
    overwrite_count = 0
    
    for file in uploaded_files:
        df = parse_csv(file)
        if df is not None:
            # 既に同じ名前のファイルがあるかチェック
            if file.name in st.session_state['data_store']:
                overwrite_count += 1
            else:
                new_count += 1
            
            # 辞書に保存（キーはファイル名なので、自動的に上書きされる）
            st.session_state['data_store'][file.name] = df
    
    # メッセージ表示
    if new_count > 0 or overwrite_count > 0:
        st.success(f"処理完了: 新規 {new_count}件 / 上書き {overwrite_count}件 を保存しました！")

# ---------------------------------------------------------
# 4. 分析データの構築
# ---------------------------------------------------------
# 保存されている全データを結合
if st.session_state['data_store']:
    all_df = pd.concat(st.session_state['data_store'].values(), ignore_index=True)
else:
    all_df = pd.DataFrame()

# データがあれば表示
if not all_df.empty:
    # 集計
    summary = all_df.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
    summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).round(1)
    
    tab1, tab2, tab3 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📅 計画"])
    
    # --- Tab 1 ---
    with tab1:
        st.subheader("データ分析")
        st.markdown(f"現在、**{len(st.session_state['data_store'])}** ファイル分のデータを分析中")
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("⚠️ 優先復習単元")
            st.dataframe(summary.sort_values('得点率(%)').head(10).style.format({'得点率(%)': '{:.1f}%'}).background_gradient(subset=['得点率(%)'], cmap='RdYlGn'))
        with col2:
            st.write("教科別平均")
            sub_sum = all_df.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率'] = (sub_sum['点数']/sub_sum['配点']*100).round(1)
            st.dataframe(sub_sum)

    # --- Tab 2 ---
    with tab2:
        st.subheader("弱点克服")
        c1, c2 = st.columns(2)
        with c1: sel_sub = st.selectbox("教科", summary['教科'].unique())
        with c2: sel_top = st.selectbox("単元", summary[summary['教科']==sel_sub].sort_values('得点率(%)')['内容'])
        
        rate = summary[(summary['教科']==sel_sub)&(summary['内容']==sel_top)]['得点率(%)'].values[0]
        st.info(f"得点率: **{rate}%**")
        
        book = textbooks.get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.spinner("AI思考中..."):
                p = f"新潟高校志望。教科{sel_sub}、単元{sel_top}、得点率{rate}%。参考書『{book}』のどこを見るべきか、新潟高校レベルの理解の深さ、チェック項目3つを教えて。"
                st.session_state['guide'] = ask_gemini(p)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストをする"):
                with st.spinner("作成中..."):
                    p2 = f"新潟高校レベル。{sel_sub}の{sel_top}の実践問題1問作成。解答解説付き。"
                    st.session_state['test'] = ask_gemini(p2)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])

    # --- Tab 3 ---
    with tab3:
        if st.button("計画作成"):
            with st.spinner("作成中..."):
                st.markdown(ask_gemini(f"今日{datetime.date.today()}から入試{EXAM_DATE}までの新潟高校合格スケジュール。"))

else:
    st.info("👈 上のボックスからCSVファイルをアップロードしてください。")
