import os
import glob
import streamlit as st
from google.colab import drive
import datetime


# 1. Google Driveの接続確認
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')


# 2. 'kokufuku' フォルダを自動捜索
print("🔍 Google Driveの中から 'kokufuku' フォルダを探しています...")
target_folder_name = "kokufuku"
found_path = None


root_dir = "/content/drive/MyDrive"
for root, dirs, files in os.walk(root_dir):
    depth = root[len(root_dir):].count(os.sep)
    if depth < 3:
        if target_folder_name in dirs:
            check_path = os.path.join(root, target_folder_name)
            if glob.glob(os.path.join(check_path, "*.csv")):
                found_path = check_path
                print(f"✅ 見つかりました！: {found_path}")
                break


if not found_path:
    print("❌ フォルダが見つからないか、中にCSVファイルが入っていません。")
else:
    # 3. app_juken.py を作成
    # （👇 ここであなたのAPIキーを入れてください）
    api_key = "AIzaSyB6FP2ackt_gAA01FaBrHTTlT1yQV5zr1o" 
    
    app_code = f"""
import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import glob
import os


# 自動検出されたパス
DATA_FOLDER_PATH = r"{found_path}"
# APIキー（Secrets対応：GitHub用とColab用の両対応）
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = "{api_key}"


if api_key == "ここに取得したキーを貼り付ける" or not api_key:
    st.warning("⚠️ APIキーが設定されていません。コード内またはSecretsを確認してください。")
else:
    genai.configure(api_key=api_key)


TARGET_SCHOOL = "新潟高校（普通科）"
EXAM_DATE = datetime.date(2026, 3, 4)


st.set_page_config(page_title="新潟高校 合格ナビ", layout="wide")
st.title("🏔️ 新潟高校 合格ストラテジー & 徹底復習")


# サイドバー：参考書設定
st.sidebar.header("📚 使用教材の設定")
textbooks = {{}}
with st.sidebar.expander("参考書名を登録する"):
    textbooks['数学'] = st.text_input("数学", placeholder="例: チャート式")
    textbooks['英語'] = st.text_input("英語", placeholder="例: 教科書")
    textbooks['理科'] = st.text_input("理科", placeholder="例: 自由自在")
    textbooks['社会'] = st.text_input("社会", placeholder="例: 用語集")
    textbooks['国語'] = st.text_input("国語", placeholder="例: 便覧")


@st.cache_data
def load_all_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files: return None
    all_df = pd.DataFrame()
    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            header_idx = df[df.apply(lambda r: r.astype(str).str.contains('大問|内容').any(), axis=1)].index
            if len(header_idx) > 0:
                idx = header_idx[0]
                subset = df.iloc[idx:].reset_index(drop=True).T
                subset.columns = subset.iloc[0]
                subset = subset[1:]
                if '大問' in subset.columns: subset = subset.dropna(subset=['大問'])
                subset['点数'] = pd.to_numeric(subset['点数'], errors='coerce').fillna(0)
                subset['配点'] = pd.to_numeric(subset['配点'], errors='coerce').fillna(0)
                subset['ファイル名'] = os.path.basename(f)
                for sub in ['数学','英語','理科','社会','国語']:
                    if sub in os.path.basename(f):
                        subset['教科'] = sub
                        break
                else: subset['教科'] = 'その他'
                all_df = pd.concat([all_df, subset], ignore_index=True)
        except: pass
    return all_df


def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content(prompt).text
    except Exception as e: return f"エラー: {{e}}"


# データ読み込み
with st.spinner(f"フォルダ({{DATA_FOLDER_PATH}})からデータを読み込み中..."):
    df = load_all_data(DATA_FOLDER_PATH)


if df is None or df.empty:
    st.error("データが読み込めませんでした。フォルダにCSVがあるか確認してください。")
else:
    # 集計
    summary = df.groupby(['教科', '内容'])[['点数', '配点']].sum().reset_index()
    summary['得点率(%)'] = (summary['点数'] / summary['配点'] * 100).round(1)
    
    tab1, tab2, tab3 = st.tabs(["📊 全体分析", "📖 復習＆テスト", "📅 計画"])
    
    with tab1:
        st.subheader("データ分析")
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("⚠️ 優先復習単元")
            st.dataframe(summary.sort_values('得点率(%)').head(10).style.format({{'得点率(%)': '{{:.1f}}%'}}).background_gradient(subset=['得点率(%)'], cmap='RdYlGn'))
        with col2:
            st.write("教科別平均")
            sub_sum = df.groupby('教科')[['点数', '配点']].sum().reset_index()
            sub_sum['得点率'] = (sub_sum['点数']/sub_sum['配点']*100).round(1)
            st.dataframe(sub_sum)


    with tab2:
        st.subheader("弱点克服")
        c1, c2 = st.columns(2)
        with c1: sel_sub = st.selectbox("教科", summary['教科'].unique())
        with c2: sel_top = st.selectbox("単元", summary[summary['教科']==sel_sub].sort_values('得点率(%)')['内容'])
        
        rate = summary[(summary['教科']==sel_sub)&(summary['内容']==sel_top)]['得点率(%)'].values[0]
        st.info(f"得点率: **{{rate}}%**")
        
        book = textbooks.get(sel_sub, "参考書")
        
        if st.button("① 復習ポイントを聞く"):
            with st.spinner("AI思考中..."):
                p = f"新潟高校志望。教科{{sel_sub}}、単元{{sel_top}}、得点率{{rate}}%。参考書『{{book}}』のどこを見るべきか、新潟高校レベルの理解の深さ、チェック項目3つを教えて。"
                st.session_state['guide'] = ask_gemini(p)
        
        if 'guide' in st.session_state:
            st.markdown(st.session_state['guide'])
            if st.button("② 確認テストをする"):
                with st.spinner("作成中..."):
                    p2 = f"新潟高校レベル。{{sel_sub}}の{{sel_top}}の実践問題1問作成。解答解説付き。"
                    st.session_state['test'] = ask_gemini(p2)
        
        if 'test' in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state['test'])


    with tab3:
        if st.button("計画作成"):
            with st.spinner("作成中..."):
                st.markdown(ask_gemini(f"今日{datetime.date.today()}から入試{datetime.date(2026, 3, 4)}までの新潟高校合格スケジュール。"))
    """


    # ファイル名を app_juken.py にして保存
    with open("app_juken.py", "w", encoding="utf-8") as f:
        f.write(app_code)
    
    print("📝 app_juken.py を作成しました！")