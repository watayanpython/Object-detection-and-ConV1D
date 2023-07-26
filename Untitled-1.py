import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込み、特定の列を抽出する
df = pd.read_csv('C:/Users/watat/Documents/230301FeSi2_No9_Nonlocal4_SpinValve_10mA_01.csv')
b_col = df.iloc[17:415, 1] # B列の18-417行を抽出
e_col = df.iloc[17:415, 4] # E列の18-417行を抽出

# 2つの列を散布図としてプロットする
plt.scatter(b_col, e_col)

# グラフの軸やタイトル、凡例を設定する（適宜変更してください）
plt.xlabel('Magnetic Field[kOe]')
plt.ylabel('Resistance[Ω]')
plt.title('magnetoresistance curves of spinvalve junctions')

# グラフを表示する
plt.show()
