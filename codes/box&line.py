
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
# layers = ("ALex_Conv1", "ALex_Conv2", "ALex_Conv3", "ALex_Conv4", "ALex_Conv5", "Alex_Fc1", "Alex_Fc2")


data_layer1_RL=pd.read_csv('box/1/final_output_1.csv')
data_layer1_random=pd.read_csv('box/1/random_output_1.csv')
# data1=data1['Injection Number']
# data2=data2['Injection Number']
data_layer2_RL=pd.read_csv('box/1/final_output_2.csv')
data_layer2_random=pd.read_csv('box/1/random_output_2.csv')

data_layer3_RL=pd.read_csv('box/1/final_output_3.csv')
data_layer3_random=pd.read_csv('box/1/random_output_3.csv')

data_layer4_RL=pd.read_csv('box/1/final_output_4.csv')
data_layer4_random=pd.read_csv('box/1/random_output_4.csv')

data_layer5_RL=pd.read_csv('box/1/final_output_5.csv')
data_layer5_random=pd.read_csv('box/1/random_output_5.csv')

data_layer6_RL=pd.read_csv('box/1/final_output_6.csv')
data_layer6_random=pd.read_csv('box/1/random_output_6.csv')

data_layer7_RL=pd.read_csv('box/1/final_output_7.csv')
data_layer7_random=pd.read_csv('box/1/random_output_7.csv')

data_layer1_RL['layer'] = 'Alex_layer1'
data_layer1_random['layer'] = 'Alex_layer1'
data_layer1_RL['Group'] = 'RL'
data_layer1_random['Group'] = 'Random'

data_layer2_RL['layer'] = 'Alex_layer2'
data_layer2_random['layer'] = 'Alex_layer2'
data_layer2_RL['Group'] = 'RL'
data_layer2_random['Group'] = 'Random'

data_layer3_RL['layer'] = 'Alex_layer3'
data_layer3_random['layer'] = 'Alex_layer3'
data_layer3_RL['Group'] = 'RL'
data_layer3_random['Group'] = 'Random'

data_layer4_RL['layer'] = 'Alex_layer4'
data_layer4_random['layer'] = 'Alex_layer4'
data_layer4_RL['Group'] = 'RL'
data_layer4_random['Group'] = 'Random'

data_layer5_RL['layer'] = 'Alex_layer5'
data_layer5_random['layer'] = 'Alex_layer5'
data_layer5_RL['Group'] = 'RL'
data_layer5_random['Group'] = 'Random'

data_layer6_RL['layer'] = 'Alex_layer6'
data_layer6_random['layer'] = 'Alex_layer6'
data_layer6_RL['Group'] = 'RL'
data_layer6_random['Group'] = 'Random'

data_layer7_RL['layer'] = 'Alex_layer7'
data_layer7_random['layer'] = 'Alex_layer7'
data_layer7_RL['Group'] = 'RL'
data_layer7_random['Group'] = 'Random'

combined_data = pd.concat([data_layer1_RL, data_layer1_random, data_layer2_RL, data_layer2_random,
                    data_layer2_RL, data_layer2_random, data_layer3_RL, data_layer3_random, 
                    data_layer4_RL, data_layer4_random, data_layer5_RL, data_layer5_random,
                    data_layer6_RL, data_layer6_random, data_layer7_RL, data_layer7_random])
print(combined_data)
# data = pd.melt(combined_data, value_name='Injection Number')
# print(data)
# data = df.melt(id_vars=['Categories'], var_name='dataset', value_name='values')
plt.figure(figsize=(15, 10))
# fig, ax = plt.subplots(layout='constrained')
# if 'Values' not in combined_data.columns:
#     raise ValueError("The 'Values' column does not exist in the combined data.")
ax = sns.boxplot(x='layer', y='Injection Number', data=combined_data, hue='Group', width=0.5, showfliers=False)
# sns.boxplot(data=data, y='Injection Number', hue='Group')
# plt.legend(title='dataset', loc='upper left', bbox_to_anchor=(1, 1))
ax.set_ylim(100, 2500)
plt.show()

