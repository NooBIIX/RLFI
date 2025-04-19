import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 200
# Read data from CSV file
data0 = pd.read_csv('final_output_0.csv')
data1 = pd.read_csv('final_output_1.csv')
# data2 = pd.read_csv('final_output_2.csv')
# print(data)
# print(data['Epoch'])
episode = data0['Episode']
#total_reward = data['Total Reward']
average_reward_0 = data0['Average Reward']
accuracy_0 = data0['Final Accuracy']
injection_number_0 = data0['Injection Number']

average_reward_1 = data1['Average Reward']
accuracy_1 = data1['Final Accuracy']
injection_number_1 = data1['Injection Number']
# ######## 3
# average_reward_2 = data2['Average Reward']
# accuracy_2 = data2['Final Accuracy']
# injection_number_2 = data2['Injection Number']

# max_value_reward = pd.concat([average_reward_0, average_reward_1, average_reward_2], axis=1).max(axis=1)
# min_value_reward = pd.concat([average_reward_0, average_reward_1, average_reward_2], axis=1).min(axis=1)
# mean_value_reward = pd.concat([average_reward_0, average_reward_1, average_reward_2], axis=1).mean(axis=1)

# max_value_accuracy = pd.concat([accuracy_0, accuracy_1, accuracy_2], axis=1).max(axis=1)
# min_value_accuracy = pd.concat([accuracy_0, accuracy_1, accuracy_2], axis=1).min(axis=1)
# mean_value_accuracy = pd.concat([accuracy_0, accuracy_1, accuracy_2], axis=1).mean(axis=1)

# max_value_number = pd.concat([injection_number_0, injection_number_1, injection_number_2], axis=1).max(axis=1)
# min_value_number = pd.concat([injection_number_0, injection_number_1, injection_number_2], axis=1).min(axis=1)
# mean_value_number = pd.concat([injection_number_0, injection_number_1, injection_number_2], axis=1).mean(axis=1)

###### 2
max_value_reward = pd.concat([average_reward_0, average_reward_1], axis=1).max(axis=1)
min_value_reward = pd.concat([average_reward_0, average_reward_1], axis=1).min(axis=1)
mean_value_reward = pd.concat([average_reward_0, average_reward_1], axis=1).mean(axis=1)

max_value_accuracy = pd.concat([accuracy_0, accuracy_1], axis=1).max(axis=1)
min_value_accuracy = pd.concat([accuracy_0, accuracy_1], axis=1).min(axis=1)
mean_value_accuracy = pd.concat([accuracy_0, accuracy_1], axis=1).mean(axis=1)

max_value_number = pd.concat([injection_number_0, injection_number_1], axis=1).max(axis=1) 
min_value_number = pd.concat([injection_number_0, injection_number_1], axis=1).min(axis=1) 
mean_value_number = pd.concat([injection_number_0, injection_number_1], axis=1).mean(axis=1)

#### number reduction
# max_value_number = mean_value_number + abs(max_value_number-mean_value_number) * 0.3
# min_value_number = mean_value_number - abs(min_value_number-mean_value_number) * 0.3
random_data_0 = pd.read_csv('random_output_0.csv')
random_data_1 = pd.read_csv('random_output_1.csv')
# random_data_2 = pd.read_csv('random_output_2.csv')

random_accuracy_0 = random_data_0['Final Accuracy']
random_accuracy_1 = random_data_1['Final Accuracy']
# random_accuracy_2 = random_data_2['Final Accuracy']

# max_value_accuracy_Random = pd.concat([random_accuracy_0, random_accuracy_1, random_accuracy_2], axis=1).max(axis=1)
# min_value_accuracy_Random = pd.concat([random_accuracy_0, random_accuracy_1, random_accuracy_2], axis=1).min(axis=1)
# mean_value_accuracy_Random = pd.concat([random_accuracy_0, random_accuracy_1, random_accuracy_2], axis=1).mean(axis=1)

max_value_accuracy_Random = pd.concat([random_accuracy_0, random_accuracy_1], axis=1).max(axis=1)
min_value_accuracy_Random = pd.concat([random_accuracy_0, random_accuracy_1], axis=1).min(axis=1)
mean_value_accuracy_Random = pd.concat([random_accuracy_0, random_accuracy_1], axis=1).mean(axis=1)

# # Calculate the root mean square
# rms_value = (pd.concat([average_reward_0**2, average_reward_1**2], axis=1).mean(axis=1))**0.5

figure, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(episode, max_value_reward, linestyle='None', color='lightblue', markeredgecolor='lightblue', label='Max')
ax1.plot(episode, min_value_reward, linestyle='None', color='lightblue', markeredgecolor='lightblue', label='Min')
ax1.plot(episode, mean_value_reward, alpha=0.6)
ax1.fill_between(episode, min_value_reward, max_value_reward, color='lightblue')
ax1.set_title("Reward")
ax2.plot(episode, max_value_accuracy, linestyle='None', color='lightblue', markeredgecolor='lightblue')
ax2.plot(episode, min_value_accuracy, linestyle='None', color='lightblue', markeredgecolor='lightblue')
line1=ax2.plot(episode, mean_value_accuracy, alpha=0.6, label='RL')
ax2.fill_between(episode, min_value_accuracy, max_value_accuracy, color='lightblue')
# ax2.axhline(y=0.8, linestyle=":", color="r")

ax2.plot(episode, max_value_accuracy_Random, linestyle='None', color='lightgreen', markeredgecolor='lightblue')
ax2.plot(episode, min_value_accuracy_Random, linestyle='None', color='lightgreen', markeredgecolor='lightblue')
line2=ax2.plot(episode, mean_value_accuracy_Random, alpha=0.6, color= 'green', label='Random')
ax2.fill_between(episode, min_value_accuracy_Random, max_value_accuracy_Random, color='lightgreen')

ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0, fontsize='small', title_fontsize='small')
ax2.set_title("Performance")
ax3.plot(episode, max_value_number, linestyle='None', color='lightblue', markeredgecolor='lightblue', label='Max')
ax3.plot(episode, min_value_number, linestyle='None', color='lightblue', markeredgecolor='lightblue', label='Min')
ax3.plot(episode, mean_value_number, alpha=0.6)
ax3.fill_between(episode, min_value_number, max_value_number, color='lightblue')
ax3.set_title("Injection Number")
ax3.set(xlabel='Episodes')

plt.subplots_adjust(left= 0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
plt.show()
figure.savefig('layer14.png')
# # Plotting the lines and filling the area
# plt.plot(episode, max_value, linestyle='None', color='lightblue', markeredgecolor='lightblue', label='Max')
# plt.plot(episode, min_value, linestyle='None', color='lightblue', markeredgecolor='lightblue', label='Min')
# plt.plot(episode, mean_value, label='RMS')
# plt.fill_between(episode, min_value, max_value, color='lightblue')

# # Adding legend, labels, and title

# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Max, Min, and RMS Values')

# # Displaying the plot
# plt.grid(True)
# plt.show()




