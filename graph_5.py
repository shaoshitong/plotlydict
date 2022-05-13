import argparse
import os
import matplotlib.pyplot as plt
font1 = {'family':"Times New Roman",
         'weight': 'bold',
         'style': 'normal',
         'size': 30,
         }
plt.rc('font',**font1)
import seaborn as sns
import numpy as np
mean_average = [85.12590667,84.74069333,84.29626667,85.12592, 86.01476,84.59256,
 84.29630667, 84.65185333, 85.83705333, 85.18517333, 85.24441333,84.65181333,
 84.17778667,84.68146667]
mean_std = [0.551,1.439,0.761,1.287,1.213,0.956,1.167,0.841,0.77,0.854,0.305,0.862,1.037,0.551]

# colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#9394E7', '#D76364', '#54B345', '#05B9E2']
val_performance_dict = dict()

begin=(1,0,0)
end=(0,0,1)
end_begin=(-1,0,1)
colors=[(begin[0]+end_begin[0]*i/len(mean_std),begin[1]+end_begin[1]*i/len(mean_std),begin[2]+end_begin[2]*i/len(mean_std)) for i in range(len(mean_std))]
fig=plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace=0.0)
ax=fig.add_subplot(111)
tag=1
plt.grid(b=True, which='major', axis='both',c='black', linestyle='-.')
x=np.arange(0,len(mean_std))

"============================"

ax.fill_between(x, [i-j for i,j in zip(mean_average,mean_std)],
                    [i+j for i,j in zip(mean_average,mean_std)],
                 where=[i-j < i+j for i, j in zip(mean_average,mean_std)], alpha=0.15,
                 interpolate=True, facecolor='r')
ax.plot(x, mean_average,"*--", markersize=14, label='Top-h Classifiers on SEED',c='r', linewidth=2)
# ax.scatter(x,mean_average,s=20,cmap='RdPu',label='Top-h Classifiers on SEED')

plt.tick_params(labelsize=25)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontstyle('normal') for label in labels]
ax.patch.set_facecolor("white")
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.set_facecolor('white')
ax.set_title("",font1)
ax.legend(loc='best',fontsize=20,)
plt.xlabel('Top Number',font1)
plt.ylabel('Top-1 Test Accuracy [%]',font1)
plt.xticks(x,[ r'Top {}'.format(i+1) for i in x],rotation=60)
plt.yticks([round(i,2) for i in np.linspace(83,87,14)])
plt.xlim(-0.5,13.5)
plt.ylim(83,87)
plt.tight_layout()
plt.savefig("./graph5.png")