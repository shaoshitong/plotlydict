import math,sys,os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

mm_list=np.load("./mm_list.npy")
ax = plt.gca()
# x_major_locator = MultipleLocator(1)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
counter_list=[[] for i in range(14)]
for i in range(mm_list.shape[0]):
    a,b=mm_list[i,0],mm_list[i,1]
    counter_list[int(a.item())].append(b.item())
print(counter_list)
for i in range(len(counter_list)):
    counter_list[i]=sum(counter_list[i])/(len(counter_list[i])+1)
rgb=(177/255,212/255,219/255)
plt.scatter(mm_list[:,0],mm_list[:,1],s=5, alpha=0.3, cmap='viridis',c=rgb)
plt.plot([i for i in range(14)],counter_list,"*b--",markersize=14,label='Mean Value Change Curve')



plt.legend()
# plt.plot(x_axis_data, mean_average, 'k*--', alpha=0.5, linewidth=1.5)  # 'bo-'表示蓝色实线，数据点实心原点标注
plt.xlabel(r'$Strength of Data Augmentation$', fontsize=12, fontweight='bold')  # x_label
plt.ylabel('Cosine Confidence Weight', fontsize=12, fontweight='bold')  # y_label
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
# plt.xticks([1.05,2.05,3.05],x)
# plt.ylim(74, 77.5,0.5)#仅设置y轴坐标范围
plt.savefig("./graph_11.png")
