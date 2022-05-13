import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

CIFAR100=[
[[75.35 ,0.296128993771939],
[74.48 ,0.253231850475024]],

[[76.18 ,0.215996031709583],
[75.99 ,0.143931765272977]],
[
[76.86 ,0.158375683559242],
[76.66 ,0.190553851381375]]
]
CIFAR100=np.array(CIFAR100) # 3,2,2

plt.rc('font', family='Times New Roman')
x=[r"$\times1$",r"$\times2$",r"$\times4$"]
nums_x=[1,2,3]
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xticks(fontproperties='Times New Roman', size=10)

ax = plt.gca()
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.5)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.annotate('',xy=(1.1,CIFAR100[0,0,0]),xytext=(1,CIFAR100[0,1,0]),arrowprops={'width':1,'headwidth':3,'headlength':7,'shrink':0.15,'facecolor':'#b800c5','ec':'#b800c5'})
plt.annotate('',xy=(2.1,CIFAR100[1,0,0]),xytext=(2,CIFAR100[1,1,0]),arrowprops={'width':1,'headwidth':3,'headlength':7,'shrink':0.15,'facecolor':'#b800c5','ec':'#b800c5'})
plt.annotate('',xy=(3.1,CIFAR100[2,0,0]),xytext=(3,CIFAR100[2,1,0]),arrowprops={'width':1,'headwidth':3,'headlength':7,'shrink':0.15,'facecolor':'#b800c5','ec':'#b800c5'})

for a_x, y in zip([i+0.1 for i in nums_x],CIFAR100[:,0,0].tolist()):
    plt.text(a_x-0.11, y+0.12, '%.2f' % y, ha='center', va='bottom', fontsize=12)
l1=ax.plot([i+0.1 for i in nums_x], CIFAR100[:,0,0].tolist(),'*--', c='r',markersize=14,label='Multi-Policies')
ax.errorbar([i+0.1 for i in nums_x],CIFAR100[:,0,0],\
            yerr=CIFAR100[:,0,1],\
            fmt="*",color="r",ecolor='r',elinewidth=2,capsize=4)
for a_x, y in zip(nums_x,CIFAR100[:,1,0].tolist()):
    if y==76.66:
        plt.text(a_x+0.14, y-0.43, '%.2f' % y, ha='center', va='bottom', fontsize=12)
    else:
        plt.text(a_x+0.14, y-0.33, '%.2f' % y, ha='center', va='bottom', fontsize=12)
l2=ax.plot(nums_x, CIFAR100[:,1,0].tolist(),'*--', c='b',markersize=14,label='Rotating')
ax.errorbar([i for i in nums_x],CIFAR100[:,1,0],\
            yerr=CIFAR100[:,1,1],\
            fmt="*",color="b",ecolor='b',elinewidth=2,capsize=4)
ax.axhline(76.44,c="black",ls='--',lw=1,label="Teacher Model")
plt.legend()
# plt.plot(x_axis_data, mean_average, 'k*--', alpha=0.5, linewidth=1.5)  # 'bo-'表示蓝色实线，数据点实心原点标注
plt.xlabel(r'$n_{step}$', fontsize=12, fontweight='bold')  # x_label
plt.ylabel('Top-1 Test Accuracy [%]', fontsize=12, fontweight='bold')  # y_label
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks([1.05,2.05,3.05],x)
plt.ylim(74, 77.5,0.5)#仅设置y轴坐标范围
plt.savefig("./graph4.png")