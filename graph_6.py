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

def read_files(file_path):
    label = os.path.basename(file_path)
    with open(os.path.expanduser(file_path), 'r') as fp:
        log = ([line.strip() for line in fp], label)
    return log
def extract_val_acc(message, acc1_str='Acc@1 '):
    if acc1_str not in message:
        return None
    acc1 = float(message[message.find(acc1_str) + len(acc1_str):])
    return acc1
def extract_val_performance(log_lines):
    val_acc1_list =  list()
    for line in log_lines:
        elements = line.split('\t')
        if len(elements) < 3:
            continue
        message = elements[3]
        val_acc1 = extract_val_acc(message)
        if isinstance(val_acc1, float):
            val_acc1_list.append(val_acc1)
        if 'Training time' in message:
            break
    return val_acc1_list


def get_data_from_txt(file_path:str):
    policy_ratio=float(file_path.split(".txt")[0][-5:-2])
    log,label=read_files(file_path)
    val_acc1_list=extract_val_performance(log)
    return val_acc1_list,policy_ratio
get_data_from_txt("./results/resnet18_from_resnet50_2倍_policy_0.0_1.txt")

def get_one_policy_ratio_list(policy_ratio):
    val_acc1_list2=[]
    for i in range(3):
        file_path=f"./results/resnet18_from_resnet50_2倍_policy_{policy_ratio}_{i+1}.txt"
        print(file_path)
        val_acc1_list,_=get_data_from_txt(file_path)
        val_acc1_list2.append(val_acc1_list)
    val_acc1_list2=np.array(val_acc1_list2)
    std,mean=val_acc1_list2[:,-5:].std(),val_acc1_list2[:,-5:].mean()
    return std,mean
def get_all_policy_ratio_list():
    std_list=[]
    mean_list=[]
    x=['0.0','0.2','0.4','0.6','0.8','1.0']
    for ratio in x:
        std,mean=get_one_policy_ratio_list(ratio)
        std_list.append(std)
        mean_list.append(mean)
    return x,std_list,mean_list


x,std_list,mean_list=get_all_policy_ratio_list()
mean_average = mean_list
mean_std = std_list

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

"============================"

ax.fill_between(x, [i-j for i,j in zip(mean_average,mean_std)],
                    [i+j for i,j in zip(mean_average,mean_std)],
                 where=[i-j < i+j for i, j in zip(mean_average,mean_std)], alpha=0.15,
                 interpolate=True, facecolor='r')
ax.plot(x, mean_average,"*--", markersize=6, label='Curve on CIFAR-10',c='r', linewidth=2)
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
plt.xlabel('Ratio (p)',font1)
plt.ylabel('Top-1 Test Accuracy [%]',font1)
plt.xticks(x,[ r'{}'.format(round(float(i),2)) for i in x],rotation=60)
plt.yticks([round(i,2) for i in np.linspace(92,94,6)])
plt.xlim(-0.1,5.1)
plt.ylim(92,94)
plt.tight_layout()
plt.savefig("./graph6.png")