import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#9394E7', '#D76364', '#54B345', '#05B9E2']
val_performance_dict = dict()
C10_LOG = [[76.16, 0.17],
           [76.31, 0.19],
           [75.51, 0.13],
           [75.31, 0.09],
           [77.20, 0.17],
           [76.05, 0.11],
           [76.73, 0.20],
           [75.55, 0.12]]

C100_LOG = [[92.73, 0.21],
            [92.73, 0.16],
            [92.56, 0.15],
            [92.72, 0.22],
            [93.46, 0.19],
            [92.88, 0.13],
            [93.22, 0.17],
            [92.68, 0.12]]

fig=plt.figure(figsize=(20,10))
plt.subplots_adjust(wspace=0.0)
tag=1
teacher_acc={'CIFAR-100':76.44,'CIFAR-10':93.56}
for name,LOG in zip(["CIFAR-10","CIFAR-100"],[C100_LOG,C10_LOG]):
    ax=fig.add_subplot(1,2,tag)
    tag+=1
    xs=[r"2$\times$",r"4$\times$"]
    SSKD_WITHSS=[LOG[2],LOG[0]]
    SSKD_WITHOUTSS=[LOG[3],LOG[1]]
    HSAKD_WITHSS=[LOG[6],LOG[4]]
    HSAKD_WITHOUTSS=[LOG[7],LOG[5]]
    "============================"
    ax.fill_between(xs, [SSKD_WITHSS[0][0]-SSKD_WITHSS[0][1],SSKD_WITHSS[1][0]-SSKD_WITHSS[1][1]],
                        [SSKD_WITHSS[0][0]+SSKD_WITHSS[0][1],SSKD_WITHSS[1][0]+SSKD_WITHSS[1][1]],
                     where=[i < j for i, j in zip([SSKD_WITHSS[0][0]-SSKD_WITHSS[0][1],SSKD_WITHSS[1][0]-SSKD_WITHSS[1][1]],
                                                  [SSKD_WITHSS[0][0]+SSKD_WITHSS[0][1],SSKD_WITHSS[1][0]+SSKD_WITHSS[1][1]])], alpha=0.2,
                     interpolate=True, facecolor="#FFC0CB")
    ax.plot(xs, [SSKD_WITHSS[0][0],SSKD_WITHSS[1][0]],"x--", markersize=20,label=r'${}$'.format("SSKD\ with\ \mathcal{L}_{ss}"), color="#DC143C", linewidth=2)
    "============================"
    ax.fill_between(xs, [SSKD_WITHOUTSS[0][0]-SSKD_WITHOUTSS[0][1],SSKD_WITHOUTSS[1][0]-SSKD_WITHOUTSS[1][1]],
                        [SSKD_WITHOUTSS[0][0]+SSKD_WITHOUTSS[0][1],SSKD_WITHOUTSS[1][0]+SSKD_WITHOUTSS[1][1]],
                     where=[i < j for i, j in zip([SSKD_WITHOUTSS[0][0]-SSKD_WITHOUTSS[0][1],SSKD_WITHOUTSS[1][0]-SSKD_WITHOUTSS[1][1]],
                                                  [SSKD_WITHOUTSS[0][0]+SSKD_WITHOUTSS[0][1],SSKD_WITHOUTSS[1][0]+SSKD_WITHOUTSS[1][1]])], alpha=0.2,
                     interpolate=True, facecolor="#9370DB")
    ax.plot(xs, [SSKD_WITHOUTSS[0][0],SSKD_WITHOUTSS[1][0]],"x--",markersize=20, label=r'${}$'.format("SSKD\ without\ \mathcal{L}_{ss}"), color="#8A2BE2", linewidth=2)
    "============================="
    ax.fill_between(xs, [HSAKD_WITHSS[0][0]-HSAKD_WITHSS[0][1],HSAKD_WITHSS[1][0]-HSAKD_WITHSS[1][1]],
                        [HSAKD_WITHSS[0][0]+HSAKD_WITHSS[0][1],HSAKD_WITHSS[1][0]+HSAKD_WITHSS[1][1]],
                     where=[i < j for i, j in zip([HSAKD_WITHSS[0][0]-HSAKD_WITHSS[0][1],HSAKD_WITHSS[1][0]-HSAKD_WITHSS[1][1]],
                                                  [HSAKD_WITHSS[0][0]+HSAKD_WITHSS[0][1],HSAKD_WITHSS[1][0]+HSAKD_WITHSS[1][1]])], alpha=0.2,
                     interpolate=True, facecolor="#7FFFAA")
    ax.plot(xs, [HSAKD_WITHSS[0][0],HSAKD_WITHSS[1][0]],"x--", markersize=20,label=r'${}$'.format("HSAKD\ with\ \mathcal{L}_{kl\_q}"), color="#2E8B57", linewidth=2)
    "============================"
    ax.fill_between(xs, [HSAKD_WITHOUTSS[0][0]-HSAKD_WITHOUTSS[0][1],HSAKD_WITHOUTSS[1][0]-HSAKD_WITHOUTSS[1][1]],
                        [HSAKD_WITHOUTSS[0][0]+HSAKD_WITHOUTSS[0][1],HSAKD_WITHOUTSS[1][0]+HSAKD_WITHOUTSS[1][1]],
                     where=[i < j for i, j in zip([HSAKD_WITHOUTSS[0][0]-HSAKD_WITHOUTSS[0][1],HSAKD_WITHOUTSS[1][0]-HSAKD_WITHOUTSS[1][1]],
                                                  [HSAKD_WITHOUTSS[0][0]+HSAKD_WITHOUTSS[0][1],HSAKD_WITHOUTSS[1][0]+HSAKD_WITHOUTSS[1][1]])], alpha=0.2,
                     interpolate=True, facecolor="#C0C0C0")
    ax.plot(xs, [HSAKD_WITHOUTSS[0][0],HSAKD_WITHOUTSS[1][0]],"x--",markersize=20, label=r'${}$'.format("HSAKD\ without\ \mathcal{L}_{kl\_q}"), color="#000000", linewidth=2)
    "============================="
    font1 = {'family':"Times New Roman",
             'weight': 'bold',
             'style': 'normal',
             'size': 30,
             }
    ax.axhline(teacher_acc[name], c="black", ls='--', lw=2, label="Teacher Model")
    plt.tick_params(labelsize=25)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontstyle('normal') for label in labels]
    ax.patch.set_facecolor("white")
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.set_title(name,font1)
    ax.legend(loc=2,fontsize=20)
    plt.xlabel('$n_{step}$',font1)
    plt.ylabel('Top-1 Test Accuracy [%]',font1)
# plt.xlim(80, 188)
# plt.ylim(82, 93)
plt.tight_layout()
plt.savefig("./graph1.png")
