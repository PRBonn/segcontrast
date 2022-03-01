import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# font stuff
rcParams['hatch.linewidth'] = 5.0
rcParams.update({'font.size': 7})
rcParams.update({'pdf.fonttype': 42})
# rcParams['text.usetex'] = True
linewidth_latex = 245.71811

PATH = Path(__file__).parent.absolute()

def save(name):
    plt.tight_layout(pad=0.0, rect=[0.01, 0.01, 0.99, 0.99])
    plt.savefig(os.path.join(PATH, '/../pics' + name+'.pdf'))

def point2inch(x):
    return x / 72.

def plot_all_checkpoints():
    lbl_percent = [0.1, 1., 10., 50., 100.]

    scratch = [
        [7.56, 17.74, 23.54, 26.09, 26.34],
        [16.49 ,27.19 ,33.32, 40.25, 41.70],
        [25.72, 34.31, 38.93, 51.89, 53.87],
        [33.72, 40.78, 42.04, 51.27, 58.34],
        [32.11, 38.90, 44.83, 56.02, 59.63],
    ]

    depth = [
        [24.17, 27.79, 30.62, 32.05, 32.18],
        [29.47, 31.10, 38.60, 44.36, 46.41],
        [29.57, 34.13, 43.55, 50.49, 56.29],
        [35.64, 37.01, 42.77, 55.20, 58.54],
        [35.02, 37.72, 42.53, 54.29, 59.88],
    ]

    point = [
        [9.91, 22.61, 24.38, 27.59, 27.89],
        [18.90, 27.42, 34.38, 41.40, 43.40],
        [25.99, 35.42, 38.05, 50.61, 53.79],
        [32.14, 37.68, 42.44, 50.61, 57.30],
        [35.84, 37.34, 43.74, 53.29, 59.77],
    ]

    segment = [
        [22.79, 28.60, 31.60, 32.70, 32.86],
        [28.46, 28.58, 40.22, 45.53, 47.41],
        [33.35, 37.02, 41.62, 53.36, 55.21],
        [37.44, 40.41, 44.11, 54.48, 58.33],
        [41.32, 39.39, 42.70, 55.49, 60.53],
    ]

    for i in range(len(lbl_percent)):
        plt.xlim(0., 5.)
        plt.xlabel('Checkpoint step')

        plt.ylim(np.min([scratch[i][0], depth[i][0], point[i][0], segment[i][0]]) - 3., np.max([scratch[i][-1], depth[i][-1], point[i][-1], segment[i][-1]]) + 3.)
        plt.ylabel('mIoU (%)')

        plt.plot(range(len(scratch[i])), scratch[i], '.-', color='b', label='scratch')
        plt.plot(range(len(depth[i])), depth[i], '^-', color='r', label='depth contrast')
        plt.plot(range(len(point[i])), point[i], 's-', color='c', label='point contrast')
        plt.plot(range(len(segment[i])), segment[i], '*-', color='g', label='segment contrast')
        plt.plot([0.,100.], [scratch[i][-1], scratch[i][-1]], '--')
        plt.legend(loc='lower right', borderaxespad=0.)
        plt.title(f"{lbl_percent[i]}% labels")

        plt.show()


def plot_final_res():
    # just for better visualization
    plt.figure(figsize=(point2inch(linewidth_latex), 0.7*point2inch(linewidth_latex)))
    lbl_percent = [0., 1., 2., 3., 4.]

    scratch = [25.59, 41.70, 53.87, 58.34, 59.63]
    depth = [33.51, 46.41, 56.29, 58.54, 59.88]
    point = [28.52, 43.40, 53.79, 57.30, 59.77]
    segment = [34.78, 47.41, 55.21, 58.33, 60.53]

    plt.xlim(lbl_percent[0], lbl_percent[-1])
    plt.xlabel('Labels (%)')
    plt.xticks(lbl_percent, ['0.1', '1', '10', '50', '100'])

    plt.ylim(np.min([scratch[0], depth[0], point[0], segment[0]]) - 1., np.max([scratch[-1], depth[-1], point[-1], segment[-1]]) + 1.)
    plt.ylabel('mIoU (%)')
    plt.yticks([25, 30, 35, 40, 45, 50, 55, 60], ['25', '30', '35', '40', '45', '50', '55', '60'])

    plt.plot(lbl_percent, scratch, '.-', color='b', label='Scratch', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(lbl_percent, depth, '.-', color='r', label='DepthContrast', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(lbl_percent, point, '.-', color='c', label='PointContrast', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(lbl_percent, segment, '.-', color='g', label='SegmentContrast(ours)', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot([0.,100.], [scratch[-1], scratch[-1]], '--', linewidth=2.5, color='b')
    plt.legend(loc='lower right', borderaxespad=0.)
    plt.title('Contrastive SSL methods vs scratch')

    plt.grid()
    plt.show()

def plot_pretrain_epochs():
    #plt.rcParams["figure.figsize"] = (5,5)
    plt.figure(figsize=(point2inch(linewidth_latex), point2inch(linewidth_latex)))

    epoch = [10, 50, 100, 150, 200]

    point = [30.30, 31.91, 28.62, 30.81, 28.52]
    depth = [27.20, 30.57, 33.01, 33.19, 33.51]
    segment = [29.36, 33.67, 33.85, 34.36, 34.78]

    plt.xlim(0., 200.)
    plt.xlabel('Pre-training epochs')
    plt.xticks(epoch, ['10', '50', '100', '150', '200'])

    plt.ylim(26. - 1., 35. + .5)
    plt.ylabel('mIoU (%)')
    #plt.yticks([252, 35, 45, 55, 60], ['25', '35', '45', '55', '60'])

    #plt.plot(lbl_percent, scratch, '.-', color='b', label='scratch', linewidth=2, markersize=10)
    plt.plot(epoch, depth, '.-', color='r', label='DepthContrast', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1), zorder=10)
    plt.plot(epoch, point, '.-', color='c', label='PointContrast', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1), zorder=9)
    plt.plot(epoch, segment, '.-', color='g', label='SegmentContrast(ours)', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1), zorder=11)
    #plt.arrow(198., 33.51, -148., 0., width=.15, length_includes_head=True, head_length=8.0, head_width=.45, overhang=0.2, alpha=2.0, edgecolor='white', facecolor='dimgrey', zorder=8)
    #plt.annotate('4x less pre-training', [120., 33.7], zorder=7)
    plt.plot([0.,200.], [33.51, 33.51], '--', color='r', linewidth=2.5, zorder=0)
    plt.plot([0.,200.], [25.59, 25.59], '--', color='b', linewidth=2.5, label='Without pre-training')
    plt.legend(loc='lower right', borderaxespad=0.)
    plt.title('Pre-training checkpoints fine-tuned to 0.1% of labels')

    plt.grid()
    #plt.show()
    save('earlier_eval')

def plot_semantic_poss():
    plt.figure(figsize=(point2inch(linewidth_latex), point2inch(linewidth_latex)))
    epoch = [1, 4, 8, 12, 15]

    scratch = [42.12, 56.58, 60.36, 63.51, 64.22]
    point = [39.88, 55.99, 61.97, 64.17, 64.30]
    depth = [50.94, 61.98, 63.14, 64.74, 64.65]
    segment = [54.29, 61.81, 63.23, 64.75, 64.86]
    semkitti = [54.24, 57.34, 62.93, 64.04, 64.54]

    plt.xlim(epoch[0], epoch[-1])
    plt.xlabel('Training epochs')
    plt.xticks(epoch, ['1', '4', '8', '12', '15'])

    plt.ylim(np.min([scratch[0], depth[0], point[0], segment[0]]) - .5, np.max([scratch[-1], depth[-1], point[-1], segment[-1]]) + .5)
    plt.ylabel('mIoU (%)')
    #plt.yticks([25, 30, 35, 40, 45, 50, 55, 60], ['25', '30', '35', '40', '45', '50', '55', '60'])

    plt.plot(epoch, scratch, '.-', color='b', label='Without pre-training', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(epoch, depth, '.-', color='r', label='DepthContrast', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(epoch, point, '.-', color='c', label='PointContrast', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(epoch, semkitti, '.-', color='y', label='Sup. SemKITTI pre-training', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot(epoch, segment, '.-', color='g', label='SegmentContrast(ours)', linewidth=2.5, markersize=12, markeredgecolor=(1,1,1,1))
    plt.plot([0.,100.], [scratch[-1], scratch[-1]], '--', color='b', linewidth=2.5)
    plt.legend(loc='lower right', borderaxespad=0.)
    plt.title('SemanticPOSS fine-tuning')

    plt.grid()
    #plt.show()
    save('semantic_poss')

plot_pretrain_epochs()
plot_semantic_poss()
