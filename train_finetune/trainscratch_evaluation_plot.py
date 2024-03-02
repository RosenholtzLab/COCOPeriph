import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# eval_metric = 'APs'

# train_eccs = [0,5,10,15]
val_eccs = [0,5,10,15,20]

parser = argparse.ArgumentParser(description="path save plot")
parser.add_argument(
       "--output-path",
       type=str,
       default=f'./output/Detectron2_RCNN_FPN_trainscratch_bbox_evaluation.png',
       help="path to save plot",
   )

parser.add_argument(
       "--metric",
       type=str,
       default='AP',
       help="Type of evaluation metric (AP,APs,APl,AP50,AP75)",
   )

parser.add_argument('--train-eccs','--list', action='append', help='<Required> Set flag', required=True)
args = parser.parse_args()
train_eccs = [int(ecc) for ecc in args.train_eccs]
eval_metric = args.metric
# print(train_eccs)

#may need to change this in this and other finetune/eval/plot scripts!!
base_path = '.'

eval_results = {}
for train_ecc in train_eccs:
    # print(train_ecc)
    eval_results[train_ecc] = []
    for ecc in val_eccs:
        # print(ecc)
        if train_ecc == -1:
            val_results_path = f'{base_path}/baseline_pretrained_detectron2_eval_results/coco_val_2017_ecc{ecc}_eval/val_results.pkl'
        elif train_ecc == -2:
            val_results_path = f'{base_path}/trainscratch_ecc100/coco_val_2017_ecc{ecc}_eval/val_results.pkl'
            
        else:
            val_results_path = f'{base_path}/output_mixed_data_35000/finetune_ecc{train_ecc}/coco_val_2017_ecc{ecc}_eval/val_results.pkl'
        with open(val_results_path, 'rb') as f:
                val_results = pickle.load(f)
        eval_results[train_ecc].append(val_results['bbox'][eval_metric])


AP_labels = [0,5,10,15,20]
sns.set()
plt.figure(figsize=(8,6))
for train_ecc in train_eccs:
    if train_ecc==-1:
        plt.plot(AP_labels, eval_results[train_ecc],'--o',color='k',alpha=0.85,linewidth=2,markersize=10,label=f'baseline')
    elif train_ecc==-2:
        plt.plot(AP_labels, eval_results[train_ecc],'-o',color='m',alpha=0.85,linewidth=2,markersize=10,label=f'trainscratch')
    elif train_ecc==0:
        plt.plot(AP_labels, eval_results[train_ecc],'-o',color='grey',alpha=0.85,linewidth=2,markersize=10,label=f'finetune_ecc0')
    elif train_ecc==100:
        plt.plot(AP_labels, eval_results[train_ecc],'-o',linewidth=2,markersize=10,alpha=0.85,label=f'finetune_all_eccs')
    else:
        plt.plot(AP_labels, eval_results[train_ecc],'-o',linewidth=2,markersize=10,alpha=0.85,label=f'finetune_ecc{train_ecc}')
# plt.plot(AP_labels, AP_data,'--o',linewidth=4,markersize=12, color='green')
plt.xticks(AP_labels,fontsize=16)
plt.yticks(fontsize=16)
# plt.title(title,fontsize=24,y=1.08)
#plt.ylim([0,50])
plt.legend()
# plt.grid()
plt.xlabel('Uniform Mongrel Eccentricity ($^\circ$)', fontsize=18)
plt.ylabel(f'Bounding Box Average Precision ({eval_metric})',fontsize=18)

#add type of evalution to filename
args.output_path = args.output_path.replace('.png',f'_{eval_metric}.png')

plt.savefig(args.output_path,dpi=300)
