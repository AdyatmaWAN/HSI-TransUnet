import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_hsi import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/hsi-data/Test/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=30, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_HSI', help='list dir')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=96, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference(args, model, test_save_path=None):
    # Load the test dataset
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info(f"{len(testloader)} test iterations per epoch")

    # Set the model to evaluation mode
    model.eval()

    # Initialize metrics (Assuming two metrics: dice and kappa for each class)
    metric_list = np.zeros((args.num_classes, 3))  # 3 metrics for each class: dice, kappa, accuracy

    # Iterate over the test data
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # Get the image, label, and case name from the batch
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        h, w = image.size()[2:]  # Assuming image shape is [batch_size, channels, height, width]

        # Move image to the appropriate device (if not already on GPU)
        logging.info(f"input shape: {image.shape}")
        logging.info(f"input shape: {label.shape}")
        image = image.cuda()
        label = label.cuda()

        # Forward pass to get predictions
        with torch.no_grad():
            output = model(image)

        # Compute metrics: Dice, Kappa, etc.
        # Assuming a function compute_metrics that returns dice and kappa for each class
        metric_i = compute_metrics(output, label, args.num_classes)

        # Accumulate the metrics for averaging later
        metric_list += np.array(metric_i)

        # Log metrics for the current batch (per case)
        logging.info(f"idx {i_batch} case {case_name} "
                     f"mean_dice {np.mean(metric_i, axis=0)[0]:.4f} "
                     f"mean_kappa {np.mean(metric_i, axis=0)[1]:.4f}")

    # Average the metrics over the whole dataset
    metric_list /= len(db_test)

    # Log the mean metrics for each class
    for i in range(1, args.num_classes):
        logging.info(f"Mean class {i} mean_dice {metric_list[i-1][0]:.4f} "
                     f"mean_kappa {metric_list[i-1][1]:.4f}")

    # Calculate the overall performance (mean over all classes)
    performance = np.mean(metric_list, axis=0)[0]
    mean_kappa = np.mean(metric_list, axis=0)[1]

    # Log the overall performance
    logging.info(f"Testing performance in best val model: "
                 f"mean_dice: {performance:.4f} mean_kappa: {mean_kappa:.4f}")

    return "Testing Finished!"

def compute_metrics(output, label, num_classes):
    """
    Compute the Dice, Kappa, and Accuracy scores for each class.
    Args:
        output (tensor): The model's predicted output, shape [batch_size, num_classes, height, width].
        label (tensor): The ground truth label, shape [batch_size, height, width].
        num_classes (int): The number of classes in the segmentation task.
    Returns:
        metrics (list): List of tuples containing (Dice, Kappa, Accuracy) for each class.
    """
    dice_scores = []
    kappa_scores = []
    accuracy_scores = []

    for i in range(1, num_classes):  # Skip background (class 0)
        dice = compute_dice(output, label, class_id=i)
        kappa = compute_kappa(output, label, class_id=i)
        accuracy = compute_accuracy(output, label)  # Accuracy for the whole image

        dice_scores.append(dice)
        kappa_scores.append(kappa)
        accuracy_scores.append(accuracy)

    return list(zip(dice_scores, kappa_scores, accuracy_scores))

def compute_dice(output, label, class_id):
    # Compute the Dice score for a specific class
    # Example implementation (you may need to adjust based on your setup)
    output_class = (output.argmax(dim=1) == class_id).float()
    label_class = (label == class_id).float()

    intersection = torch.sum(output_class * label_class)
    dice_score = (2.0 * intersection) / (torch.sum(output_class) + torch.sum(label_class))

    return dice_score.item()

def compute_kappa(output, label, class_id):
    """
    Compute Cohen's Kappa score for a specific class in segmentation task.
    Args:
        output (tensor): The model's predicted output, shape [batch_size, num_classes, height, width].
        label (tensor): The ground truth label, shape [batch_size, height, width].
        class_id (int): The class id to compute the Kappa score for.
    Returns:
        kappa (float): Cohen's Kappa score for the specific class.
    """
    # Flatten the output and label tensors to 1D arrays
    output = output.argmax(dim=1).view(-1)  # Get predicted class (highest probability)
    label = label.view(-1)  # Flatten ground truth label

    # Mask out the class of interest
    output_class = (output == class_id).float()
    label_class = (label == class_id).float()

    # Create confusion matrix for the class (this assumes binary class - class_id vs. others)
    # 0: disagree, 1: agree
    confusion_matrix = torch.zeros(2, 2)
    for i in range(len(output_class)):
        confusion_matrix[int(label_class[i])][int(output_class[i])] += 1

    # Compute observed agreement (P_o)
    total_pixels = confusion_matrix.sum()
    P_o = confusion_matrix[1, 1] / total_pixels  # Number of correctly predicted pixels for class `class_id`

    # Compute expected agreement (P_e)
    P_e = ((confusion_matrix[0, 0] + confusion_matrix[1, 0]) * (confusion_matrix[0, 0] + confusion_matrix[0, 1]) +
           (confusion_matrix[0, 1] + confusion_matrix[1, 1]) * (confusion_matrix[1, 0] + confusion_matrix[1, 1])) / (total_pixels ** 2)

    # Calculate Cohen's Kappa
    if P_e == 1:  # Handle the case where P_e is 1 (which could lead to division by zero)
        kappa = 1.0
    else:
        kappa = (P_o - P_e) / (1 - P_e)

    return kappa.item()

def compute_accuracy(output, label):
    """
    Compute pixel-level accuracy for segmentation task.
    Args:
        output (tensor): The model's predicted output, shape [batch_size, num_classes, height, width].
        label (tensor): The ground truth label, shape [batch_size, height, width].
    Returns:
        accuracy (float): Pixel-level accuracy.
    """
    # Flatten the output and label tensors to 1D arrays
    output = output.argmax(dim=1).view(-1)  # Get predicted class (highest probability)
    label = label.view(-1)  # Flatten ground truth label

    # Compute the number of correct predictions
    correct_pixels = (output == label).sum().float()

    # Compute total number of pixels
    total_pixels = float(label.numel())

    # Calculate accuracy
    accuracy = correct_pixels / total_pixels
    return accuracy.item()

# def inference(args, model, test_save_path=None):
#     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#
#         #predict the image and label then count the dice and kappa
#
#         metric_list += np.array(metric_i)
#         logging.info('idx %d case %s mean_dice %f mean_kappa %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     metric_list = metric_list / len(db_test)
#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_kappa %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_kappa = np.mean(metric_list, axis=0)[1]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_kappa : %f' % (performance, mean_kappa))
#     return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset_config = {
    #     'Synapse': {
    #         'Dataset': Synapse_dataset,
    #         'volume_path': '../data/Synapse/test_vol_h5',
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 9,
    #         'z_spacing': 1,
    #     },
    # }
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 30,
            'z_spacing': 0,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


