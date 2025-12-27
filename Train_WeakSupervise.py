import random
import numpy as np
import torch
import os
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch
import numpy as np
import time
import shutil
import argparse
import datetime
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from Utils.modelfactory import ModelFactory
from Utils.metrics import Evaluator, TensorBoardLogger
from Utils.dataset import CustomDataset, DatasetAnalyzer, CAOISPRSDataset
import torch.optim as optim
from Framework.LabelCorrect import (train_correction_zhao, val_correction_zhao,
                                  adaptive_label_update, get_lr_scheduler_zhao)

class Trainer:
    
    def __init__(self, args):
        self.args = args
        args.model = 'labelcorrect'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.use_amp = args.mixed_precision and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"Using mixed precision training ")
        
        self.results_dir = args.results_dir
        self.checkpoints_dir = args.checkpoints_save
        self.logs_dir = args.logs_dir
        self.plots_dir = args.plots_dir 
        
        now = datetime.datetime.now()
        date = now.strftime("%Y%m%d")
        hour = now.strftime("%H")
        minute = now.strftime("%M")
        self.experiment_name = f"{date}_h{hour}m{minute}_{args.model}_lr{args.learning_rate}"
        print(f"Experiment name: {self.experiment_name}")
        
        hyperparams = {
            'model': args.model,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'classNum': args.classNum,
            'mixed_precision': args.mixed_precision,
        }
        
        self.logger = TensorBoardLogger(
            experiment_name=self.experiment_name,
            hyperparams=hyperparams,
            log_dir=self.logs_dir,
            results_dir=self.results_dir
        )

        warmup = CAOISPRSDataset(
            image_dir=args.train_image_folder,
            label_dir=args.train_label_folder,
            update_label_dir=None,
            mode='train',
            mean_list=args.MeanList,
            std_list=args.StdList,
        )
        
        self.zhao_warmup = DataLoader(
            warmup,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloadworkers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        update = CAOISPRSDataset(
            image_dir=args.train_image_folder,
            label_dir=args.train_label_folder,
            update_label_dir=None,
            mode='val',
            mean_list=args.MeanList,
            std_list=args.StdList,
        )
        
        self.zhao_update = DataLoader(
            update,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloadworkers,
            pin_memory=True,
            drop_last=False,
        )

        val_dataset = CAOISPRSDataset(
            image_dir=args.valid_image_folder,
            label_dir=args.valid_label_folder,
            update_label_dir=None,
            mode='val',
            mean_list=args.MeanList,
            std_list=args.StdList,
        )
    
        self.zhao_valid = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloadworkers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2
        )    
        
        self.model = ModelFactory.create_model(
            args.model,
            classNum=args.classNum,
            bands=args.bands,
            input_size=args.input_size,
            device=self.device
        )
        
        self.optimizer = ModelFactory.create_optimizer(
            args.model, self.model, args
        )
        
        self.iters_per_epoch = len(self.zhao_warmup)
        self.max_iters = args.max_epochs * self.iters_per_epoch
        
        self.metric = Evaluator(num_classes=args.classNum, device=self.device)
        
        if self.args.monitor_metric.endswith('loss'):
            self.best_pred = float('inf')
        else:
            self.best_pred = float('-inf')
            
        self.patience = args.patience
        self.early_stop_counter = 0
        self.early_stop = False

    def train_one_epoch_labelcorrect(self, epoch):
        save_dir = os.path.join(self.args.save_dir, "train_probability_maps")

        if epoch > 0 and epoch < self.args.warmup_epochs:
            print(f"The length of IoU history records : {len(self.iou_history)}")

            iters_per_epoch = len(self.zhao_warmup)
            
            update_epoch = adaptive_label_update(
                self.iou_history, 
                current_epoch=epoch, 
                threshold=0.90,
                eval_interval=20,
                num_iter_per_epoch=iters_per_epoch
            )
            print(f"Warmup need {update_epoch} epoch")

        training_results = train_correction_zhao(
            model=self.model,
            warmup_dataloader=self.zhao_warmup,
            update_dataloader=self.zhao_update,
            optimizer=self.optimizer, 
            device=self.device, 
            epoch=epoch, 
            metric=self.metric, 
            warmup_epoch=self.args.warmup_epochs, 
            correct_epoch=self.args.correct_epochs,
            max_epochs=self.args.max_epochs, 
            use_amp=self.use_amp, 
            save_dir=save_dir,
            eval_interval=20,
            updatelabel_dir=os.path.join(self.args.save_dir,'update_label'),
            args=self.args,
        )
        
        train_loss = training_results['loss']
        metrics = training_results['metrics']
        
        self.logger.log_learning_rate(self.optimizer.param_groups[0]['lr'], epoch)
        metrics_for_logging = {k: v for k, v in metrics.items() if k != 'iou_history'}
        self.logger.log_metrics(metrics_for_logging, epoch, prefix='train')
        
        if not hasattr(self, 'iou_history'):
            self.iou_history = []
            print("Initialize IoU history record")
        if 'iou_history' in metrics and metrics['iou_history']:
            self.iou_history.extend(metrics['iou_history'])
    
        return (train_loss, metrics['acc'], metrics['f1_score'], metrics['f1_score_Back'], metrics['iou'], metrics['iou_Back'],
                metrics['precision'], metrics['recall'], metrics['kappa'])


    def valid_one_epoch_labelcorrect(self, epoch):
        save_dir = os.path.join(self.args.save_dir, "val_probability_maps")
        
        validation_results = val_correction_zhao(
            model=self.model,
            dataloader=self.zhao_valid,
            device=self.device,
            epoch=epoch,
            metric=self.metric,
            use_amp=self.use_amp,
            save_dir=save_dir,
            max_epochs=self.args.max_epochs
        )
        
        valid_loss = validation_results['loss']
        metrics = validation_results['metrics']
        
        self.logger.log_metrics(metrics, epoch, prefix='val')
        
        return (valid_loss, metrics['acc'], metrics['f1_score'], 
                metrics['f1_score_Back'], metrics['iou'], metrics['iou_Back'],
                metrics['precision'], metrics['recall'], metrics['kappa'])
    
                    
    def check_improvement(self, metric_value, metric_name, epoch):
        is_best = False
        
        if metric_name.endswith('loss'):
            improved = metric_value < self.best_pred
        else:
            improved = metric_value > self.best_pred
            
        if improved:
            is_best = True
            self.best_pred = metric_value
            self.early_stop_counter = 0
            print(f"Model performance improved! Best {metric_name}: {self.best_pred:.6f}")
        else:
            if epoch >= self.args.correct_epochs:
                self.early_stop_counter += 1
                print(f"Model performance did not improve. Early stopping counter: {self.early_stop_counter}/{self.patience}")
                if self.early_stop_counter >= self.patience:
                    self.early_stop = True
                    print(f"Early stopping condition met, stopping training")
            else:
                phase = "warmup" if epoch < self.args.warmup_epochs else "correction"
                print(f"Model performance did not improve, but still in {phase} phase. Early stopping not activated yet.")
                
        return is_best
        
    def save_checkpoint(self, epoch, is_best=False):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
            
        filename = os.path.join(self.checkpoints_dir, 'model_checkpoint.pth')

        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_pred': self.best_pred,
            'hyperparameters': {
                'model': self.args.model,
                'classNum': self.args.classNum,
                'bands': self.args.bands,
                'input_size': self.args.input_size,
                'MeanList': self.args.MeanList,
                'StdList': self.args.StdList,
                'mixed_precision': self.args.mixed_precision,
            }
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler'] = self.lr_scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['amp_scaler'] = self.scaler.state_dict()

        torch.save(checkpoint, filename)
        if epoch < self.args.warmup_epochs:
            warmupepoch_filename = os.path.join(self.checkpoints_dir, 'warmup_epoch.pth')
            shutil.copyfile(filename, warmupepoch_filename)
        if is_best:
            best_filename = os.path.join(self.checkpoints_dir, 'best_model_checkpoint.pth')
            shutil.copyfile(filename, best_filename)
            print(f"Saved best model checkpoint: {best_filename}")
            
    def train(self):
        print(f"Starting training: from epoch {self.args.start_epochs} to {self.args.max_epochs}")
        print(f"Using device: {self.device}")
        
        train_epoch_fn = self.train_one_epoch_labelcorrect
        valid_epoch_fn = self.valid_one_epoch_labelcorrect
        
        start_time = time.time()
        
        self.lr_scheduler = get_lr_scheduler_zhao(
            self.optimizer, 
            warmup_epochs=self.args.warmup_epochs, 
            correct_epochs=self.args.correct_epochs,
            max_epochs=self.args.max_epochs
        )
        
        for epoch in range(self.args.start_epochs, self.args.max_epochs):
            if self.early_stop:
                print(f"Early stopping training: epoch {epoch}")
                break
                
            epoch_start_time = time.time()
            
            train_metrics = train_epoch_fn(epoch)
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            valid_metrics = valid_epoch_fn(epoch)
            
            valid_loss, valid_acc, valid_F1score, valid_F1score_Back, valid_IoU, valid_IoU_Back, valid_Precision, valid_Recall, valid_Kappa = valid_metrics
                
            epoch_time = time.time() - epoch_start_time
            minutes, seconds = divmod(epoch_time, 60)
            time_str = f"{int(minutes)}m {int(seconds)}s"
        
            self.logger.log_epoch_time(time_str, epoch)
            
            if self.args.monitor_metric == 'val_loss':
                metric_to_monitor = valid_loss
            elif self.args.monitor_metric == 'val_acc':
                metric_to_monitor = valid_acc
            elif self.args.monitor_metric == 'val_iou':
                metric_to_monitor = valid_IoU
            elif self.args.monitor_metric == 'val_f1':
                metric_to_monitor = valid_F1score
            elif self.args.monitor_metric == 'train_loss':
                metric_to_monitor = train_metrics[0]
            else:
                metric_to_monitor = valid_loss
                
            is_best = self.check_improvement(metric_to_monitor, self.args.monitor_metric, epoch)
            
            self.save_checkpoint(epoch, is_best)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(
                f"[TRAIN] Loss: {train_metrics[0]:.5f} "
                f"Acc: {train_metrics[1]:.5f} "
                f"IoU: {train_metrics[4]:.5f} "
                f"IoU_Back: {train_metrics[5]:.5f} "
                f"F1: {train_metrics[2]:.5f} "
                f"F1_Back: {train_metrics[3]:.5f}")
            print(
                f"[VALID] Loss: {valid_metrics[0]:.5f} "
                f"Acc: {valid_metrics[1]:.5f} "
                f"IoU: {valid_metrics[4]:.5f} "
                f"IoU_Back: {valid_metrics[5]:.5f} "
                f"F1: {valid_metrics[2]:.5f} "
                f"F1_Back: {valid_metrics[3]:.5f}")
            print(
                f"Current LR: {current_lr:.2e} | "
                f"Best Val IoU: {self.best_pred:.5f} | "
                f"Time: {time_str}")
            
        total_training_time = time.time() - start_time
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        print(f"Training complete! Total training time: {total_time_str}")
        
        best_metric = (self.args.monitor_metric, self.best_pred)
        self.logger.export_history_to_json(total_time=total_time_str, 
                                        best_metric=best_metric, 
                                        save_dir=self.results_dir)

        self.logger.plot_learning_curves(save_dir=self.plots_dir)
        
        self.logger.close()
        
        return self.logger.get_history()

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic segmentation model training with Zhao method')
    
    parser.add_argument('--train-image-folder', type=str, default='',
                        help='Training images folder path')
    parser.add_argument('--train-label-folder', type=str, default='',
                        help='Training labels folder path')
    parser.add_argument('--valid-image-folder', type=str, default='',
                        help='Validation images folder path')
    parser.add_argument('--valid-label-folder', type=str, default='',
                        help='Validation labels folder path')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Base directory to save all outputs (logs, results, checkpoints)')
    parser.add_argument('--warmup-epochs', type=int,)
    parser.add_argument('--correct-epochs', type=int,)
    
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--start-epochs', type=int, default=0,
                        help='Starting epoch')
    parser.add_argument('--max-epochs', type=int, default=70,
                        help='Maximum number of epochs')
    parser.add_argument('--dataloadworkers', type=int, default=16,
                        help='Number of data loading worker threads')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='Weight decay factor for AdamW optimizer')
    parser.add_argument('--checkpoints-save', type=str, default='Weight',
                        help='Checkpoint save directory')
                        
    parser.add_argument('--monitor-metric', type=str, default='val_iou',
                        help='Performance metric to monitor')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience value for early stopping')
                        
    parser.add_argument('--mixed-precision', type=lambda x: str(x).lower() == 'true', 
                        default=False,
                        help='Enable mixed precision training')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    args.label_correction_method = 'zhao'
        
    base_save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(base_save_dir, exist_ok=True)
    
    args.checkpoints_save = os.path.join(base_save_dir, args.checkpoints_save)
    args.results_dir = os.path.join(base_save_dir, 'Results')
    args.logs_dir = os.path.join(base_save_dir, 'Logs')
    args.plots_dir = os.path.join(base_save_dir, 'Plots')
    
    os.makedirs(args.checkpoints_save, exist_ok=True)  
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print("Analyzing dataset properties...")
    try:
        input_size, bands, classNum = DatasetAnalyzer.get_dataset_properties(
            args.valid_image_folder, args.valid_label_folder
        )
        args.input_size = input_size
        args.bands = bands
        args.classNum = classNum
        print(f"Image size: {input_size}, Bands: {bands}, Classes: {classNum}")
    except Exception as e:
        print(f"Dataset property analysis failed: {e}")
        return
    
    print("Calculating dataset mean and standard deviation...")
    try:
        args.MeanList = [0.485, 0.456, 0.406]
        args.StdList = [0.229, 0.224, 0.225]
        print(f"Mean: {args.MeanList}, Standard deviation: {args.StdList}")
        
    except Exception as e:
        print(f"Mean and standard deviation calculation failed: {e}")
        return
    
    print("Initializing trainer...")
    trainer = Trainer(args)
    
    print("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()