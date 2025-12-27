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
import torch.nn as nn
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
from Utils.dataset import CustomDataset, DatasetAnalyzer


class Trainer:
    
    def __init__(self, args):
        self.args = args
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
        save_dir_name = os.path.basename(args.save_dir)
        self.experiment_name = f"{date}_{hour}h{minute}m_{save_dir_name}_{args.model}_lr{args.learning_rate}_ep{args.max_epochs}"
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

        self.train_loader = DataLoader(
            args.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloadworkers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.valid_loader = DataLoader(
            args.valid_dataset,
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
        
        self.criterion = ModelFactory.create_loss(
            args.model, 
            args, 
            self.device
        )
        
        self.optimizer = ModelFactory.create_optimizer(
            args.model, self.model, args
        )
        
        self.iters_per_epoch = len(args.train_dataset) // args.batch_size
        self.max_iters = args.max_epochs * self.iters_per_epoch
        self.lr_scheduler = ModelFactory.create_scheduler(
            args.model, self.optimizer, self.max_iters, args
        )
        
        self.metric = Evaluator(num_classes=args.classNum, device=self.device)
        
        if self.args.monitor_metric.endswith('loss'):
            self.best_pred = float('inf')
        else:
            self.best_pred = float('-inf')
            
        self.patience = args.patience
        self.early_stop_counter = 0
        self.early_stop = False
        
        self._load_checkpoint_if_available()
        
    def _load_checkpoint_if_available(self):
        if self.args.resumecheckpoint and os.path.isfile(self.args.resumecheckpoint):
            print(f"Loading checkpoint: {self.args.resumecheckpoint}")
            checkpoint = torch.load(self.args.resumecheckpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            
            if not self.args.fine_tuning:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.lr_scheduler is not None and 'scheduler' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
                
                if self.use_amp and 'amp_scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['amp_scaler'])
                    
            self.best_pred = checkpoint['best_pred']
            self.args.start_epochs = checkpoint['epoch'] + 1
            print(f"Successfully loaded checkpoint (epoch {checkpoint['epoch']})")
        
        elif self.args.resumecheckpoint:
            raise RuntimeError(f"Checkpoint not found: '{self.args.resumecheckpoint}'")
            
        if self.args.fine_tuning:
            self.args.start_epochs = 0
            print("Fine-tuning mode: Starting from epoch 0")

    def train_one_epoch(self, epoch):
        self.model.train()
        self.metric.reset()
        train_loss = 0.0
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs} [Train]")
        
        for iteration_train, (images, labels) in enumerate(tbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = labels.float()

            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{self.args.max_epochs} [Train] Loss: {train_loss / (iteration_train + 1):.5f}")
            
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).long()
                self.metric.add_batch(labels, preds)

        train_loss = train_loss / len(self.train_loader)
        train_acc = self.metric.Overall_Accuracy()
        train_F1score = self.metric.F1Score(class_index=1)
        train_F1score_Back = self.metric.F1Score(class_index=0)
        train_IoU = self.metric.Intersection_over_Union(class_index=1)
        train_IoU_Back = self.metric.Intersection_over_Union(class_index=0)
        train_Precision = self.metric.Precision()
        train_Recall = self.metric.Recall()
        train_Kappa = self.metric.Kappa()
        
        metrics = {
            'loss': train_loss,
            'acc': train_acc,
            'f1_score': train_F1score,
            'f1_score_Back': train_F1score_Back,
            'iou': train_IoU,
            'iou_Back': train_IoU_Back,
            'precision': train_Precision,
            'recall': train_Recall,
            'kappa': train_Kappa
        }
        self.logger.log_metrics(metrics, epoch, prefix='train')
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_learning_rate(current_lr, epoch)        
        
        return (train_loss, train_acc, train_F1score, train_F1score_Back, train_IoU,train_IoU_Back,
                train_Precision, train_Recall, train_Kappa)

    def train_one_epoch_reaunet(self, epoch):

        self.model.train()
        self.metric.reset()
        train_loss = 0.0
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs} [Train]")
        
        for iteration_train, (images, labels) in enumerate(tbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = labels.float()

            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                results = self.model(images)
                
                if isinstance(results, list):
                    loss = torch.zeros(1).to(self.device)
                    for k in range(len(results) - 1):
                        loss += (0.1 * k + 0.2) * self.criterion(results[k], labels)
                    loss += self.criterion(results[-1], labels)
                else:
                    loss = self.criterion(results, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{self.args.max_epochs} [Train] Loss: {train_loss / (iteration_train + 1):.5f}")
            
            with torch.no_grad():
                final_output = results[-1] if isinstance(results, list) else results
                preds = (final_output > 0.5).long()
                self.metric.add_batch(labels, preds)

        train_loss = train_loss / len(self.train_loader)
        train_acc = self.metric.Overall_Accuracy()
        train_F1score = self.metric.F1Score(class_index=1)
        train_F1score_Back = self.metric.F1Score(class_index=0)
        train_IoU = self.metric.Intersection_over_Union(class_index=1)
        train_IoU_Back = self.metric.Intersection_over_Union(class_index=0)
        train_Precision = self.metric.Precision()
        train_Recall = self.metric.Recall()
        train_Kappa = self.metric.Kappa()
        
        metrics = {
            'loss': train_loss,
            'acc': train_acc,
            'f1_score': train_F1score,
            'f1_score_Back': train_F1score_Back,
            'iou': train_IoU,
            'iou_Back': train_IoU_Back,
            'precision': train_Precision,
            'recall': train_Recall,
            'kappa': train_Kappa
        }
        self.logger.log_metrics(metrics, epoch, prefix='train')
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_learning_rate(current_lr, epoch)        
        
        return (train_loss, train_acc, train_F1score, train_F1score_Back, train_IoU,train_IoU_Back,
                train_Precision, train_Recall, train_Kappa)

    def train_one_epoch_unetformer(self, epoch):
        self.model.train()
        self.metric.reset()
        train_loss = 0.0
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs} [Train]")
        
        for iteration_train, (images, labels) in enumerate(tbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
    
                main_output, aux_output= outputs
                main_loss = self.criterion(main_output, labels)
                aux_loss = self.criterion(aux_output, labels)

                loss = main_loss + 0.4*aux_loss

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                            
            train_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{self.args.max_epochs} [Train] Loss: {train_loss / (iteration_train + 1):.5f}")
            
            with torch.no_grad():
                evaluation_output = outputs[0]
                preds = (torch.sigmoid(evaluation_output) > 0.5).long()
                self.metric.add_batch(labels, preds)

        train_loss = train_loss / len(self.train_loader)
        train_acc = self.metric.Overall_Accuracy()
        train_F1score = self.metric.F1Score(class_index=1)
        train_F1score_Back = self.metric.F1Score(class_index=0)
        train_IoU = self.metric.Intersection_over_Union(class_index=1)
        train_IoU_Back = self.metric.Intersection_over_Union(class_index=0)
        train_Precision = self.metric.Precision()
        train_Recall = self.metric.Recall()
        train_Kappa = self.metric.Kappa()
        
        metrics = {
            'loss': train_loss,
            'acc': train_acc,
            'f1_score': train_F1score,
            'f1_score_Back': train_F1score_Back,
            'iou': train_IoU,
            'iou_Back': train_IoU_Back,
            'precision': train_Precision,
            'recall': train_Recall,
            'kappa': train_Kappa
        }
        self.logger.log_metrics(metrics, epoch, prefix='train')
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_learning_rate(current_lr, epoch)        
        
        return (train_loss, train_acc, train_F1score, train_F1score_Back,train_IoU, train_IoU_Back,
                train_Precision, train_Recall, train_Kappa)
    
    def valid_one_epoch(self, epoch):
        self.model.eval()
        self.metric.reset()
        valid_loss = 0.0
        tbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs} [Valid]")

        for iteration_valid, (images, labels) in enumerate(tbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = labels.float()

            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).long()

            valid_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{self.args.max_epochs} [Valid] Loss: {valid_loss / (iteration_valid + 1):.5f}")

            self.metric.add_batch(labels, preds)

        valid_loss = valid_loss / len(self.valid_loader)
        valid_acc = self.metric.Overall_Accuracy()
        valid_F1score = self.metric.F1Score(class_index=1)
        valid_F1score_Back = self.metric.F1Score(class_index=0)
        valid_IoU = self.metric.Intersection_over_Union(class_index=1)
        valid_IoU_Back = self.metric.Intersection_over_Union(class_index=0)
        valid_Precision = self.metric.Precision()
        valid_Recall = self.metric.Recall()
        valid_Kappa = self.metric.Kappa()

        metrics = {
            'loss': valid_loss,
            'acc': valid_acc,
            'f1_score': valid_F1score,
            'f1_score_Back': valid_F1score_Back,
            'iou': valid_IoU, 
            'iou_Back': valid_IoU_Back,
            'precision': valid_Precision,
            'recall': valid_Recall,
            'kappa': valid_Kappa
        }
        self.logger.log_metrics(metrics, epoch, prefix='val')
                   
        return (valid_loss, valid_acc, valid_F1score, valid_F1score_Back, valid_IoU, valid_IoU_Back,
                valid_Precision, valid_Recall, valid_Kappa)

    def valid_one_epoch_reaunet(self, epoch):
        self.model.eval()
        self.metric.reset()
        valid_loss = 0.0
        tbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs} [Valid]")

        for iteration_valid, (images, labels) in enumerate(tbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = labels.float()

            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    results = self.model(images)
                    
                    if isinstance(results, list):
                        loss = torch.zeros(1).to(self.device)
                        for k in range(len(results) - 1):
                            loss += (0.1 * k + 0.2) * self.criterion(results[k], labels)
                        loss += self.criterion(results[-1], labels)
                        final_output = results[-1]
                    else:
                        loss = self.criterion(results, labels)
                        final_output = results

                    preds = (final_output > 0.5).long()

            valid_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{self.args.max_epochs} [Valid] Loss: {valid_loss / (iteration_valid + 1):.5f}")

            self.metric.add_batch(labels, preds)

        valid_loss = valid_loss / len(self.valid_loader)
        valid_acc = self.metric.Overall_Accuracy()
        valid_F1score = self.metric.F1Score(class_index=1)
        valid_F1score_Back = self.metric.F1Score(class_index=0)
        valid_IoU = self.metric.Intersection_over_Union(class_index=1)
        valid_IoU_Back = self.metric.Intersection_over_Union(class_index=0)
        valid_Precision = self.metric.Precision()
        valid_Recall = self.metric.Recall()
        valid_Kappa = self.metric.Kappa()

        metrics = {
            'loss': valid_loss,
            'acc': valid_acc,
            'f1_score': valid_F1score,
            'f1_score_Back': valid_F1score_Back,
            'iou': valid_IoU, 
            'iou_Back': valid_IoU_Back,
            'precision': valid_Precision,
            'recall': valid_Recall,
            'kappa': valid_Kappa
        }
        self.logger.log_metrics(metrics, epoch, prefix='val')
                   
        return (valid_loss, valid_acc, valid_F1score, valid_F1score_Back, valid_IoU, valid_IoU_Back,
                valid_Precision, valid_Recall, valid_Kappa)


    def valid_one_epoch_unetformer(self, epoch):
        self.model.eval()
        self.metric.reset()
        valid_loss = 0.0
        tbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs} [Valid]")

        for iteration_valid, (images, labels) in enumerate(tbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                
            valid_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{self.args.max_epochs} [Valid] Loss: {valid_loss / (iteration_valid + 1):.5f}")

            self.metric.add_batch(labels, preds)

        valid_loss = valid_loss / len(self.valid_loader)
        valid_acc = self.metric.Overall_Accuracy()
        valid_F1score = self.metric.F1Score(class_index=1)
        valid_F1score_Back = self.metric.F1Score(class_index=0)
        valid_IoU = self.metric.Intersection_over_Union(class_index=1)
        valid_IoU_Back = self.metric.Intersection_over_Union(class_index=0)
        valid_Precision = self.metric.Precision()
        valid_Recall = self.metric.Recall()
        valid_Kappa = self.metric.Kappa()

        metrics = {
            'loss': valid_loss,
            'acc': valid_acc,
            'f1_score': valid_F1score,
            'f1_score_Back': valid_F1score_Back,
            'iou': valid_IoU,
            'iou_Back': valid_IoU_Back, 
            'precision': valid_Precision,
            'recall': valid_Recall,
            'kappa': valid_Kappa
        }
        self.logger.log_metrics(metrics, epoch, prefix='val')
                
        return (valid_loss, valid_acc, valid_F1score, valid_F1score_Back, valid_IoU, valid_IoU_Back,
                valid_Precision, valid_Recall, valid_Kappa)
                
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
            print(f"Model performance improved!")
        else:
            self.early_stop_counter += 1
            print(f"Model performance did not improve. Early stopping counter: {self.early_stop_counter}/{self.patience}")
            if self.early_stop_counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping condition met, stopping training")
                
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
            
        if is_best:
            best_filename = os.path.join(self.checkpoints_dir, 'best_model_checkpoint.pth')
            shutil.copyfile(filename, best_filename)
            print(f"Saved best model checkpoint: {best_filename}")
            
    def train(self):
        print(f"Starting training: from epoch {self.args.start_epochs} to {self.args.max_epochs}")
        print(f"Using device: {self.device}")
        

        if 'unetformer' in self.args.model.lower():
            print("Using specialized training functions for UNetFormer model")
            train_epoch_fn = self.train_one_epoch_unetformer
            valid_epoch_fn = self.valid_one_epoch_unetformer
        elif 'reaunet' in self.args.model.lower():
            print("Using specialized training functions for REAUNet model")
            train_epoch_fn = self.train_one_epoch_reaunet
            valid_epoch_fn = self.valid_one_epoch_reaunet
        else:
            train_epoch_fn = self.train_one_epoch
            valid_epoch_fn = self.valid_one_epoch
        
        start_time = time.time()
        
        for epoch in range(self.args.start_epochs, self.args.max_epochs):
            if self.early_stop:
                print(f"Early stopping training: epoch {epoch}")
                break
                
            epoch_start_time = time.time()
            
            train_metrics = train_epoch_fn(epoch)
            valid_metrics = valid_epoch_fn(epoch)
            
            valid_loss, valid_acc, valid_F1score, valid_F1score_Back, valid_IoU, valid_IoU_Back, valid_Precision, valid_Recall, valid_Kappa = valid_metrics
                
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
                

            self.lr_scheduler.step()
    
            epoch_time = time.time() - epoch_start_time
            minutes, seconds = divmod(epoch_time, 60)
            time_str = f"{int(minutes)}m {int(seconds)}s"
            
            self.logger.log_epoch_time(time_str, epoch)
            
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
                f"Best {self.args.monitor_metric.replace('_', ' ').title()}: {self.best_pred:.5f} | "
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
    parser = argparse.ArgumentParser(description='Semantic segmentation model training')
    
    parser.add_argument('--train-image-folder', type=str, default ='',
                        help='Training images folder path')
    parser.add_argument('--train-label-folder', type=str, default ='',
                        help='Training labels folder path')
    parser.add_argument('--valid-image-folder', type=str, default ='',
                        help='Validation images folder path')
    parser.add_argument('--valid-label-folder', type=str, default ='',
                        help='Validation labels folder path')
    parser.add_argument('--save-dir', type=str, default = '',
                        help='Base directory to save all outputs (logs, results, checkpoints)')
    
    parser.add_argument('--model', type=str, default='ablation',
                        help='Model type')
                        
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--start-epochs', type=int, default=0,
                        help='Starting epoch')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--dataloadworkers', type=int, default=16,
                        help='Number of data loading worker threads')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                    help='Minimum learning rate for Cosine Annealing')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='Weight decay factor for AdamW optimizer')
                        
    parser.add_argument('--checkpoints-save', type=str, default='Weight',
                        help='Checkpoint save directory')
    parser.add_argument('--resumecheckpoint', type=str, default= '',
                        help='Path to checkpoint to resume training')
    parser.add_argument('--fine-tuning', action='store_true',
                        help='Whether to fine-tune')
                        
    parser.add_argument('--monitor-metric', type=str, default='val_iou',
                        help='Performance metric to monitor')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience value for early stopping')
                        
    parser.add_argument('--mixed-precision', type=lambda x: str(x).lower() == 'true', 
                        default=False,
                        help='Enable mixed precision training (default: True)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    print("Creating datasets...")

    args.train_dataset = CustomDataset(
        image_dir=args.train_image_folder, 
        label_dir=args.train_label_folder, 
        mode='train',
        crop_size=args.input_size[0],
        mean_list=args.MeanList,
        std_list=args.StdList,
        model=args.model
    )
    args.valid_dataset = CustomDataset(
        image_dir=args.valid_image_folder, 
        label_dir=args.valid_label_folder, 
        mode='val',
        crop_size=args.input_size[0],
        mean_list=args.MeanList,
        std_list=args.StdList,
        model=args.model
    )
    
    print(f"Training set size: {len(args.train_dataset)}, Validation set size: {len(args.valid_dataset)}")

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