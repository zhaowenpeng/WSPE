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
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import rasterio
import numpy as np
from scipy.optimize import curve_fit
import os


def adaptive_label_update(iou_history, current_epoch=None, threshold=0.9, eval_interval=100, num_iter_per_epoch=10581/10):
    
    def curve_func(x, a, b, c):
        return a * (1 - np.exp(-1 / c * x ** b))
    
    def fit(func, x, y):
        popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), method='trf', 
                              sigma=np.geomspace(1, .1, len(y)),
                              absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]))
        return tuple(popt)
    
    def derivation(x, a, b, c):
        x = x + 1e-6
        return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))
    
    xdata_fit = np.linspace(0, len(iou_history) * eval_interval / num_iter_per_epoch, len(iou_history))
    
    a, b, c = fit(curve_func, xdata_fit, iou_history)
    
    epoch_range = np.arange(1, 16)
    y_hat = curve_func(epoch_range, a, b, c)
    
    relative_change = abs(abs(derivation(epoch_range, a, b, c)) - abs(derivation(1, a, b, c))) / abs(derivation(1, a, b, c))
    relative_change[relative_change > 1] = 0
    
    update_epoch = np.sum(relative_change <= threshold) + 1
    
    curve_formula = f"f(x) = {a:.4f} * (1 - exp(-1/{c:.4f} * x^{b:.4f}))"
    
    tangent_formula = "Current epoch not provided, cannot calculate tangent"
    current_derivative = None
    
    if current_epoch is not None:
        current_value = curve_func(current_epoch, a, b, c)
        current_derivative = derivation(current_epoch, a, b, c)
        tangent_formula = f"y = {current_value:.4f} + {current_derivative:.4f}(x - {current_epoch})"
        
        print(f"IoU Curve: {curve_formula}")
        print(f"Relative Change: {relative_change}")
        print(f"Derivative at epoch={current_epoch}: {current_derivative:.6f}")
        print(f"Tangent Line at epoch={current_epoch}: {tangent_formula}")

    if current_epoch is not None:
         return int(update_epoch)
    else:
        return int(update_epoch)

def curve_func(x, a, b, c):
    return a * (1 - np.exp(- b * x ** c))

def curve_derivation(x, a, b, c):
    return a * c * b * np.exp(-b * x ** c) * (x ** (c - 1))

def fit_curve(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 0.5, 0.5), 
                           method='trf', bounds=([0,0,0],[1,np.inf,1]))
    return tuple(popt)

def linear_func(x, a, b):
    return a*x+b

def fit_linear(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 0), 
                           method='trf', bounds=([0, -np.inf], [np.inf, np.inf]))
    return tuple(popt)

def cal_ngs_to_dict(data, wsizes):
    n_ep = len(data)
    ngs_dict = {b:[] for b in wsizes}
    for bi, ws in enumerate(wsizes):
        if n_ep>=ws:
            x0 = np.arange(n_ep-ws+1,n_ep+1)
            y0 = np.array(data[n_ep-ws:n_ep])
            a, b = fit_linear(linear_func, x0, y0)
            ngs_dict[ws].append(a)
    return ngs_dict

def adaptive_weight_schedule_improved(epoch, warmup_epoch, correct_epoch):
    if epoch == warmup_epoch:
        orig_weight = 0.6  
    else:
        progress = (epoch - warmup_epoch) / (correct_epoch - warmup_epoch)
        
        orig_weight = 0.6 - progress * 0.4
        orig_weight = max(orig_weight, 0.2)
    
    return orig_weight

def adaptive_confidence_improved(epoch, warmup_epoch, correct_epoch):
    if epoch == warmup_epoch:
        confidence = 0.9
    else:
        progress = (epoch - warmup_epoch) / (correct_epoch - warmup_epoch)
        
        confidence = 0.9 - progress * 0.2
        confidence = max(confidence, 0.7)

    return confidence

def get_lr_scheduler_zhao(optimizer, warmup_epochs, correct_epochs, max_epochs):
    import math
        
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 1.0
        elif epoch < correct_epochs:
            return 0.1
        else:
            progress = (epoch - correct_epochs) / (max_epochs - correct_epochs)
            cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
            return 0.1 * (0.01 + 0.99 * cosine_factor)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def progressive_label_update_complete(model, update_dataloader, device, epoch, 
                                    updatelabel_dir, save_dir, confidence_threshold=0.7, 
                                    args=None, warmup_epoch=5, correct_epoch=10):
    os.makedirs(updatelabel_dir, exist_ok=True)
    
    if save_dir is not None:
        prob_dir = os.path.join(save_dir, f"epoch_{epoch+1}", "probabilities")
        diff_dir = os.path.join(save_dir, f"epoch_{epoch+1}", "differences")
        os.makedirs(prob_dir, exist_ok=True)
        os.makedirs(diff_dir, exist_ok=True)
    
    progress = (epoch - warmup_epoch) / (correct_epoch - warmup_epoch)
    print(f"warmup_epoch {warmup_epoch}, correct_epoch: {correct_epoch}, epoch: {epoch+1}, progress: {progress}")
    correction_ratio = 0.6 + progress * 0.4

    print(f"Epoch {epoch+1}: Progressive correction ratio: {correction_ratio:.2f}")
    
    model.eval()
    updated_labels = {}
    
    tbar = tqdm(update_dataloader, desc=f"Epoch {epoch+1}: Generating Progressive Updated Labels")
    
    with torch.no_grad():
        for iteration, batch in enumerate(tbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float()
            filenames = batch['filename']
            
            outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            
            for i, filename in enumerate(filenames):
                base_name = os.path.splitext(os.path.basename(filename))[0]
                
                sample_pred_probs = pred_probs[i]
                sample_labels = labels[i]
                
                if sample_pred_probs.dim() > 2:
                    sample_pred_probs = sample_pred_probs.squeeze()
                if sample_labels.dim() > 2:
                    sample_labels = sample_labels.squeeze()
                
                corrected_labels, correction_mask = progressive_label_correction_single(
                    pred_probs=sample_pred_probs,
                    original_labels=sample_labels,
                    correction_ratio=correction_ratio,
                    confidence_threshold=confidence_threshold
                )
                
                corrected_labels_np = corrected_labels.detach().cpu().numpy().astype(np.uint8)
                
                updated_labels[filename] = corrected_labels_np
                
                save_path = os.path.join(updatelabel_dir, f"{base_name}.tif")
                with rasterio.open(
                    save_path,
                    'w',
                    driver='GTiff',
                    height=corrected_labels_np.shape[0],
                    width=corrected_labels_np.shape[1],
                    count=1,
                    dtype='uint8',
                    compress='lzw'
                ) as dst:
                    dst.write(corrected_labels_np, 1)
    
    print(f"Epoch {epoch+1}: Progressive label update completed, {len(updated_labels)} samples processed")
    return updated_labels

def progressive_label_correction_single(pred_probs, original_labels, correction_ratio=0.3, confidence_threshold=0.7):
    correction_ratio = max(0.05, min(correction_ratio, 1.0))
    
    pred_confidence = torch.max(pred_probs, 1 - pred_probs)
    
    if pred_confidence.numel() == 0:
        print("Warning: Empty confidence tensor, returning original labels")
        return original_labels.clone(), torch.zeros_like(original_labels, dtype=torch.bool)
    
    flat_confidence = pred_confidence.flatten()
    
    try:
        quantile_param = max(0.0, min(1.0, 1 - correction_ratio))
        threshold_value = torch.quantile(flat_confidence, quantile_param)
        
        correction_mask = pred_confidence > threshold_value
    except RuntimeError as e:
        print(f"Warning: quantile calculation failed: {e}")
        threshold_value = torch.mean(pred_confidence)
        correction_mask = pred_confidence > threshold_value
    
    corrected_labels = original_labels.clone()
    
    original_dtype = original_labels.dtype
    
    new_labels = (pred_probs[correction_mask] > confidence_threshold)
    
    if original_dtype == torch.float32 or original_dtype == torch.float64:
        corrected_labels[correction_mask] = new_labels.float()
    else:
        corrected_labels[correction_mask] = new_labels.to(original_dtype)
    
    return corrected_labels, correction_mask

def train_correction_zhao(model, warmup_dataloader,update_dataloader, optimizer, device, 
                             epoch, metric=None, warmup_epoch=5, correct_epoch=10, 
                             max_epochs=20, use_amp=False, save_dir=None, eval_interval=20,updatelabel_dir=None,args=None): 
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    torch.cuda.empty_cache()
    if metric is not None:
        metric.reset()
        
    train_loss = 0.0
    fg_iou = 0.0
    bg_iou = 0.0
    curr_epoch_iou_history = []
    
    if epoch ==0:
        import torch.nn as nn
        weight = nn.Parameter(torch.ones(3, device=device))
        optimizer.add_param_group({'params': weight})
        print(f"Initial weights: {weight.data}")
    else:
        weight = optimizer.param_groups[-1]['params'][0]
        print(f"Using existing weights: {weight.data}")

    if epoch < warmup_epoch:
        model.train()
    
        from Utils.loss import Mult_SmoothLoss      
        criterion = Mult_SmoothLoss(
                    smoothing=0.1,
                    threshold=0.9,
                    ce_weight=1.0,
                    consistency_weight=1.0,
                    ).to(device)
        
        tbar = tqdm(warmup_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Warmup Train ZHAO]")
        
        for iteration, batch in enumerate(tbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float()
            filenames = batch['filename']
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)
                
                inputs_small = F.interpolate(images, scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                outputs_small = model(inputs_small)
                
                inputs_large = F.interpolate(images, scale_factor=1.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                outputs_large = model(inputs_large)
                
                h, w = outputs.shape[2], outputs.shape[3]
                pred1 = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
                pred2 = F.interpolate(outputs_small, size=(h, w), mode='bilinear', align_corners=True)
                pred3 = F.interpolate(outputs_large, size=(h, w), mode='bilinear', align_corners=True)
                
                total_loss, ce_loss, consistency_loss, mixture_label = criterion(
                    labels.unsqueeze(1) if outputs.dim() == 4 and labels.dim() == 3 else labels, 
                    pred1, pred2, pred3, weight
                )
                loss = total_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            if (iteration + 1) % eval_interval == 0 and metric is not None:
                with torch.no_grad():
                    current_iou = metric.Intersection_over_Union(class_index=1)
                    curr_epoch_iou_history.append(current_iou)
            
            if metric is not None:
                with torch.no_grad():
                    preds = (pred_probs > 0.5).long()
                    if preds.dim() == 4:
                        preds = preds.squeeze(1)
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)
                    
                    metric.add_batch(labels.long(), preds)
            fg_iou = metric.Intersection_over_Union(class_index=1)

            tbar.set_description(
                f"Epoch {epoch+1}/{max_epochs} [Warmup Train ZHAO] "
                f"Loss: {train_loss/(iteration+1):.5f} | "
                f"FG_IoU: {fg_iou:.4f}"
            )
    
    elif epoch < correct_epoch:
        torch.cuda.empty_cache()
        
        orig_weight = adaptive_weight_schedule_improved(epoch, warmup_epoch, correct_epoch)
        
        confidence = adaptive_confidence_improved(epoch, warmup_epoch, correct_epoch)

        print(f"Epoch {epoch+1}: Original label weight: {orig_weight:.3f}")
        print(f"Epoch {epoch+1}: Confidence threshold: {confidence:.3f}")
            
        updated_labels = progressive_label_update_complete(
            model=model,
            update_dataloader=update_dataloader,
            device=device,
            epoch=epoch,
            updatelabel_dir=updatelabel_dir,
            save_dir=save_dir,
            confidence_threshold=confidence,
            args=args,
            warmup_epoch=warmup_epoch,
            correct_epoch=correct_epoch
        )

        from Utils.dataset import CAOISPRSDataset
        correct_dataset = CAOISPRSDataset(
            image_dir=args.train_image_folder,
            label_dir=args.train_label_folder,
            update_label_dir=os.path.join(args.save_dir,'update_label'),
            mode='train',
            mean_list=args.MeanList,
            std_list=args.StdList,
        )
        
        correct_dataloader = DataLoader(
            correct_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloadworkers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        model.train()
        from Utils.loss import Mult_SmoothLoss, EdgeTanimotoLoss
        criterion_ori = Mult_SmoothLoss(
            smoothing=0.1,
            threshold=0.9,
            consistency_weight=1.0,
            ce_weight=1.0,
            ignore_index=255).to(device)
        
        criterion_update = torch.nn.BCEWithLogitsLoss().to(device)
        
        tbar = tqdm(correct_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Correction Train ZHAO]")
        for iteration, batch in enumerate(tbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float()
            update_labels = batch['update_label'].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)

                inputs_small = F.interpolate(images, scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                outputs_small = model(inputs_small)
                
                inputs_large = F.interpolate(images, scale_factor=1.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                outputs_large = model(inputs_large)
                
                h, w = outputs.shape[2], outputs.shape[3]
                pred1 = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
                pred2 = F.interpolate(outputs_small, size=(h, w), mode='bilinear', align_corners=True)
                pred3 = F.interpolate(outputs_large, size=(h, w), mode='bilinear', align_corners=True)
                
                total_loss, ce_loss, consistency_loss, mixture_label = criterion_ori(
                    labels.unsqueeze(1) if outputs.dim() == 4 and labels.dim() == 3 else labels, 
                    pred1, pred2, pred3, weight
                )
                loss_orig = total_loss
                loss_updated = criterion_update(outputs, update_labels.unsqueeze(1) if outputs.dim() == 4 and update_labels.dim() == 3 else update_labels)

                loss = orig_weight * loss_orig + (1.0 - orig_weight) * loss_updated 
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if (iteration + 1) % eval_interval == 0 and metric is not None:
                with torch.no_grad():
                    current_iou = metric.Intersection_over_Union(class_index=1)
                    curr_epoch_iou_history.append(current_iou)
            if metric is not None:
                with torch.no_grad():
                    preds = (pred_probs > 0.5).long()
                    if preds.dim() == 4:
                        preds = preds.squeeze(1)
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)
                    
                    metric.add_batch(labels.long(), preds)
                    
            fg_iou = metric.Intersection_over_Union(class_index=1)
            tbar.set_description(
                f"Epoch {epoch+1}/{max_epochs} [Correction Train ZHAO] "
                f"Loss: {train_loss/(iteration+1):.5f} | "
                f"FG_IoU: {fg_iou:.4f}"
            )

    else:
        reset_to_warmup = False
        if reset_to_warmup and epoch == correct_epoch:
            warmup_checkpoint_path = os.path.join(args.checkpoints_save, 'warmup_epoch.pth')
            if os.path.exists(warmup_checkpoint_path):
                print(f"Reload warmup phase pretrained weights (epoch {warmup_epoch}) to start scratch phase training...")
                checkpoint = torch.load(warmup_checkpoint_path)
                model.load_state_dict(checkpoint['state_dict'])
                
                optimizer.param_groups[0]['lr'] = args.learning_rate
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param.grad = None
                        
                print(f"Successfully loaded warmup phase pretrained weights, will continue training based on this")
            else:
                print(f"Warning: Could not find warmup pretrained weight file {warmup_checkpoint_path}, will continue using current model weights")
        else:
            print("Use current model state to continue training final phase")

        torch.cuda.empty_cache()
        from Utils.dataset import CAOISPRSDataset
        train_loss = 0.0
        fg_iou = 0.0
        final_dataset = CAOISPRSDataset(
            image_dir=args.train_image_folder,
            label_dir=args.train_label_folder,  
            update_label_dir=os.path.join(args.save_dir,'update_label'),  
            mode='train',
            mean_list=args.MeanList,
            std_list=args.StdList,
        )
        
        scratch_dataloader = DataLoader(
            final_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloadworkers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        model.train()

        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        tbar = tqdm(scratch_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Scratch Train ZHAO]")
        for iteration, batch in enumerate(tbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float()  
            update_labels = batch['update_label'].to(device, non_blocking=True).float()  
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)
                
                loss = criterion(outputs, labels.unsqueeze(1) if outputs.dim() == 4 and labels.dim() == 3 else labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if metric is not None:
                with torch.no_grad():
                    preds = (pred_probs > 0.5).long()
                    if preds.dim() == 4:
                        preds = preds.squeeze(1)
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)
                    
                    labels_for_metric = labels.clone()
                    if labels_for_metric.dim() == 4:
                        labels_for_metric = labels_for_metric.squeeze(1)
                    
                    metric.add_batch(labels_for_metric.long(), preds)
                    
            fg_iou = metric.Intersection_over_Union(class_index=1)
            bg_iou = metric.Intersection_over_Union(class_index=0)
            tbar.set_description(
                f"Epoch {epoch+1}/{max_epochs} [Scratch Train ZHAO] "
                f"Loss: {train_loss/(iteration+1):.5f} | "
                f"FG_IoU: {fg_iou:.4f}"
            )
    
    if metric is not None:
        metrics = {
            'acc': metric.Overall_Accuracy(),
            'f1_score': metric.F1Score(class_index=1),
            'f1_score_Back': metric.F1Score(class_index=0),
            'iou': metric.Intersection_over_Union(class_index=1),
            'iou_Back': metric.Intersection_over_Union(class_index=0),
            'precision': metric.Precision(),
            'recall': metric.Recall(),
            'kappa': metric.Kappa(),
            'iou_history': curr_epoch_iou_history
        }
    else:
        metrics = {}
    
    dataloader_length = len(warmup_dataloader)
    train_loss = train_loss / dataloader_length
    
    return {
        'loss': train_loss, 
        'metrics': metrics
    }

def val_correction_zhao(model, dataloader, device, epoch=0, metric=None, 
                           use_amp=False, save_dir=None, max_epochs=100):
    torch.cuda.empty_cache()
    model.eval()
    if metric is not None:
        metric.reset()
    
    valid_loss = 0.0
    
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    
    if save_dir is not None:
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_save_dir, exist_ok=True)
    
    tbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Valid ZHAO]")
    
    for iteration, batch in enumerate(tbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).float()
        filenames = batch['filename']
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.dim() == 4 and outputs.shape[1] > 1:
                    outputs = outputs[:, 1:2, :, :]
                
                loss = criterion(outputs, labels.unsqueeze(1) if outputs.dim() == 4 and labels.dim() == 3 else labels)
                valid_loss += loss.item()
                
                pred_probs = torch.sigmoid(outputs)
                pred_masks = (pred_probs > 0.5).float()
                
                if metric is not None:
                    preds = (pred_probs > 0.5).long()
                    if preds.dim() == 4:
                        preds = preds.squeeze(1)
                    metric.add_batch(labels.long(), preds)
        
        tbar.set_description(f"Epoch {epoch+1}/{max_epochs} [Valid ZHAO] Loss: {valid_loss/(iteration+1):.5f}")
    
    valid_loss = valid_loss / len(dataloader)
    
    if metric is not None:
        metrics = {
            'acc': metric.Overall_Accuracy(),
            'f1_score': metric.F1Score(class_index=1),
            'f1_score_Back': metric.F1Score(class_index=0),
            'iou': metric.Intersection_over_Union(class_index=1),
            'iou_Back': metric.Intersection_over_Union(class_index=0),
            'precision': metric.Precision(),
            'recall': metric.Recall(),
            'kappa': metric.Kappa()
        }
    else:
        metrics = {}
    
    return {
        'loss': valid_loss,
        'metrics': metrics
    }