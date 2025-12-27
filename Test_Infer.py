import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import rasterio
from torch.utils.data import DataLoader
from torch.amp import autocast
import gc
from Utils.modelfactory import ModelFactory
from Utils.metrics import Evaluator
from Utils.dataset import CustomDataset
import itertools
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Polygon
import math

class DouglasPeuckerProcessor:
    def __init__(self, tolerance=1.5, spike_angle=30.0, min_area=10, target_value=255, args=None):
        self.tolerance = tolerance
        self.spike_angle = spike_angle
        self.min_area = min_area
        self.target_value = target_value
        self.args = args or {}

    def process_raster(self, input_raster_path, output_raster_path, output_vector_folder=None):
        if output_vector_folder is None:
            output_dir = os.path.dirname(output_raster_path)
            output_vector_folder = os.path.join(output_dir, "vectors")
        
        os.makedirs(output_vector_folder, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(input_raster_path))[0]
        vector_path = os.path.join(output_vector_folder, f"{base_name}.shp")
        simplified_vector_path = os.path.join(output_vector_folder, f"{base_name}_simplified.shp")
        
        self.raster_to_vector(input_raster_path, vector_path, self.target_value, self.min_area)
        
        self.simplify_vector(vector_path, simplified_vector_path, self.tolerance, self.spike_angle)
        
        self.vector_to_raster(simplified_vector_path, input_raster_path, output_raster_path, crop_value=self.args.get('foreground_value', 255),
                              non_crop_value=self.args.get('background_value', 0), nodata_value=self.args.get('nodata', None))
        
        return output_raster_path, simplified_vector_path
    
    @staticmethod
    def eliminate_spike(polygon, max_angle_deg=30.0):
        max_angle_rad = math.radians(max_angle_deg)
        
        coords = list(polygon.exterior.coords)
        
        coords.append(coords[0])
        coords.append(coords[1])
        new_coords = []
        
        for i in range(1, len(coords) - 1):
            p1 = coords[i - 1]
            p2 = coords[i]
            p3 = coords[i + 1]
            
            angle = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
            
            if angle < 0:
                angle += 2 * math.pi
                
            if max_angle_rad <= angle <= 2 * math.pi - max_angle_rad:
                new_coords.append(p2)
                
        if len(new_coords) >= 4:
            try:
                new_polygon = Polygon(new_coords)
                area1 = new_polygon.area
                area2 = polygon.area
                if 1.2 >= (area1 / area2) >= 0.8:
                    return new_polygon
            except:
                pass
        
        return polygon
    
    def simplify_vector(self, input_shp_path, output_shp_path, tolerance=None, spike_angle=None):
        if tolerance is None:
            tolerance = self.tolerance
        if spike_angle is None:
            spike_angle = self.spike_angle
            
        try:
            gdf = gpd.read_file(input_shp_path)
        except Exception as e:
            print(f"Failed to read shapefile: {e}")
            return None
        
        if gdf.empty:
            gdf.to_file(output_shp_path, driver="ESRI Shapefile")
            return gdf
        
        for index, row in gdf.iterrows():
            if row['geometry'] is not None and not row['geometry'].is_empty:
                simplified_polygon = row['geometry'].simplify(tolerance=tolerance)
                simplified_polygon = self.eliminate_spike(simplified_polygon, spike_angle)
                gdf.at[index, 'geometry'] = simplified_polygon
        
        gdf.to_file(output_shp_path, driver="ESRI Shapefile")
        return gdf
    
    @staticmethod
    def raster_to_vector(raster_path, output_shp_path, target_value=255, min_area=30):
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            binary_data = (raster_data == target_value).astype(np.uint8)
            
            from scipy.ndimage import label, sum as ndsum
            labeled_array, num_features = label(binary_data)
            
            area_sizes = ndsum(np.ones_like(binary_data), labeled_array, range(1, num_features + 1))
            
            valid_regions = np.where(area_sizes >= min_area)[0] + 1
            valid_mask = np.isin(labeled_array, valid_regions)
            
            filtered_binary = np.zeros_like(binary_data)
            filtered_binary[valid_mask] = 1
            
            shapes = features.shapes(filtered_binary, transform=transform)
            
            polygons = []
            for geom, value in shapes:
                if value == 1:
                    polygon = shape(geom)
                    polygons.append(polygon)
            
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
            
            gdf.to_file(output_shp_path, driver="ESRI Shapefile")
            
            print(f"Vector file saved, containing {len(polygons)} polygons (filter threshold: {min_area} pixels)")
            return gdf
    
    @staticmethod
    def vector_to_raster(vector_path, reference_raster_path, output_raster_path, 
                         crop_value=255, non_crop_value=0, nodata_value=None):
        try:
            gdf = gpd.read_file(vector_path)
        except Exception as e:
            print(f"Failed to read vector file: {e}")
            return None
        
        with rasterio.open(reference_raster_path) as src:
            meta = src.meta.copy()
            height = meta['height']
            width = meta['width']
            transform = meta['transform']
        
        meta.update({
            'driver': 'GTiff',
            'dtype': rasterio.uint8,
            'compress': 'lzw'
        })
        if nodata_value is not None:
            meta['nodata'] = nodata_value
        
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            non_crop_raster = np.full((height, width), non_crop_value, dtype=np.uint8)
            
            if not gdf.empty:
                shapes = [(geom, crop_value) for geom in gdf.geometry]
                mask = features.rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=non_crop_value,
                    dtype=np.uint8
                )
                
                dst.write(mask, 1)
            else:
                dst.write(non_crop_raster, 1)
        
        return output_raster_path

class BaseModel:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.isfile(self.args['checkpoint_path']):
            checkpoint = torch.load(self.args['checkpoint_path'], weights_only=False)
            hyperparameters = checkpoint['hyperparameters']
            self.args.update(hyperparameters)
        else:
            raise RuntimeError(f"No checkpoint found at '{self.args['checkpoint_path']}'")

        self.model = ModelFactory.create_model(
            self.args.get('model', 'deeplabv3plus'),
            classNum=self.args['classNum'],
            bands=self.args['bands'],
            input_size=self.args.get('input_size', None),
            device=self.device
        )

        self.model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint '{self.args['checkpoint_path']}'")

class Tester(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.metric = Evaluator(num_classes= self.args['classNum'], device=self.device)
        self.test_result_dir = self.args['test_result']
        os.makedirs(self.test_result_dir, exist_ok=True)
                
        self.model.to(self.device)
        self.model.eval()
        
        if self.args.get('enable_tta'):
            import ttach as tta
            transforms = tta.Compose([
                tta.Scale(scales=[0.75, 1.25],interpolation='bilinear',align_corners=True),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ])
            self.model = tta.SegmentationTTAWrapper(
                self.model, 
                transforms, 
                merge_mode='mean'
            )
            print(f"TTA enabled with transforms")
        
        print(f"Use Device Type: {self.device}")

    def test(self):
        self.model.eval()
        self.metric.reset()
        
        test_dataset = CustomDataset(
            self.args['test_image_folder'], 
            self.args['test_label_folder'],
            mode='test',
            mean_list=self.args['MeanList'],
            std_list=self.args['StdList'],
            model=self.args['model'],
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args['batch_size'],
            shuffle=False,
            num_workers=self.args['dataloadworkers'],
            pin_memory=True,
            drop_last=False
        )

        all_filenames = []
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, labels, filenames) in enumerate(tqdm(test_loader, desc="Testing Progress")):
            images = images.to(self.device)
            
            all_labels.append(labels.numpy())

            with torch.no_grad():
                with autocast(enabled=self.args.get('mixed_precision', False), device_type=self.device.type):
                    outputs = self.model(images)
                    if isinstance(outputs, list) and self.args.get('model', '').lower() == 'reaunet':
                        outputs = outputs[-1]
        
            all_filenames.extend(filenames)
            
            if outputs.shape[1] == 1:
                if self.args.get('model', '').lower() == 'reaunet':
                    probs = outputs
                else:
                    probs = torch.sigmoid(outputs)
                all_predictions.append(probs.cpu())
            else:
                probs = torch.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu())
            
            del images, outputs
            torch.cuda.empty_cache()
        
        all_predictions = torch.cat(all_predictions, dim=0)
        
        saved_prediction_paths = self._save_predictions(all_predictions, all_filenames)
        
        self.metric.reset()
        
        processed_preds = []
        processed_labels = []

        for i, (filename, label) in enumerate(tqdm(zip(all_filenames, itertools.chain(*all_labels)), 
                                                     total=len(all_filenames), 
                                                     desc="Loading Predictions and Labels")):
            if isinstance(filename, tuple):
                filename = filename[0]
            
            pred_path = os.path.join(self.args['test_result'], f"{filename}")
            if not pred_path.endswith('.tif'):
                pred_path = f"{pred_path}.tif"
            
            if os.path.exists(pred_path):
                with rasterio.open(pred_path) as src:
                    pred = src.read(1)
                    
                    pred_tensor = torch.from_numpy(pred).unsqueeze(0).to(self.device)
                    
                    if pred_tensor.max() > 1:
                        pred_tensor = (pred_tensor > 0).float()
            
                label_tensor = torch.from_numpy(label).to(self.device)
                if len(label_tensor.shape) == 2:
                    label_tensor = label_tensor.unsqueeze(0)
                
                self.metric.add_batch(label_tensor, pred_tensor)
                
                pred_array = pred_tensor.cpu().numpy().astype(np.uint8)
                if len(pred_array.shape) > 2:
                    if pred_array.shape[0] == 1:
                        pred_array = pred_array.squeeze(0)
                    elif len(pred_array.shape) > 2 and pred_array.shape[1] == 1:
                        pred_array = pred_array.squeeze(1)
                processed_preds.append(pred_array)

                label_array = label_tensor.cpu().numpy().astype(np.uint8)
                if len(label_array.shape) > 2:
                    if label_array.shape[0] == 1:
                        label_array = label_array.squeeze(0)
                    elif len(label_array.shape) > 2 and label_array.shape[1] == 1:
                        label_array = label_array.squeeze(1)
                processed_labels.append(label_array)

        all_preds = np.array(processed_preds)
        all_targets = np.array(processed_labels)
        
        pixelwise_metrics = self.metric.get_classwise_metrics()
        
        goc, guc, gtc = self.metric.calculate_over_under(all_preds, all_targets)
        
        object_metrics = {
            'goc': goc if goc is not None else float('nan'),
            'guc': guc if guc is not None else float('nan'),
            'gtc': gtc if gtc is not None else float('nan')
        }
                
        results = {**pixelwise_metrics, **object_metrics}
        
        self._print_test_results(results)

        self._save_metrics_to_json(results, self.metric.Confusion_Matrix().cpu().numpy())
        
        return results
    
    def _save_predictions(self, predictions, filenames):
        output_dir = self.args['test_result']
        os.makedirs(output_dir, exist_ok=True)
        
        foreground_value = self.args.get('foreground_value', 255)
        background_value = self.args.get('background_value', 0)
        
        use_dp = self.args.get('enable_douglas_peucker', False)
        
        if use_dp:
            vector_dir = os.path.join(output_dir, "Douglas-Peucker_Vectors")
            os.makedirs(vector_dir, exist_ok=True)
            
            dp_processor = DouglasPeuckerProcessor(
                tolerance=self.args.get('douglas_peucker_tolerance', 1.5),
                spike_angle=self.args.get('douglas_peucker_angle', 30.0),
                min_area=self.args.get('douglas_peucker_min_area', 10),
                target_value=self.args.get('douglas_peucker_target_value', foreground_value),
                args=self.args
            )
        
        is_batch = len(predictions.shape) > 3
        if not is_batch:
            predictions = predictions.unsqueeze(0)
            filenames = [filenames]
        
        all_processed_paths = []
        
        for i, (pred, filename) in enumerate(tqdm(zip(predictions, filenames), 
                                                     total=len(filenames), 
                                                     desc="Saving Predictions")):
            if isinstance(filename, tuple):
                filename = filename[0]
            
            original_image_path = os.path.join(self.args['test_image_folder'], f"{filename}")
            if not os.path.exists(original_image_path) and not original_image_path.endswith('.tif'):
                original_image_path = f"{original_image_path}.tif"
            
            output_path = os.path.join(output_dir, f"{filename}")
            if not output_path.endswith('.tif'):
                output_path = f"{output_path}.tif"
            
            try:
                with rasterio.open(original_image_path) as src:
                    profile = src.profile.copy()
                    profile.update(
                        count=1,
                        dtype=rasterio.uint8,
                        compress='lzw'
                    )
            except (rasterio.errors.RasterioIOError, FileNotFoundError) as e:
                print(f"Warning: Could not get georeferencing from original image: {str(e)}")
                profile = {
                    'driver': 'GTiff',
                    'height': pred.shape[0] if len(pred.shape) > 1 else pred.shape[0],
                    'width': pred.shape[1] if len(pred.shape) > 1 else 1,
                    'count': 1,
                    'dtype': rasterio.uint8,
                    'compress': 'lzw',
                    'nodata': 0
                }
            
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu()
            
            num_classes = self.args.get('classNum', 1)
            is_binary = num_classes <= 2
            is_classification = (is_binary and self.args.get('binary_output') == "classification") or \
                                  (not is_binary and self.args.get('multiclass_output', "labels") == "labels")
            
            if len(pred.shape) > 2:
                pred = pred.squeeze()
                
            if len(pred.shape) == 3 and pred.shape[0] == 2:
                if num_classes <= 2:
                    pred_numpy = pred[1].numpy() if isinstance(pred, torch.Tensor) else pred[1]
                    
                    if self.args.get('binary_output') == "classification":
                        pred_array = (pred_numpy > 0.5).astype(np.uint8) * foreground_value
                    else:
                        pred_array = pred_numpy.astype(np.float32)
                        profile['dtype'] = rasterio.float32
                else:
                    if isinstance(pred, torch.Tensor):
                        pred_array = torch.argmax(pred, dim=0).numpy().astype(np.uint8)
                    else:
                        pred_array = np.argmax(pred, axis=0).astype(np.uint8)
            else:
                if isinstance(pred, torch.Tensor):
                    pred_numpy = pred.squeeze().numpy()
                else:
                    pred_numpy = pred.squeeze()
                    
                if num_classes <= 2:
                    if self.args.get('binary_output') == "classification":
                        pred_array = np.where(pred_numpy > 0.5, foreground_value, background_value).astype(np.uint8)
                    else:
                        pred_array = pred_numpy.astype(np.float32)
                        profile['dtype'] = rasterio.float32
                else:
                    pred_array = pred_numpy.astype(np.uint8)
                    
            if len(pred_array.shape) > 2:
                pred_array = pred_array.squeeze()
            
            profile['height'] = pred_array.shape[0]
            profile['width'] = pred_array.shape[1]

            if self.args.get('enable_morphology', False) and is_classification and is_binary:
                morphology_op = self.args.get('morphology_operation', 'dilation')
                kernel_size = self.args.get('erosion_size', 2) if morphology_op == 'erosion' else self.args.get('dilation_size', 2)
                
                pred_array = apply_morphology(
                    pred_array,
                    operation=morphology_op,
                    kernel_size=kernel_size,
                    foreground_value=foreground_value
                )

            if use_dp and is_classification:
                temp_path = output_path.replace('.tif', '_temp.tif')
                
                with rasterio.open(temp_path, 'w', **profile) as dst:
                    dst.write(pred_array, 1)
                    if 'descriptions' in profile:
                        dst.descriptions = profile['descriptions']
                    if 'tags' in profile:
                        for tag, value in profile['tags'].items():
                            dst.update_tags(**{tag: value})
                
                processed_path, vector_path = dp_processor.process_raster(
                    temp_path,
                    output_path,
                    vector_dir
                )
                all_processed_paths.append(processed_path)
                
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            else:
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(pred_array, 1)
                    if 'descriptions' in profile:
                        dst.descriptions = profile['descriptions']
                    if 'tags' in profile:
                        for tag, value in profile['tags'].items():
                            dst.update_tags(**{tag: value})
                
                all_processed_paths.append(output_path)
        
        return all_processed_paths
    
    def _save_metrics_to_json(self, metrics, confusion_matrix):
        import json
        import datetime
        
        conf_matrix_list = confusion_matrix.tolist() if hasattr(confusion_matrix, 'tolist') else confusion_matrix
        
        class_names = ['background', 'cropland']
        
        json_metrics = {
            "global_metrics": {},
            "class_metrics": {
                "background": {},
                "cropland": {}
            },
            "object_metrics": {},
            "confusion_matrix": {
                "matrix": conf_matrix_list,
                "true_labels": [f"True {i}" for i in range(len(confusion_matrix))],
                "predicted_labels": [f"Pred {i}" for i in range(len(confusion_matrix))]
            },
            "metadata": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.args.get('model', 'unknown'),
                "classes": self.args.get('classNum', 1),
                "input_size": self.args.get('input_size', 'unknown'),
                "batch_size": self.args.get('batch_size', 1),
                "device": str(self.device),
                "test_image_folder": self.args.get('test_image_folder', 'unknown'),
                "test_label_folder": self.args.get('test_label_folder', 'unknown'),
            }
        }
        
        for metric_name, value in metrics.items():
            if metric_name in ['goc', 'guc', 'gtc']:
                json_metrics["object_metrics"][metric_name] = float(value) if not np.isnan(value) else None
                continue
                
            if metric_name in ['kappa', 'overall_accuracy']:
                json_metrics["global_metrics"][metric_name] = float(value)
                continue
                
            for class_name in class_names:
                if f"_{class_name}" in metric_name:
                    base_metric = metric_name.split(f"_{class_name}")[0]
                    json_metrics["class_metrics"][class_name][base_metric] = float(value)
                    break
        
        metrics_path = os.path.join(self.args['test_result'], 'test_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_metrics, jsonfile, indent=2)
            
        print(f"Test metrics saved to {metrics_path}")
        
    def _print_test_results(self, results):
        print("\n===== TEST RESULTS =====")
        print(f"Overall Accuracy: {results.get('overall_accuracy', 'N/A'):.4f}")
        
        print("\nCropland Class Metrics:")
        print(f"IoU: {results.get('iou_cropland', 'N/A'):.4f}")
        print(f"F1-Score: {results.get('f1_score_cropland', 'N/A'):.4f}")
        print(f"Precision: {results.get('precision_cropland', 'N/A'):.4f}")
        print(f"Recall: {results.get('recall_cropland', 'N/A'):.4f}")
        
        print("\nBackground Class Metrics:")
        print(f"IoU: {results.get('iou_background', 'N/A'):.4f}")
        print(f"F1-Score: {results.get('f1_score_background', 'N/A'):.4f}")
        print(f"Precision: {results.get('precision_background', 'N/A'):.4f}")
        print(f"Recall: {results.get('recall_background', 'N/A'):.4f}")
        
        print("\nObject-Level Metrics:")
        print(f"Global Over-segmentation (GOC): {results.get('goc', 'N/A'):.4f}")
        print(f"Global Under-segmentation (GUC): {results.get('guc', 'N/A'):.4f}")
        print(f"Global Total Segmentation Error (GTC): {results.get('gtc', 'N/A'):.4f}")
        
        print(f"\nKappa Coefficient: {results.get('kappa', 'N/A'):.4f}")

class Inferencer(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.infer_result_dir = self.args['infer_result_folder']
        os.makedirs(self.infer_result_dir, exist_ok=True)
        
        self.window_func = args.get('window_func', 'spline')
        
        torch.cuda.empty_cache()
        
        self.model.to(self.device)
        self.model.eval()
        
        if self.args.get('enable_tta'):
            import ttach as tta
            transforms = tta.Compose([
                tta.Scale(scales=[0.75, 1.25],interpolation='bilinear',align_corners=True),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ])
            self.tta_model = tta.SegmentationTTAWrapper(
                self.model, 
                transforms, 
                merge_mode='mean'
            )
            print(f"TTA enabled with transforms")
        else:
            self.tta_model = self.model
            
        print(f"Model ready for inference on {self.device}")
        
        self.cached_window_functions = {}
        self.cached_2d_windows = {}
    
    def predict_folder(self, input_folder, output_folder=None, clip_size=512, subdivisions=2, nodata=None, batch_size=1):
        if output_folder is None:
            output_folder = self.infer_result_dir
            
        os.makedirs(output_folder, exist_ok=True)
        
        supported_extensions = ['*.tif', '*.tiff', '*.img']
        image_files = []
        
        for pattern in supported_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, pattern)))
        
        if not image_files:
            raise ValueError(f"No images found in {input_folder} with extensions {supported_extensions}")
        
        if self.args.get('huge_image_infer', False):
            for i, image_file in enumerate(image_files):
                base_name = os.path.basename(image_file)
                output_file = os.path.join(output_folder, f"pred_{base_name}")
                
                print(f"[{i+1}/{len(image_files)}] Processing huge image {base_name}")
                
                self.predict_huge_image(
                    image_file,
                    output_file,
                    self.args.get('split_width', 2),
                    self.args.get('split_length', 2),
                    clip_size=clip_size,
                    subdivisions=subdivisions,
                    nodata=nodata,
                    batch_size=batch_size
                )
                
                torch.cuda.empty_cache()
                gc.collect()
        else:
            for i, image_file in enumerate(image_files):
                base_name = os.path.basename(image_file)
                output_file = os.path.join(output_folder, f"pred_{base_name}")
                
                print(f"[{i+1}/{len(image_files)}] Processing image {base_name}")
                
                self.predict_image(
                    image_file,
                    output_file,
                    window_size=clip_size,
                    subdivisions=subdivisions,
                    nodata=nodata,
                    batch_size=batch_size
                )
                
                torch.cuda.empty_cache()
                gc.collect()
            
        return output_folder
    
    def predict_huge_image(self, image_path, output_path, grid_cols, grid_rows, clip_size=512, 
                         subdivisions=2, nodata=None, batch_size=1):
        with rasterio.open(image_path) as src:
            meta = src.meta.copy()
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs
        tile_width = width // grid_cols
        tile_height = height // grid_rows
        
        last_col_width = width - (grid_cols - 1) * tile_width
        last_row_height = height - (grid_rows - 1) * tile_height
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        temp_folder = os.path.join(os.path.dirname(image_path), f"{image_name}_temp")
        os.makedirs(temp_folder, exist_ok=True)
        
        temp_output_folder = os.path.join(os.path.dirname(output_path), f"{image_name}_temp_output")
        os.makedirs(temp_output_folder, exist_ok=True)
        
        print(f"Splitting huge image into {grid_rows}x{grid_cols} grid...")
        
        temp_files = []
        temp_output_files = []
        grid_info = []
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                win_row_start = row * tile_height
                win_col_start = col * tile_width
                
                win_width = last_col_width if col == grid_cols - 1 else tile_width
                win_height = last_row_height if row == grid_rows - 1 else tile_height
                
                window = rasterio.windows.Window(
                    win_col_start, win_row_start, win_width, win_height
                )
                
                temp_file = os.path.join(temp_folder, f"{image_name}_r{row}_c{col}.tif")
                temp_output_file = os.path.join(temp_output_folder, f"pred_{image_name}_r{row}_c{col}.tif")
                
                temp_files.append(temp_file)
                temp_output_files.append(temp_output_file)
                grid_info.append((row, col, win_row_start, win_col_start, win_height, win_width))
                
                with rasterio.open(image_path) as src:
                    data = src.read(window=window)
                    
                    temp_transform = rasterio.windows.transform(window, src.transform)
                    temp_meta = src.meta.copy()
                    temp_meta.update({
                        'height': win_height,
                        'width': win_width,
                        'transform': temp_transform
                    })
                    
                    with rasterio.open(temp_file, 'w', **temp_meta) as dst:
                        dst.write(data)
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Performing inference on {len(temp_files)} tiles...")
        for i, temp_file in enumerate(temp_files):
            print(f"Processing tile {i+1}/{len(temp_files)}")
            self.predict_image(
                temp_file,
                temp_output_files[i],
                window_size=clip_size,
                subdivisions=subdivisions,
                nodata=nodata,
                batch_size=batch_size,
                skip_dp=True
            )
        
        num_classes = self.args.get('classNum', 1)
        is_binary = num_classes <= 2
        
        if is_binary:
            if self.args.get('binary_output') == 'regression':
                out_dtype = rasterio.float32
            else:
                out_dtype = rasterio.uint8
        else:
            if self.args.get('multiclass_output') == 'labels':
                out_dtype = rasterio.uint8
            else:
                out_dtype = rasterio.float32
        
        print("Merging tiles into final output...")
        with rasterio.open(temp_output_files[0]) as first_tile:
            count = first_tile.count
            
            meta.update(dtype=first_tile.dtypes[0])
            meta.update(count=count)
            meta.update(compress='lzw')
            meta.update(nodata=nodata)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                for i, output_file in enumerate(tqdm(temp_output_files, desc="Merging tiles")):
                    row, col, row_start, col_start, height, width = grid_info[i]
                    
                    with rasterio.open(output_file) as src:
                        data = src.read()
                        
                        window = rasterio.windows.Window(col_start, row_start, width, height)
                        dst.write(data, window=window)
        
        use_dp = self.args.get('enable_douglas_peucker', False)
        is_classification = (is_binary and self.args.get('binary_output', 'classification') == "classification") or \
                              (not is_binary and self.args.get('multiclass_output', "labels") == "labels")
        
        if use_dp and is_classification:
            output_dir = os.path.dirname(output_path)
            vector_dir = os.path.join(output_dir, "vectors")
            os.makedirs(vector_dir, exist_ok=True)
            
            dp_processor = DouglasPeuckerProcessor(
                tolerance=self.args.get('douglas_peucker_tolerance', 1.5),
                spike_angle=self.args.get('douglas_peucker_angle', 30.0),
                min_area=self.args.get('douglas_peucker_min_area', 10),
                target_value=self.args.get('douglas_peucker_target_value', 255),
                args=self.args
            )
            
            dp_processor.process_raster(output_path, output_path, vector_dir)
        
        if self.args.get('clean_temp_files', True):
            import shutil
            shutil.rmtree(temp_folder, ignore_errors=True)
            shutil.rmtree(temp_output_folder, ignore_errors=True)
            
        print(f"Huge image inference completed. Output saved to {output_path}")
        return output_path

    def predict_image(self, image_path, output_path, window_size=512, subdivisions=2, nodata=0, batch_size=1, skip_dp=False):
        foreground_value = self.args.get('foreground_value', 255)
        background_value = self.args.get('background_value', 0)
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            input_img = src.read()
            input_img = np.transpose(input_img, (1, 2, 0))
        
        pad = self._pad_img(input_img, window_size, subdivisions)
        
        padded_results = self._predict_with_smooth_window(
            pad, window_size, subdivisions, batch_size=batch_size
        )
        
        prd = self._unpad_img(padded_results, window_size, subdivisions)
        
        prd = prd[:input_img.shape[0], :input_img.shape[1], :]
        
        num_classes = self.args.get('classNum', 1)
        is_binary = num_classes <= 2
        
        use_dp = self.args.get('enable_douglas_peucker', False)
        is_classification = (is_binary and self.args.get('binary_output', 'classification') == "classification") or \
                              (not is_binary and self.args.get('multiclass_output', "labels") == "labels")
        
        temp_output_path = output_path
        if use_dp and is_classification:
            temp_output_path = output_path.replace('.tif', '_temp.tif')
        
        if is_binary:
            prd = prd[:, :, 0]
            
            if self.args.get('binary_output') == 'regression':
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress='lzw',
                    nodata=nodata
                )
                prd = prd.reshape((1, prd.shape[0], prd.shape[1]))
                
                with rasterio.open(temp_output_path, 'w', **profile) as dst:
                    dst.write(prd.astype(rasterio.float32))
            else:
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    compress='lzw',
                    nodata=nodata
                )
                prd = np.where(prd > 0.5, foreground_value, background_value).astype(np.uint8)
                
                if self.args.get('enable_morphology', False):
                    morphology_op = self.args.get('morphology_operation', 'dilation')
                    kernel_size = self.args.get('erosion_size', 2) if morphology_op == 'erosion' else self.args.get('dilation_size', 2)
                    
                    prd = apply_morphology(
                        prd,
                        operation=morphology_op,
                        kernel_size=kernel_size,
                        foreground_value=foreground_value
                    )
                
                prd = prd.reshape((1, prd.shape[0], prd.shape[1]))
                
                with rasterio.open(temp_output_path, 'w', **profile) as dst:
                    dst.write(prd.astype(rasterio.uint8))
        else:
            output_mode = self.args.get('multiclass_output', 'labels')
            
            if output_mode == 'labels':
                if prd.shape[2] == 1:
                    class_indices = prd[:, :, 0]
                else:
                    class_indices = np.argmax(prd, axis=2)
                    
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    compress='lzw',
                    nodata=nodata
                )
                
                with rasterio.open(temp_output_path, 'w', **profile) as dst:
                    dst.write(class_indices.reshape(1, *class_indices.shape).astype(rasterio.uint8))
            else:
                profile.update(
                    dtype=rasterio.float32,
                    count=prd.shape[2],
                    compress='lzw',
                    nodata=nodata
                )
                
                prd = np.transpose(prd, (2, 0, 1))
                
                with rasterio.open(temp_output_path, 'w', **profile) as dst:
                    dst.write(prd.astype(rasterio.float32))

        if use_dp and is_classification and not skip_dp:
            output_dir = os.path.dirname(output_path)
            vector_dir = os.path.join(output_dir, "vectors")
            os.makedirs(vector_dir, exist_ok=True)
            
            dp_processor = DouglasPeuckerProcessor(
                tolerance=self.args.get('douglas_peucker_tolerance', 1.5),
                spike_angle=self.args.get('douglas_peucker_angle', 30.0),
                min_area=self.args.get('douglas_peucker_min_area', 10),
                target_value=self.args.get('douglas_peucker_target_value', 255)
            )
            
            dp_processor.process_raster(temp_output_path, output_path, vector_dir)
            
            if temp_output_path != output_path and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
        elif temp_output_path != output_path:
            os.rename(temp_output_path, output_path)
        
        print(f"Inference completed. Output saved to {output_path}")
        return output_path
    
    def _predict_with_smooth_window(self, padded_img, window_size, subdivisions, batch_size=1):
        use_weighted_window = self.args.get('enable_weighted_window', True)
        num_classes = self.args.get('classNum', 1)
        is_binary = num_classes <= 2
        out_channels = 1 if is_binary else num_classes
        
        step = int(window_size/subdivisions)
        padx_len = padded_img.shape[0]
        pady_len = padded_img.shape[1]
        grid_x = (padx_len - window_size) // step + 1
        grid_y = (pady_len - window_size) // step + 1
        
        total_windows = grid_x * grid_y
        
        if use_weighted_window:
            if self.window_func != 'none':
                window_weight = self._window_2D(window_size=window_size, power=2)
                window_weight = window_weight.reshape(window_size, window_size, 1)
            else:
                window_weight = np.ones((window_size, window_size, 1), dtype=np.float32)
                if subdivisions > 1:
                    edge_width = int(window_size/subdivisions/2)
                    if edge_width > 0:
                        for i in range(edge_width):
                            factor = (i + 1) / (edge_width + 1)
                            window_weight[i, :, 0] *= factor
                            window_weight[-i-1, :, 0] *= factor
                            window_weight[:, i, 0] *= factor
                            window_weight[:, -i-1, 0] *= factor
            
            result = np.zeros((padx_len, pady_len, out_channels), dtype=np.float32)
            count_mask = np.zeros_like(result)
            
            patches = []
            positions = []
            
            window_positions = [(i, j) for i in range(grid_x) for j in range(grid_y)]
            for idx, (i, j) in enumerate(tqdm(window_positions, total=total_windows, desc="Processing windows")):
                x = i * step
                y = j * step
                
                window = padded_img[x:x+window_size, y:y+window_size, :]
                patches.append(window)
                positions.append((x, y))
                        
                if len(patches) == batch_size or (i, j) == window_positions[-1]:
                    preds = self._predict_batch(patches)
                    
                    for k, (px, py) in enumerate(positions):
                        if k < len(preds):
                            pred = preds[k]
                            
                            if is_binary:
                                pred = np.transpose(pred, (1, 2, 0))
                                result[px:px+window_size, py:py+window_size] += pred * window_weight
                                count_mask[px:px+window_size, py:py+window_size] += window_weight
                            else:
                                pred = np.transpose(pred, (1, 2, 0))
                                
                                result[px:px+window_size, py:py+window_size] += pred * window_weight
                                count_mask[px:px+window_size, py:py+window_size] += window_weight
                    
                    patches = []
                    positions = []
            
            result = np.divide(result, count_mask, out=np.zeros_like(result), where=count_mask!=0)
        
        else:
            result = np.zeros((padx_len, pady_len, out_channels), dtype=np.float32)
            processed_mask = np.zeros((padx_len, pady_len), dtype=bool)
            
            window_positions = [(i, j) for i in range(grid_x) for j in range(grid_y)]
            
            for idx, (i, j) in enumerate(tqdm(window_positions, total=total_windows, desc="Processing windows")):
                x = i * step
                y = j * step
                
                window = padded_img[x:x+window_size, y:y+window_size, :]
                
                pred = self._predict_batch([window])[0]
                
                if is_binary:
                    pred = np.transpose(pred, (1, 2, 0))
                else:
                    pred = np.transpose(pred, (1, 2, 0))
                
                mask = np.zeros((window_size, window_size), dtype=bool)
                
                if i == 0:
                    mask[:, :step//2] = True
                else:
                    mask[:, :step//2] = True
                
                if i == grid_x - 1:
                    mask[:, step//2:] = True
                else:
                    mask[:, step//2:-step//2] = True
                
                if j == 0:
                    mask[:step//2, :] = True
                else:
                    mask[:step//2, :] = True
                
                if j == grid_y - 1:
                    mask[step//2:, :] = True
                else:
                    mask[step//2:-step//2, :] = True
                
                effective_mask = ~processed_mask[x:x+window_size, y:y+window_size] & mask
                
                if is_binary:
                    result[x:x+window_size, y:y+window_size, 0][effective_mask] = pred[:, :, 0][effective_mask]
                else:
                    for c in range(out_channels):
                        result[x:x+window_size, y:y+window_size, c][effective_mask] = pred[:, :, c][effective_mask]
                
                processed_mask[x:x+window_size, y:y+window_size][mask] = True
        
        return result

    def _pad_img(self, img, window_size, subdivisions):
        aug = int(round(window_size * (1 - 1.0/subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        ret = np.pad(img, pad_width=more_borders, mode='reflect')
        return ret
    
    def _unpad_img(self, padded_img, window_size, subdivisions):
        aug = int(round(window_size * (1 - 1.0/subdivisions)))
        ret = padded_img[aug:-aug, aug:-aug, :]
        return ret
    
    def _create_window_function(self, window_size, function_type='spline', power=2):
        key = f"{window_size}_{function_type}_{power}"
        if key in self.cached_window_functions:
            return self.cached_window_functions[key]
        
        if function_type == 'spline':
            from scipy.signal.windows import triang
            
            intersection = int(window_size/4)
            wind_outer = (abs(2*(triang(window_size))) ** power)/2
            wind_outer[intersection:-intersection] = 0

            wind_inner = 1 - (abs(2*(triang(window_size) - 1)) ** power)/2
            wind_inner[:intersection] = 0
            wind_inner[-intersection:] = 0

            window = wind_inner + wind_outer
            
        elif function_type == 'linear':
            center = window_size // 2
            window = np.zeros(window_size)
            for i in range(window_size):
                window[i] = 1.0 - abs(i - center) / center
                
        elif function_type == 'quadratic':
            center = window_size / 2
            window = np.zeros(window_size)
            for i in range(window_size):
                relative_pos = (i - center) / center
                window[i] = 1.0 - (relative_pos ** 2)
                
        elif function_type == 'cubic':
            center = window_size / 2
            window = np.zeros(window_size)
            for i in range(window_size):
                relative_pos = (i - center) / center
                window[i] = 1.0 - (abs(relative_pos) ** 3)
                
        elif function_type == 'gaussian':
            from scipy.signal.windows import gaussian
            sigma = window_size / 6.0
            window = gaussian(window_size, sigma)
            
        else:
            raise ValueError(f"Unknown window function type: {function_type}")
        
        window = window / np.average(window)
        
        self.cached_window_functions[key] = window
        return window

    def _window_2D(self, window_size, power=2):
        key = f"{window_size}_{self.window_func}_{power}"
        if key in self.cached_2d_windows:
            return self.cached_2d_windows[key]
        
        window_1d = self._create_window_function(window_size, self.window_func, power)
        
        wind_col = window_1d.reshape(window_size, 1)
        window_2d = wind_col * window_1d.reshape(1, window_size)
        window_2d = np.reshape(window_2d, (window_size, window_size, 1))
        
        self.cached_2d_windows[key] = window_2d
        return window_2d
            
    def _predict_batch(self, batch):
        from Utils.dataset import CustomDataset
        
        batch_tensors = []
        
        for chunk in batch:
            tensor = CustomDataset.transform_infer_array(
                image_array=chunk,
                mean_list=self.args['MeanList'],
                std_list=self.args['StdList']
            ).to(self.device)
            
            batch_tensors.append(tensor)
        
        if len(batch_tensors) > 1:
            input_batch = torch.stack(batch_tensors)
        else:
            input_batch = batch_tensors[0].unsqueeze(0)
        
        with torch.no_grad():
            with autocast(enabled=self.args.get('mixed_precision'), device_type=self.device.type):
                outputs = self.tta_model(input_batch)
                if isinstance(outputs, list) and self.args.get('model', '').lower() == 'reaunet':
                    outputs = outputs[-1]
        if outputs.shape[1] == 1:
            if self.args.get('model', '').lower() == 'reaunet':
                preds = outputs.cpu().numpy()
            else:
                preds = torch.sigmoid(outputs).cpu().numpy()
        else:
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return preds

def apply_morphology(image, operation='dilation', kernel_size=2, foreground_value=255):
    from scipy import ndimage
    
    binary = (image == foreground_value).astype(np.uint8)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    if operation == 'erosion':
        result = ndimage.binary_erosion(binary, structure=kernel).astype(np.uint8)
    else:
        result = ndimage.binary_dilation(binary, structure=kernel).astype(np.uint8)
    
    return np.where(result > 0, foreground_value, 0).astype(image.dtype)

def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description="Model testing and inference with advanced edge blending")
    parser.add_argument('--mode', type=str, default='inference', choices=['test', 'inference'],
                        help='Operation mode: test or inference')
    parser.add_argument('--checkpoint-path', type=str, default= '',
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for testing')
    parser.add_argument('--dataloadworkers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--foreground-value', type=int, default=255,
                        help='Value for foreground pixels in binary classification output (default: 255)')
    parser.add_argument('--background-value', type=int, default=0,
                        help='Value for background pixels in binary classification output (default: 0)')
    parser.add_argument('--enable-tta', type=lambda x: str(x).lower() == 'true', 
                        default=True,
                        help='Enable test-time augmentation with 8 transformations (default: True)')
    parser.add_argument('--enable-morphology', type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='Enable morphological operations on binary output (default: False)')
    parser.add_argument('--morphology-operation', type=str, default='erosion',
                        choices=['erosion', 'dilation'],
                        help='Type of morphological operation to apply (default: dilation)')
    parser.add_argument('--erosion-size', type=int, default=3,
                        help='Size of erosion kernel (default: 2 pixels)')
    parser.add_argument('--dilation-size', type=int, default=3,
                        help='Size of dilation kernel (default: 2 pixels)')
    parser.add_argument('--enable-douglas-peucker', type=lambda x: str(x).lower() == 'true', 
                        default=True,
                        help='Apply Douglas-Peucker algorithm for polygon simplification')
    parser.add_argument('--douglas-peucker-tolerance', type=float, default=1.2,
                        help='Tolerance value for Douglas-Peucker simplification')
    parser.add_argument('--douglas-peucker-angle', type=float, default=30,
                        help='Maximum angle for spike elimination (degrees)')
    parser.add_argument('--douglas-peucker-min-area', type=int, default=30,
                        help='Minimum polygon area for filtering (pixels)')
    parser.add_argument('--douglas-peucker-target-value', type=int, default=255,
                        help='Target value for polygonization (default: 255 for binary classification)')
    parser.add_argument('--multiclass-output', type=str, default='labels',
                        choices=['labels', 'probabilities'],
                        help='Output format for multiclass: labels (single channel with class indices) or '
                             'probabilities (multiple channels with class probabilities)')
    parser.add_argument('--binary-output', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='For binary model: classification (output binary mask) or regression (output probability map)')
    parser.add_argument('--nodata', type=int, default=None,
                        help='No data value in input images')
                        
    parser.add_argument('--test-image-folder', type=str,default='',
                        help='Folder containing test images')
    parser.add_argument('--test-label-folder', type=str, default='',
                        help='Folder containing test labels')
    parser.add_argument('--test-result', type=str, default='',
                        help='Folder to save test results')
    parser.add_argument('--buffer-pixels', type=int, default=5,
                        help='Buffer size for F1-boundary calculation (pixels, 5m = 5 pixels at 1m resolution)')
                        
    parser.add_argument('--infer-image-folder', type=str, default= '',
                        help='Folder containing images for inference')
    parser.add_argument('--infer-result-folder', type=str, default= '',
                        help='Folder to save inference results')
    parser.add_argument('--clip-size', type=int, default=512,
                        help='Size of image chunks for processing')
    parser.add_argument('--stride-size', type=int, default=128,
                        help='Stride between consecutive chunks')
    parser.add_argument('--window-func', type=str, default='spline',
                        choices=['spline', 'linear', 'quadratic', 'cubic', 'gaussian'],
                        help='Weight window function for edge blending (default: spline)')
    parser.add_argument('--enable-weighted-window', type=lambda x: str(x).lower() == 'true', 
                        default=False,
                        help='Enable weighted window for smooth blending (default: True)')
    parser.add_argument('--huge-image-infer', type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='Enable grid-based large image processing mode (default: False)')
    parser.add_argument('--split-width', type=int, default=2,
                        help='Number of horizontal grid divisions for huge image processing (e.g., 4 means split into 4 columns)')
    parser.add_argument('--split-length', type=int, default=2,
                        help='Number of vertical grid divisions for huge image processing (e.g., 4 means split into 4 rows)')
    parser.add_argument('--clean-temp-files', type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='Remove temporary files after processing (default: True)')

    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    args_dict = vars(args)
    
    args_dict['foreground_value'] = args.foreground_value
    args_dict['background_value'] = args.background_value
    
    if args.mode == 'test':
        tester = Tester(args_dict)
        results = tester.test()
        print("Testing completed")
    
    elif args.mode == 'inference':
        inferencer = Inferencer(args_dict)
        
        inferencer.window_func = args.window_func
        inferencer.enable_tta = args.enable_tta
        
        subdivisions = int(args.clip_size / args.stride_size) if args.stride_size > 0 else 1
        
        inferencer.predict_folder(
            args.infer_image_folder,
            args.infer_result_folder,
            clip_size=args.clip_size,
            subdivisions=subdivisions,
            nodata=args.nodata,
            batch_size=args.batch_size
        )
        
        num_classes = inferencer.args.get('classNum', 1)
        is_binary = num_classes <= 2

        if args.enable_weighted_window:
            window_strategy = f"{args.window_func} weighted window"
        else:
            window_strategy = "hard boundary splitting"

        if is_binary:
            task_type = "probability map" if args.binary_output == "regression" else f"binary mask"
        else:
            task_type = "class labels" if args.multiclass_output == "labels" else "class probabilities"

        print("\n" + "="*50)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"✓ Model: {inferencer.args.get('model', 'unknown')} with {num_classes} {'class' if is_binary else 'classes'}")
        print(f"✓ Window size: {args.clip_size}px with {args.stride_size}px stride ({subdivisions}x subdivisions)")
        print(f"✓ Blending strategy: {window_strategy}")
        print(f"✓ Processing: {'With TTA (8 transforms)' if args.enable_tta else 'No test-time augmentation'}")
        print(f"✓ Output format: {task_type}")
        
        if args.enable_douglas_peucker:
            print(f"✓ Douglas-Peucker enabled: tolerance={args.douglas_peucker_tolerance}, "
                  f"angle={args.douglas_peucker_angle}°, min_area={args.douglas_peucker_min_area}px")
            print(f"✓ Vector results saved to: {os.path.join(args.infer_result_folder, 'vectors')}")
        
        print(f"✓ Results saved to: {args.infer_result_folder}")
        print("="*50)

if __name__ == '__main__':
    main()
