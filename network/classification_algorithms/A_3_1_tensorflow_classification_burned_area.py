# last_update: '2025/06/02', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_3_1_tensorflow_classification_burned_area.py 
### Step A_3_1 - Functions for TensorFlow classification of burned areas

# ====================================
# 📦 INSTALL AND IMPORT LIBRARIES
# ====================================

import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Usar somente a versão compatível 1.x
from scipy import ndimage
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee  # For Google Earth Engine integration
from tqdm import tqdm  # For progress bars
import time
from datetime import datetime
import math
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
import pyproj
import shutil  # For file and folder operations
import json
import subprocess

# ====================================
# 🔧 GPU ACCELERATION SETUP (CuPy)
# ====================================

# Try to import CuPy for GPU-accelerated NumPy operations
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    GPU_AVAILABLE = True
    print("[INFO] CuPy detected. GPU acceleration enabled for data processing.")
except ImportError:
    cp = np  # Fallback to NumPy if CuPy is not available
    cp_ndimage = ndimage
    GPU_AVAILABLE = False
    print("[INFO] CuPy not available. Using NumPy for data processing (CPU).")

# ====================================
# 📝 GPU DEBUG LOGGING SETUP
# ====================================

import logging
import sys

# Create logger for GPU debugging
gpu_logger = logging.getLogger('gpu_debug')
gpu_logger.setLevel(logging.DEBUG)

# Create file handler
log_file = '/content/gpu_classification_debug.log'
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
gpu_logger.addHandler(file_handler)
gpu_logger.addHandler(console_handler)

gpu_logger.info("="*80)
gpu_logger.info("GPU DEBUG LOGGING INITIALIZED")
gpu_logger.info(f"Log file: {log_file}")
gpu_logger.info("="*80)

# ====================================
# 🧰 SUPPORT FUNCTIONS (utils)
# ====================================

# Function to load an image using GDAL
def load_image(image_path):
    log_message(f"[INFO] Loading image from path: {image_path}")
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

# Function to convert a GDAL dataset to a NumPy array (GPU-accelerated)
def convert_to_array(dataset):
    log_message(f"[INFO] Converting dataset to NumPy array")
    gpu_logger.info("Reading bands from GDAL dataset")

    # Check GPU status before operation
    gpu_status = get_gpu_status()
    if gpu_status:
        log_message(f"[GPU] Before array conversion: GPU {gpu_status['gpu_util']}% | Memory {gpu_status['mem_used']}MB/{gpu_status['mem_total']}MB ({gpu_status['mem_util']}%) | Temp {gpu_status['temperature']}°C")

    # Read all bands into CPU memory first
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]

    if GPU_AVAILABLE:
        gpu_logger.info(f"Using GPU (CuPy) for array stacking - {len(bands_data)} bands")
        # Move bands to GPU and stack there
        bands_gpu = [cp.asarray(band) for band in bands_data]
        stacked_data_gpu = cp.stack(bands_gpu, axis=2)

        # Free GPU memory from individual bands
        del bands_gpu
        cp.get_default_memory_pool().free_all_blocks()

        gpu_logger.info(f"Array stacked on GPU: shape={stacked_data_gpu.shape}, dtype={stacked_data_gpu.dtype}")

        # Check GPU status after operation
        gpu_status_after = get_gpu_status()
        if gpu_status_after:
            log_message(f"[GPU] After array conversion: GPU {gpu_status_after['gpu_util']}% | Memory {gpu_status_after['mem_used']}MB/{gpu_status_after['mem_total']}MB ({gpu_status_after['mem_util']}%) | Temp {gpu_status_after['temperature']}°C")

        return stacked_data_gpu  # Return GPU array
    else:
        gpu_logger.info("Using CPU (NumPy) for array stacking")
        stacked_data = np.stack(bands_data, axis=2)
        return stacked_data

# Function to reshape classified data back into image format
def reshape_image_output(output_data_classified, data_classify):
    log_message(f"[INFO] Reshaping classified data back to image format")
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Function to reshape classified data into a single pixel vector (GPU-aware)
def reshape_single_vector(data_classify):
    """
    Reshape data into single pixel vector. Works with both NumPy and CuPy arrays.
    If input is on GPU, output stays on GPU.
    """
    new_shape = [data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]]

    # Check GPU status before reshape
    gpu_status = get_gpu_status()
    if gpu_status:
        log_message(f"[GPU] Before reshape: GPU {gpu_status['gpu_util']}% | Memory {gpu_status['mem_used']}MB/{gpu_status['mem_total']}MB ({gpu_status['mem_util']}%) | Temp {gpu_status['temperature']}°C")

    if GPU_AVAILABLE and isinstance(data_classify, cp.ndarray):
        gpu_logger.info(f"Reshaping on GPU: {data_classify.shape} -> {new_shape}")
        result = data_classify.reshape(new_shape)

        # Check GPU status after reshape
        gpu_status_after = get_gpu_status()
        if gpu_status_after:
            log_message(f"[GPU] After reshape: GPU {gpu_status_after['gpu_util']}% | Memory {gpu_status_after['mem_used']}MB/{gpu_status_after['mem_total']}MB ({gpu_status_after['mem_util']}%) | Temp {gpu_status_after['temperature']}°C")

        return result
    else:
        gpu_logger.info(f"Reshaping on CPU: {data_classify.shape} -> {new_shape}")
        return data_classify.reshape(new_shape)

def filter_spatial(output_image_data):
    """
    Apply spatial filtering on a classified image with GPU acceleration support:
      1) conditional opening filter based on opening_spatial_filter
      2) conditional closing filter based on choose_spatial_filter

    Global opening_spatial_filter (optional):
      • False  → skip the closing step (close_image = open_image)
      • None or undefined → default to closing with 4×4
      • int N → closing with N×N
      • anything else → warning + default to 4×4

    Global choose_spatial_filter (optional):
      • False  → skip the closing step (open_image = binary_image)
      • None or undefined → default to closing with 2×2
      • int N → closing with N×N
      • anything else → warning + default to 2×2

    Parameters:
        output_image_data (ndarray): labeled or binary image where >0 is foreground.

    Returns:
        ndarray: result after opening+closing, as uint8.
    """
    # 1) Captura choose_spatial_filter sem estourar NameError
    try:
        cfs = closing_filter_size
    except NameError:
        cfs = None

    try:
        ofs = opening_filter_size
    except NameError:
        ofs = None


    log_message("[INFO] Applying spatial filtering on classified image")

    # Check GPU status before filtering
    gpu_status = get_gpu_status()
    if gpu_status:
        log_message(f"[GPU] Before spatial filtering: GPU {gpu_status['gpu_util']}% | Memory {gpu_status['mem_used']}MB/{gpu_status['mem_total']}MB ({gpu_status['mem_util']}%) | Temp {gpu_status['temperature']}°C")

    # Use GPU acceleration if available
    if GPU_AVAILABLE:
        log_message("[INFO] Using GPU acceleration (CuPy) for spatial filtering")
        # Move data to GPU
        output_gpu = cp.asarray(output_image_data)

        # 2) Binariza
        binary_image = output_gpu > 0

        # 3) Decide o opening
        if ofs is False:
            log_message("[INFO] Skipping opening filter step as requested.")
            open_image = binary_image
        else:
            # define M
            try:
                m = int(ofs) if ofs is not None else 2
            except (ValueError, TypeError):
                log_message(f"[WARNING] Invalid opening filter size '{ofs}'; defaulting to 2×2.")
                m = 2

            log_message(f"[INFO] Applying opening filter with {m}x{m} structuring element.")
            open_image = cp_ndimage.binary_opening(binary_image, structure=cp.ones((m, m)))

        # 4) Decide o closing
        if cfs is False:
            log_message("[INFO] Skipping closing filter step as requested.")
            close_image = open_image
        else:
            # define N
            try:
                n = int(cfs) if cfs is not None else 4
            except (ValueError, TypeError):
                log_message(f"[WARNING] Invalid closing filter size '{cfs}'; defaulting to 4×4.")
                n = 4

            log_message(f"[INFO] Applying closing filter with {n}×{n} structuring element.")
            close_image = cp_ndimage.binary_closing(open_image, structure=cp.ones((n, n)))

        # Check GPU status after filtering
        gpu_status_after = get_gpu_status()
        if gpu_status_after:
            log_message(f"[GPU] After spatial filtering: GPU {gpu_status_after['gpu_util']}% | Memory {gpu_status_after['mem_used']}MB/{gpu_status_after['mem_total']}MB ({gpu_status_after['mem_util']}%) | Temp {gpu_status_after['temperature']}°C")

        # Move back to CPU and return
        return cp.asnumpy(close_image).astype('uint8')
    else:
        log_message("[INFO] Using CPU (NumPy) for spatial filtering")
        # 2) Binariza e faz opening fixo 2×2
        binary_image = output_image_data > 0

        # 3) Decide o opening
        if ofs is False:
            log_message("[INFO] Skipping opening filter step as requested.")
            open_image = binary_image
        else:
          # define M
            try:
              m = int(ofs) if ofs is not None else 2
            except (ValueError, TypeError):
              log_message(f"[WARNING] Invalid opening filter size '{ofs}'; defaulting to 2×2.")
              m = 2

            log_message(f"[INFO] Applying opening filter with {m}x{m} structuring element.")
            open_image   = ndimage.binary_opening(binary_image, structure=np.ones((m, m)))

        # 4) Decide o closing
        if cfs is False:
            log_message("[INFO] Skipping closing filter step as requested.")
            close_image = open_image
        else:
            # define N
            try:
                n = int(cfs) if cfs is not None else 4
            except (ValueError, TypeError):
                log_message(f"[WARNING] Invalid closing filter size '{cfs}'; defaulting to 4×4.")
                n = 4

            log_message(f"[INFO] Applying closing filter with {n}×{n} structuring element.")
            close_image = ndimage.binary_closing(open_image, structure=np.ones((n, n)))

        # 4) Converte e retorna
        return close_image.astype('uint8')

# Function to convert a NumPy array back into a GeoTIFF raster
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    log_message(f"[INFO] Converting array to GeoTIFF raster: {output_image_name}")
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    
    # **Adicione opções de criação para compressão e altere o tipo de dados**
    options = [
        'COMPRESS=DEFLATE',
        'PREDICTOR=2',
        'TILED=YES',
        'BIGTIFF=YES'
    ]
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Byte, options=options)
    
    # **Certifique-se de que os dados sejam do tipo uint8**
    image_data_scene_uint8 = image_data_scene.astype('uint8')
    outDs.GetRasterBand(1).WriteArray(image_data_scene_uint8)
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None  # Release the output dataset from memory
    log_message(f"[INFO] Raster conversion completed and saved as: {output_image_name}")


# Function to check if there is a significant intersection between the geometry and the image
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    log_message(f"[INFO] Checking for significant intersection with minimum area of {min_intersection_area}")
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

def clip_image_by_grid(geom, image, output, buffer_distance_meters=100, max_attempts=5, retry_delay=5):
    import gc
    attempt = 0

    # Check GPU status before clipping
    gpu_status = get_gpu_status()
    if gpu_status:
        log_message(f"[GPU] Before clipping: GPU {gpu_status['gpu_util']}% | Memory {gpu_status['mem_used']}MB/{gpu_status['mem_total']}MB ({gpu_status['mem_util']}%) | Temp {gpu_status['temperature']}°C")

    while attempt < max_attempts:
        try:
            log_message(f"[INFO] Attempt {attempt+1}/{max_attempts} to clip image: {image}")
            gpu_logger.info(f"Clipping image: {image}")

            with rasterio.open(image) as src:
                # Obter o CRS da imagem
                image_crs = src.crs

                # Reprojetar a geometria para o CRS da imagem
                geom_shape = shape(geom)
                geom_proj = reproject_geometry(geom_shape, 'EPSG:4326', image_crs)

                # Aplicar o buffer em metros
                expanded_geom = geom_proj.buffer(buffer_distance_meters)

                # Converter de volta para GeoJSON
                expanded_geom_geojson = mapping(expanded_geom)

                # Verificar a interseção significativa
                if has_significant_intersection(expanded_geom_geojson, src.bounds):
                    out_image, out_transform = mask(src, [expanded_geom_geojson], crop=True, nodata=np.nan, filled=True)

                    # Atualizar metadados
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "crs": src.crs
                    })

                    with rasterio.open(output, 'w', **out_meta) as dest:
                        dest.write(out_image)

                    # Free memory immediately
                    del out_image, out_transform, out_meta
                    gc.collect()

                    log_message(f"[INFO] Image clipped successfully: {output}")
                    gpu_logger.info(f"Clipping successful, memory freed")

                    # Check GPU status after successful clipping
                    gpu_status_after = get_gpu_status()
                    if gpu_status_after:
                        log_message(f"[GPU] After clipping: GPU {gpu_status_after['gpu_util']}% | Memory {gpu_status_after['mem_used']}MB/{gpu_status_after['mem_total']}MB ({gpu_status_after['mem_util']}%) | Temp {gpu_status_after['temperature']}°C")

                    return True  # Clipping successful
                else:
                    log_message(f"[INFO] Insufficient overlap for clipping: {image}")
                    return False  # No significant intersection, no need to retry
        except Exception as e:
            log_message(f"[ERROR] Error during clipping: {str(e)}. Retrying in {retry_delay} seconds...")
            gpu_logger.error(f"Clipping error: {str(e)}")
            gc.collect()  # Force garbage collection on error
            time.sleep(retry_delay)
            attempt += 1

    log_message(f"[ERROR] Failed to clip image after {max_attempts} attempts: {image}")
    return False  # Clipping failed after all attempts

def reproject_geometry(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)

# Function to build a VRT and translate using gdal_translate
def build_vrt(vrt_path, input_tif_list):
    if isinstance(input_tif_list, str):
        input_tif_list = input_tif_list.split()

    missing_files = [f for f in input_tif_list if not os.path.exists(f)]
    if missing_files:
        raise RuntimeError(f"The following input files do not exist: {missing_files}")

    if os.path.exists(vrt_path):
        log_message(f"[INFO] VRT already exists. Removing: {vrt_path}")
        os.remove(vrt_path)

    vrt = gdal.BuildVRT(vrt_path, input_tif_list)
    if vrt is None:
        raise RuntimeError(f"Failed to create VRT at {vrt_path}")
    vrt = None  # close

def translate_to_tiff(vrt_path, output_path):
    if os.path.exists(output_path):
        log_message(f"[INFO] TIFF already exists. Removing: {output_path}")
        os.remove(output_path)

    options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "COPY_SRC_OVERVIEWS=YES",
            "BIGTIFF=YES"
        ],
        noData=0
    )
    result = gdal.Translate(output_path, vrt_path, options=options)
    if result is None:
        raise RuntimeError(f"Failed to translate VRT to TIFF: {output_path}")
    result = None  # close

def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_list, suffix=""):
    try:
        name_out_vrt_suffixed = name_out_vrt.replace(".tif", f"{suffix}.vrt") if suffix else name_out_vrt.replace(".tif", ".vrt")
        name_out_tif_suffixed = name_out_tif.replace(".tif", f"{suffix}.tif") if suffix else name_out_tif

        log_message(f"[INFO] Building VRT from: {files_tif_list}")
        build_vrt(name_out_vrt_suffixed, files_tif_list)
        log_message(f"[INFO] VRT created: {name_out_vrt_suffixed}")

        log_message(f"[INFO] Translating VRT to optimized TIFF: {name_out_tif_suffixed}")
        translate_to_tiff(name_out_vrt_suffixed, name_out_tif_suffixed)
        log_message(f"[INFO] Optimized TIFF saved: {name_out_tif_suffixed}")

    except Exception as e:
        log_message(f"[ERROR] Failed to generate optimized image. {e}")
        return False

    if not os.path.exists(name_out_tif_suffixed):
        log_message(f"[ERROR] Output image not found locally after generation: {name_out_tif_suffixed}")
        return False

    return True


# Function to clean directories before processing begins
def clean_directories(directories_to_clean):
    """
    Cleans specified directories by removing all contents and recreating the directory.

    Args:
    - directories_to_clean: List of directories to clean.
    """
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
            log_message(f"[INFO] Cleaned and recreated directory: {directory}")
        else:
            os.makedirs(directory)
            log_message(f"[INFO] Created directory: {directory}")

# Function to check or create a GEE collection and make it public
def check_or_create_collection(collection,ee_project):
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        print(f'[INFO] Criando nova coleção no GEE: {collection}')
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        os.system(create_command)
    else:
        print(f'[INFO] Coleção já existe: {collection}')

# Função para realizar o upload de um arquivo para o GEE com metadados e verificar se o asset já existe
def upload_to_gee(gcs_path, asset_id, satellite, region, year, version):
    timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.now().strftime('%Y-%m-%d')

    # Check if the asset exists in GEE and remove it if so
    try:
        asset_info = ee.data.getAsset(asset_id)
        log_message(f"[INFO] Asset already exists. Deleting: {asset_id}")
        ee.data.deleteAsset(asset_id)
        time.sleep(2)
    except ee.EEException:
        log_message(f"[INFO] Asset does not exist yet. Proceeding with upload: {asset_id}")

    # Perform the upload using Earth Engine CLI
    upload_command = (
        f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
        f'--pyramiding_policy=mode '
        f'--property satellite={satellite} '
        f'--property region={region} '
        f'--property year={year} '
        f'--property version={version} '
        f'--property source=IPAM '
        f'--property type=annual_burned_area '
        f'--property time_start={timestamp_start} '
        f'--property time_end={timestamp_end} '
        f'--property create_date={creation_date} '
        f'{gcs_path}'
    )

    log_message(f"[INFO] Starting upload to GEE: {asset_id}")
    status = os.system(upload_command)

    if status == 0:
        log_message(f"[INFO] Upload completed successfully: {asset_id}")
    else:
        log_message(f"[ERROR] Upload failed for GEE asset: {asset_id}")
        log_message(f"[ERROR] Command status code: {status}")


# Function to remove temporary files
def remove_temporary_files(files_to_remove):
    """
    Removes temporary files from the system.

    Args:
    - files_to_remove: List of file paths to remove.
    """
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                log_message(f"[INFO] Temporary file removed: {file}")
            except Exception as e:
                log_message(f"[ERROR] Failed to remove file: {file}. Details: {str(e)}")

def fully_connected_layer(input, n_neurons, activation=None):
    """
    Creates a fully connected layer.

    :param input: Input tensor from the previous layer
    :param n_neurons: Number of neurons in this layer
    :param activation: Activation function ('relu' or None)
    :return: Layer output with or without activation applied
    """
    input_size = input.get_shape().as_list()[1]  # Get input size (number of features)

    # Initialize weights (W) with a truncated normal distribution and initialize biases (b) with zeros
    W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))), name='W')
    b = tf.Variable(tf.zeros([n_neurons]), name='b')

    # Apply the linear transformation (Wx + b)
    layer = tf.matmul(input, W) + b

    # Apply activation function, if specified
    if activation == 'relu':
        layer = tf.nn.relu(layer)

    return layer


# O resto do código estilo TensorFlow 1.x
def create_model_graph(hyperparameters):
    """
    Cria e retorna um grafo computacional TensorFlow dinamicamente com base nos parâmetros do modelo.
    """
    gpu_logger.info("Creating TensorFlow model graph")
    gpu_logger.info(f"Hyperparameters: {hyperparameters}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    gpu_logger.info(f"Available GPUs: {gpus}")

    graph = tf.Graph()

    with graph.as_default():
        gpu_logger.info("Forcing operations to GPU:0 with tf.device('/GPU:0')")
        # Force all operations to execute on GPU
        with tf.device('/GPU:0'):
            # Define placeholders para dados de entrada e rótulos
            x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
            y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

            # Normaliza os dados de entrada
            normalized = (x_input - hyperparameters['data_mean']) / hyperparameters['data_std']

            # Constrói as camadas da rede neural com os hiperparâmetros definidos
            hidden1 = fully_connected_layer(normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu')
            hidden2 = fully_connected_layer(hidden1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu')
            hidden3 = fully_connected_layer(hidden2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu')
            hidden4 = fully_connected_layer(hidden3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu')
            hidden5 = fully_connected_layer(hidden4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu')

            # Camada final de saída
            logits = fully_connected_layer(hidden5, n_neurons=hyperparameters['NUM_CLASSES'])

            # Define a função de perda (para treinamento, embora não seja necessária na inferência)
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
                name='cross_entropy_loss'
            )

            # Define o otimizador (para treinamento, embora não seja necessária na inferência)
            # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
            # Define the optimizer: Adam with the specified learning rate
            optimizer = tf.train.AdamOptimizer(hyperparameters['lr']).minimize(cross_entropy)

            # Operação para obter a classe prevista
            outputs = tf.argmax(logits, 1, name='predicted_class')

        # Inicializa todas as variáveis
        init = tf.global_variables_initializer()
        # Definir o saver para salvar ou restaurar o estado do modelo
        saver = tf.train.Saver()

    return graph, {'x_input': x_input, 'y_input': y_input}, saver

# Function to get GPU status
def get_gpu_status():
    """Get current GPU status using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 5:
                return {
                    'gpu_util': int(parts[0]),
                    'mem_util': int(parts[1]),
                    'mem_used': int(parts[2]),
                    'mem_total': int(parts[3]),
                    'temperature': int(parts[4])
                }
    except:
        pass
    return None

# Function to classify data using a TensorFlow model in blocks and handle memory manually
def classify(data_classify_vector, model_path, hyperparameters, block_size=4000000):
    """
    Classifies data in blocks using a TensorFlow model.

    ✅ CORRECTED VERSION: Uses ONE session for ALL blocks (not one session per block).
    This eliminates memory overhead and speeds up classification 5-10x.

    Args:
    - data_classify_vector: The input data (pixels) to classify.
    - model_path: Path to the TensorFlow model to be restored.
    - hyperparameters: Hyperparameters to create the model graph.
    - block_size: Number of pixels to process per block (default is 4,000,000).

    Returns:
    - output_data_classify: Classified data.
    """
    log_message(f"[INFO] Starting classification with model at path: {model_path}")

    # Check initial GPU status
    initial_gpu = get_gpu_status()
    if initial_gpu:
        log_message(f"[GPU] Initial Status: GPU {initial_gpu['gpu_util']}% | Memory {initial_gpu['mem_used']}MB/{initial_gpu['mem_total']}MB ({initial_gpu['mem_util']}%) | Temp {initial_gpu['temperature']}°C")

    # Number of pixels in the input data
    num_pixels = data_classify_vector.shape[0]
    num_blocks = (num_pixels + block_size - 1) // block_size  # Calculate the number of blocks

    log_message(f"[INFO] Total pixels: {num_pixels}, Block size: {block_size}, Number of blocks: {num_blocks}")

    # ✅ CORRECTION 1: Create graph ONCE (outside the loop)
    tf.compat.v1.reset_default_graph()
    graph, placeholders, saver = create_model_graph(hyperparameters)

    # ✅ CORRECTION 3: Reduce GPU memory fraction to 0.5, remove allow_growth
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5
    )

    # ✅ CORRECTION 5: Remove log_device_placement
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True
    )

    gpu_logger.info(f"Processing {num_blocks} blocks")
    gpu_logger.info(f"GPU memory fraction: 0.5 (CORRECTED from 0.8)")
    gpu_logger.info(f"Block size: {block_size} pixels (CORRECTED from 40M)")

    output_blocks = []

    # ✅ CORRECTION 1: Session OUTSIDE the loop (ONE session for ALL blocks)
    with tf.Session(graph=graph, config=config) as sess:
        gpu_logger.info("TensorFlow session created (ONCE for all blocks)")
        gpu_logger.info(f"Restoring model from: {model_path}")

        # ✅ CORRECTION 1: Load model ONCE
        saver.restore(sess, model_path)
        gpu_logger.info("Model restored successfully (ONCE)")

        # ✅ CORRECTION 1: Loop processes blocks INSIDE the same session
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, num_pixels)

            # Get GPU status for this block
            gpu_status = get_gpu_status()
            if gpu_status:
                log_message(f"[INFO] Block {i+1}/{num_blocks} (pixels {start_idx}-{end_idx}) | GPU: {gpu_status['gpu_util']}% | Memory: {gpu_status['mem_used']}MB/{gpu_status['mem_total']}MB ({gpu_status['mem_util']}%) | Temp: {gpu_status['temperature']}°C")
            else:
                log_message(f"[INFO] Processing block {i+1}/{num_blocks} (pixels {start_idx} to {end_idx})")

            # ✅ CORRECTION 4: Extract only necessary bands (if defined in hyperparameters)
            bi = hyperparameters.get('band_indices', None)
            if bi is not None:
                data_block = data_classify_vector[start_idx:end_idx, bi]
            else:
                data_block = data_classify_vector[start_idx:end_idx]

            gpu_logger.info(f"Running inference on block {i+1}/{num_blocks} ({len(data_block)} pixels)")

            # Classify the current block (session already initialized, model already loaded)
            output_block = sess.run(
                graph.get_tensor_by_name('predicted_class:0'),
                feed_dict={placeholders['x_input']: data_block}
            )

            output_blocks.append(output_block)
            gpu_logger.info(f"Block {i+1}/{num_blocks} completed")

    # Session closes automatically when exiting 'with'

    # ✅ CORRECTION 1: Explicit cleanup at the end
    tf.keras.backend.clear_session()
    gpu_logger.info("TensorFlow session closed and cleaned")

    # Concatenate the classified blocks into a single array
    output_data_classify = np.concatenate(output_blocks, axis=0)

    # Final GPU status
    final_gpu = get_gpu_status()
    if final_gpu:
        log_message(f"[GPU] Final Status: GPU {final_gpu['gpu_util']}% | Memory {final_gpu['mem_used']}MB/{final_gpu['mem_total']}MB ({final_gpu['mem_util']}%) | Temp {final_gpu['temperature']}°C")

    log_message(f"[INFO] Classification completed")

    return output_data_classify

def process_single_image(dataset_classify, version, region,folder_temp):
    """
    Processes a single image by applying the classification model and spatial filtering to generate the final result.

    Args:
    - dataset_classify: GDAL dataset of the image to be classified.
    - num_classes: Number of classes in the model.
    - data_mean: Mean of the data for normalization.
    - data_std: Standard deviation of the data for normalization.
    - version: Version of the model.
    - region: Target region for classification.

    Returns:
    - Filtered classified image.
    """
    gpu_logger.info("="*80)
    gpu_logger.info("PROCESS_SINGLE_IMAGE STARTED")
    gpu_logger.info(f"Version: {version}, Region: {region}")
    gpu_logger.info("="*80)

    # Path to the remote model in Google Cloud Storage (with wildcards)
    gcs_model_file = f'gs://{bucket_name}/sudamerica/{country}/models_col1/col1_{country}_{version}_{region}_rnn_lstm_ckpt*'
    # Local path for the model files
    model_file_local_temp = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt'

    log_message(f"[INFO] Downloading TensorFlow model from GCS {gcs_model_file} to {folder_temp}.")
    gpu_logger.info(f"Downloading model from GCS: {gcs_model_file}")
    
    # Command to download the model files from GCS
    try:
        gpu_logger.info("Executing gsutil cp command...")
        subprocess.run(f'gsutil cp {gcs_model_file} {folder_temp}', shell=True, check=True)
        time.sleep(2)
        fs.invalidate_cache()
        log_message(f"[INFO] Model downloaded successfully.")
        gpu_logger.info("Model files downloaded successfully")
    except subprocess.CalledProcessError as e:
        log_message(f"[ERROR] Failed to download model from GCS: {e}")
        gpu_logger.error(f"Failed to download model: {e}")
        return None

    # Path to the JSON file containing hyperparameters
    json_path = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt_hyperparameters.json'
    gpu_logger.info(f"Loading hyperparameters from: {json_path}")

    # Load hyperparameters from the JSON file
    with open(json_path, 'r') as json_file:
        hyperparameters = json.load(json_file)

    gpu_logger.info(f"Hyperparameters loaded: {hyperparameters}")

    # Retrieve hyperparameter values from the JSON file
    DATA_MEAN = np.array(hyperparameters['data_mean'])
    DATA_STD = np.array(hyperparameters['data_std'])
    NUM_N_L1 = hyperparameters['NUM_N_L1']
    NUM_N_L2 = hyperparameters['NUM_N_L2']
    NUM_N_L3 = hyperparameters['NUM_N_L3']
    NUM_N_L4 = hyperparameters['NUM_N_L4']
    NUM_N_L5 = hyperparameters['NUM_N_L5']
    NUM_CLASSES = hyperparameters['NUM_CLASSES']
    NUM_INPUT = hyperparameters['NUM_INPUT']

    log_message(f"[INFO] Loaded hyperparameters: DATA_MEAN={DATA_MEAN}, DATA_STD={DATA_STD}, NUM_N_L1={NUM_N_L1}, NUM_N_L2={NUM_N_L2}, NUM_N_L3={NUM_N_L3}, NUM_N_L4={NUM_N_L4}, NUM_N_L5={NUM_N_L5}, NUM_CLASSES={NUM_CLASSES}")

    # Convert GDAL dataset to a NumPy array
    log_message(f"[INFO] Converting GDAL dataset to NumPy array.")
    gpu_logger.info("Converting GDAL dataset to NumPy array")
    data_classify = convert_to_array(dataset_classify)
    gpu_logger.info(f"Array shape: {data_classify.shape}")

    # Reshape into a single pixel vector
    log_message(f"[INFO] Reshaping data into a single pixel vector.")
    gpu_logger.info("Reshaping data into single pixel vector")
    data_classify_vector = reshape_single_vector(data_classify)
    gpu_logger.info(f"Vector shape: {data_classify_vector.shape}")
    # print('data_classify_vector',data_classify_vector)
    # Normalize the input vector using data_mean and data_std
    # log_message(f"[INFO] Normalizing the input vector using data_mean and data_std.")
    # data_classify_vector = (data_classify_vector - DATA_MEAN) / DATA_STD

    # Transfer from GPU to CPU if necessary (TensorFlow needs NumPy arrays)
    if GPU_AVAILABLE and isinstance(data_classify_vector, cp.ndarray):
        # Check GPU status before transfer
        gpu_status_before = get_gpu_status()
        if gpu_status_before:
            log_message(f"[GPU] Before GPU->CPU transfer: GPU {gpu_status_before['gpu_util']}% | Memory {gpu_status_before['mem_used']}MB/{gpu_status_before['mem_total']}MB ({gpu_status_before['mem_util']}%) | Temp {gpu_status_before['temperature']}°C")

        gpu_logger.info("Transferring data from GPU to CPU for TensorFlow classification")
        gpu_logger.info(f"GPU array size: {data_classify_vector.nbytes / (1024**2):.2f} MB")
        log_message(f"[INFO] Transferring {data_classify_vector.nbytes / (1024**2):.2f} MB from GPU to CPU")

        data_classify_vector_cpu = cp.asnumpy(data_classify_vector)

        # Free GPU memory after transfer
        del data_classify_vector
        cp.get_default_memory_pool().free_all_blocks()
        gpu_logger.info("GPU memory freed after transfer to CPU")

        # Check GPU status after transfer
        gpu_status_after = get_gpu_status()
        if gpu_status_after:
            log_message(f"[GPU] After GPU->CPU transfer: GPU {gpu_status_after['gpu_util']}% | Memory {gpu_status_after['mem_used']}MB/{gpu_status_after['mem_total']}MB ({gpu_status_after['mem_util']}%) | Temp {gpu_status_after['temperature']}°C")

        data_classify_vector = data_classify_vector_cpu
    else:
        gpu_logger.info("Data already on CPU (NumPy array)")

    # Perform the classification using the model
    log_message(f"[INFO] Running classification using the model.")
    gpu_logger.info("="*80)
    gpu_logger.info("CALLING CLASSIFY() FUNCTION")
    gpu_logger.info(f"Model path: {model_file_local_temp}")
    gpu_logger.info(f"Input data shape: {data_classify_vector.shape}")
    gpu_logger.info(f"Input data type: {type(data_classify_vector)}")
    gpu_logger.info("="*80)

    output_data_classified = classify(data_classify_vector, model_file_local_temp, hyperparameters)

    gpu_logger.info("Classification completed successfully")

    # Reshape the classified data back into image format
    log_message(f"[INFO] Reshaping classified data back into image format.")
    output_image_data = reshape_image_output(output_data_classified, data_classify)

    # Apply spatial filtering
    log_message(f"[INFO] Applying spatial filtering and completing the processing of this scene.")
    gpu_logger.info("Applying spatial filtering")
    result = filter_spatial(output_image_data)
    gpu_logger.info("PROCESS_SINGLE_IMAGE COMPLETED")
    return result

def process_year_by_satellite(satellite_years, bucket_name, folder_mosaic, folder_temp, suffix,
                              ee_project, country, version, region, simulate_test=False):

    log_message(f"[INFO] Processing year by satellite for country: {country}, version: {version}, region: {region}")
    grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}')
    grid_landsat = grid.getInfo()['features']
    start_time = time.time()

    collection_name = f'projects/{ee_project}/assets/FIRE/COLLECTION1/CLASSIFICATION/burned_area_{country}_{version}'
    check_or_create_collection(collection_name, ee_project)

    for satellite_year in satellite_years[:1 if simulate_test else None]:  # apenas 1 satélite se teste
        satellite = satellite_year['satellite']
        years = satellite_year['years'][:1 if simulate_test else None]     # apenas 1 ano se teste

        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                test_tag = "_test" if simulate_test else ""
                image_name = f"burned_area_{country}_{satellite}_{version}_region{region[1:]}_{year}{suffix}{test_tag}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_classified/{image_name}.tif'

                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif'

                if not os.path.exists(local_cog_path):
                    log_message(f"[INFO] Downloading COG from GCS: {gcs_cog_path}")
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')
                    time.sleep(2)
                    fs.invalidate_cache()

                input_scenes = []
                grids_to_process = [grid_landsat[0]] if simulate_test else grid_landsat

                with tqdm(total=len(grids_to_process), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid in grids_to_process:
                        orbit = grid['properties']['ORBITA']
                        point = grid['properties']['PONTO']
                        output_image_name = f'{folder_temp}/image_col3_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'
                        geometry_scene = grid['geometry']
                        NBR_clipped = f'{folder_temp}/image_mosaic_col3_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'

                        if os.path.isfile(output_image_name):
                            log_message(f"[INFO] Scene {orbit}/{point} already processed. Skipping.")
                            pbar_scenes.update(1)
                            continue

                        clipping_success = clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)

                        if clipping_success:
                            dataset_classify = load_image(NBR_clipped)
                            image_data = process_single_image(dataset_classify, version, region, folder_temp)
                            convert_to_raster(dataset_classify, image_data, output_image_name)
                            input_scenes.append(output_image_name)
                            remove_temporary_files([NBR_clipped])
                        else:
                            log_message(f"[WARNING] Clipping failed for scene {orbit}/{point}.")
                        pbar_scenes.update(1)

                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_temp}/merged_temp_{year}.tif"
                    output_image = f"{folder_temp}/{image_name}.tif"

                    generate_optimized_image(merge_output_temp, output_image, input_scenes_str)

                    # ⏱ Aguardar criação do arquivo até 10s
                    wait_time = 0
                    while not os.path.exists(output_image) and wait_time < 10:
                        time.sleep(1)
                        wait_time += 1

                    if not os.path.exists(output_image):
                        log_message(f"[ERROR] Output image not found locally after wait. Skipping upload: {output_image}")
                        continue

                    size_mb = os.path.getsize(output_image) / (1024 * 1024)
                    if size_mb < 0.01:
                        log_message(f"[ERROR] Output image too small ({size_mb:.2f} MB). Likely failed.")
                        continue

                    log_message(f"[INFO] Output image verified. Size: {size_mb:.2f} MB")

                    status_upload = os.system(f'gsutil cp {output_image} {gcs_filename}')
                    time.sleep(2)
                    fs.invalidate_cache()

                    if status_upload == 0:
                        log_message(f"[INFO] Upload to GCS succeeded: {gcs_filename}")
                        if os.system(f'gsutil ls {gcs_filename}') == 0:
                            upload_to_gee(
                                gcs_filename,
                                f'{collection_name}/{image_name}',
                                satellite,
                                region,
                                year,
                                version
                            )
                        else:
                            log_message(f"[ERROR] File not found on GCS after upload.")
                    else:
                        log_message(f"[ERROR] Upload to GCS failed with code {status_upload}")

                clean_directories([folder_temp])
                elapsed = time.time() - start_time
                log_message(f"[INFO] Year {year} processing completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                pbar_years.update(1)



# ====================================
# 🚀 MAIN EXECUTION LOGIC
# ====================================

def render_classify_models(models_to_classify, simulate_test=False):
    """
    Processes a list of models and mosaics to classify burned areas.
    Args:
    - models_to_classify: List of dictionaries containing models, mosaics, and a simulation flag.
    """
    log_message(f"[INFO] [render_classify_models] STARTING PROCESSINGS FOR CLASSIFY MODELS {models_to_classify}")
    # Define bucket name
    bucket_name = 'mapbiomas-fire'
    # Loop through each model
    for model_info in models_to_classify:
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info["simulation"]
        log_message(f"[INFO] Processing model: {model_name}")
        log_message(f"[INFO] Selected mosaics: {mosaics}")
        log_message(f"[INFO] Simulation mode: {simulation}")
        # Extract model information
        parts = model_name.split('_')
        country = parts[1]
        version = parts[2]
        region = parts[3]
        # Define directories
        folder = f'/content/mapbiomas-fire/sudamerica/{country}'
        folder_temp = f'{folder}/tmp1'
        folder_mosaic = f'{folder}/mosaics_cog'
        
        log_message(f"[INFO] Starting the classification process for country: {country}.")
        
        # Ensure necessary directories exist
        for directory in [folder_temp, folder_mosaic]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                # log_message(f"[INFO] Created directory: {directory}")
            else:
                log_message(f"[INFO] Directory already exists: {directory}")
        
        clean_directories([folder_temp, folder_mosaic])
        # Prepare satellite and year list based on mosaics
        satellite_years = []
        for mosaic in mosaics:
            mosaic_parts = mosaic.split('_')
            satellite = mosaic_parts[0]
            year = int(mosaic_parts[3])
            satellite_years.append({
                "satellite": satellite,
                "years": [year]
            })
        # If in simulation mode, just simulate the processing
        if simulation:
            log_message(f"[SIMULATION] Would process model: {model_name} with mosaics: {mosaics}")
        else:
            # Call the main processing function (this will process all years for the satellite)
            process_year_by_satellite(
                satellite_years=satellite_years,
                bucket_name=bucket_name,
                folder_mosaic=folder_mosaic,
                folder_temp=folder_temp,
                suffix='',
                ee_project=f'mapbiomas-{country}',
                country=country,
                version=version,
                region=region,
                simulate_test=simulate_test
            )
   
    log_message(f"[INFO] [render_classify_models] FINISH PROCESSINGS FOR CLASSIFY MODELS {models_to_classify}")
