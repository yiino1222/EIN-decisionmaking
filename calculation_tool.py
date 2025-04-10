import numpy as np
import scanpy as sc
import anndata

import time
import os, wget
import cudf

from cuml.decomposition import PCA
from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.preprocessing import StandardScaler

import rapids_scanpy_funcs
import utils

import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')
import pandas as pd
from sh import gunzip
import scipy
from scipy import sparse
import gc
import cupy as cp

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def load_parameters():
    D_R_mtx=pd.read_csv("/data/drug_receptor_mtx.csv",index_col=0)
    GPCR_type_df=pd.read_csv("/data/GPCR_df.csv",index_col=0)

    drug_list=D_R_mtx.index.to_list()
    GPCR_list=["HTR1A","HTR1B","HTR1D","HTR1E","HTR2A","HTR2B","HTR2C",
    "HTR3A","HTR4","HTR5A","HTR6","HTR7","DRD1","DRD2","DRD3","DRD4","DRD5",
    "HRH1","HRH2","HRH3","CHRM1","CHRM2","CHRM3","CHRM4","CHRM5",
    "ADRA1A","ADRA1B","ADRA2A","ADRA2B","ADRA2C","ADRB1","ADRB2"]
    D_R_mtx.columns=GPCR_list
    return D_R_mtx,GPCR_type_df,drug_list,GPCR_list

def set_parameters_for_preprocess(GPCR_list):
    params = {}  # Create an empty dictionary to store parameters
    # maximum number of cells to load from files
    params["USE_FIRST_N_CELLS"] = 300000
    
    # Set MITO_GENE_PREFIX
    params['MITO_GENE_PREFIX'] = "mt-"
    
    # Set markers
    markers = ["CX3CR1","CLDN5","GLUL","NDRG2","PCDH15","PLP1","MBP","SATB2","SLC17A7",
               "SLC17A6","GAD2","GAD1","SNAP25"]
    markers.extend(GPCR_list)
    params['markers'] = [str.upper() for str in markers]
    
    # Set cell filtering parameters
    params['min_genes_per_cell'] = 200
    params['max_genes_per_cell'] = 6000
    
    # Set gene filtering parameters
    params['min_cells_per_gene'] = 1
    params['n_top_genes'] = 4000
    
    # Set PCA parameters
    params['n_components'] = 50
    
    # Set Batched PCA parameters
    params['pca_train_ratio'] = 0.2
    params['n_pca_batches'] = 10
    
    # Set t-SNE parameters
    params['tsne_n_pcs'] = 20
    
    # Set k-means parameters
    params['k'] = 35
    
    # Set KNN parameters
    params['n_neighbors'] = 15
    params['knn_n_pcs'] = 50
    
    # Set UMAP parameters
    params['umap_min_dist'] = 0.3
    params['umap_spread'] = 1.0
    
    return params

def preprocess_adata_in_bulk(adata_path,label=None,add_markers=None,is_gpu=True):
    preprocess_start = time.time()
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    # Set parameters
    params = set_parameters_for_preprocess(GPCR_list)

    # Add any additional markers if provided
    if add_markers is not None:
        # Ensure the additional markers are in uppercase for consistency
        add_markers = [marker.upper() for marker in add_markers]
        # Append the additional markers to the markers list in the parameters
        params['markers'].extend(add_markers)
    
    #preprocess in bulk
    print("preprocess_in_bulk")
    adata = anndata.read_h5ad(adata_path)
    if label !=None:
        adata=adata[adata.obs["label"]==label]
    genes = cudf.Series(adata.var_names).str.upper()
    barcodes = cudf.Series(adata.obs_names)
    is_label=False
    # Initialize labels dataframe if "label" column exists in adata.obs
    if "label" in adata.obs.columns:
        is_label=True
        original_labels = adata.obs["label"].copy()
    #if len(adata.obs["label"])>0:
    #    is_label=True
    #    labels=cudf.DataFrame(adata.obs["label"])
    #    labels = cudf.DataFrame({"barcode": barcodes.reset_index(drop=True), "label": adata.obs["label"]})
        #labels= cudf.DataFrame(adata.obs['label'])
    sparse_gpu_array = cp.sparse.csr_matrix(adata.X)
    sparse_gpu_array,filtered_barcodes = rapids_scanpy_funcs.filter_cells(sparse_gpu_array, min_genes=params['min_genes_per_cell'],
                                                        max_genes=params['max_genes_per_cell'],barcodes=barcodes)
    sparse_gpu_array, genes = rapids_scanpy_funcs.filter_genes(sparse_gpu_array, genes, 
                                                            min_cells=params['min_cells_per_gene'])
    """sparse_gpu_array, genes, marker_genes_raw = \
    rapids_scanpy_funcs.preprocess_in_batches(adata_path, 
                                              params['markers'], 
                                              min_genes_per_cell=params['min_genes_per_cell'], 
                                              max_genes_per_cell=params['max_genes_per_cell'], 
                                              min_cells_per_gene=params['min_cells_per_gene'], 
                                              target_sum=1e4, 
                                              n_top_genes=params['n_top_genes'],
                                              max_cells=params["USE_FIRST_N_CELLS"])
    """
    markers=params['markers'].copy()
    df=genes.to_pandas()
    
    # Before loop: create a set of markers to remove
    markers_to_remove = set()

    # Inside the loop, just add to the set if the marker needs to be removed
    for marker in markers:
        if not marker in df.values:
            print(f"{marker} is not included")
            markers_to_remove.add(marker)
            print(f"{marker} is removed from marker list")

    # After loop: remove the markers that are not found
    for marker in markers_to_remove:
        markers.remove(marker)
    
    print(markers)            
    tmp_norm = sparse_gpu_array.tocsc()
    marker_genes_raw = {
        ("%s_raw" % marker): tmp_norm[:, genes[genes == marker].index[0]].todense().ravel()
        for marker in markers
    }

    del tmp_norm

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    ## Regress out confounding factors (number of counts, mitochondrial gene expression)
    # calculate the total counts and the percentage of mitochondrial counts for each cell
    mito_genes = genes.str.startswith(params['MITO_GENE_PREFIX'])
    n_counts = sparse_gpu_array.sum(axis=1)
    percent_mito = (sparse_gpu_array[:,mito_genes].sum(axis=1) / n_counts).ravel()
    n_counts = cp.array(n_counts).ravel()
    percent_mito = cp.array(percent_mito).ravel()
    
    # regression
    print("perform regression")
    
    sparse_gpu_array = rapids_scanpy_funcs.regress_out(sparse_gpu_array.tocsc(), n_counts, percent_mito)
    del n_counts, percent_mito, mito_genes
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    
    # scale
    print("perform scale")
    mean = sparse_gpu_array.mean(axis=0)
    sparse_gpu_array -= mean
    stddev = cp.sqrt(sparse_gpu_array.var(axis=0))
    sparse_gpu_array /= stddev
    print(sparse_gpu_array.dtype)
    sparse_gpu_array = sparse_gpu_array.clip(None,10)
    del mean, stddev
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    preprocess_time = time.time()
    print("Total Preprocessing time: %s" % (preprocess_time-preprocess_start))
    
    ## Cluster and visualize
    adata = anndata.AnnData(sparse_gpu_array.get())
    adata.var_names = genes.to_pandas()
    adata.obs_names = filtered_barcodes.to_pandas()
    print(f"shape of adata: {adata.X.shape}")
    
    # Restore labels after preprocessing
    if is_label:
        # Convert filtered_barcodes to a pandas Series
        filtered_barcodes_host = filtered_barcodes.to_pandas()  # <- 追加: データをホストに移動
        filtered_labels = original_labels.loc[filtered_barcodes_host].values
        adata.obs["label"] = filtered_labels
    
    del sparse_gpu_array, genes
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    print(f"shape of adata: {adata.X.shape}")
    
    GPCR_df=pd.DataFrame()
    for name, data in marker_genes_raw.items():
        adata.obs[name] = data.get()
        if   name[:-4] in GPCR_list:
            GPCR_df[name]=data.get()
        
    # Deminsionality reduction
    #We use PCA to reduce the dimensionality of the matrix to its top 50 principal components.
    #If the number of cells was smaller, we would use the command 
    # `adata.obsm["X_pca"] = cuml.decomposition.PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)` 
    # to perform PCA on all the cells.
    #However, we cannot perform PCA on the complete dataset using a single GPU. 
    # Therefore, we use the batched PCA function in `utils.py`, which uses only a fraction 
    # of the total cells to train PCA.
    print("perform PCA")
    print(params["n_pca_batches"])
    adata = utils.pca(adata, n_components=params["n_components"], 
                  train_ratio=params["pca_train_ratio"], 
                  n_batches=params["n_pca_batches"],
                  gpu=is_gpu)
    
    #t-sne + k-means
    print("t-sne")
    #adata=tsne_kmeans(adata,params['tsne_n_pcs'],params['k'])
    
    #UMAP + Graph clustering
    print("UMAP")
    adata=UMAP_adata(adata,params["n_neighbors"],params["knn_n_pcs"],
                     params["umap_min_dist"],params["umap_spread"])
   
    #calculate response to antipsychotics
    print("calc drug response")
    default_drug_conc=100
    adata=calc_drug_response(adata,GPCR_df,GPCR_type_df,drug_list,D_R_mtx,default_drug_conc)
    
    #calculate clz selectivity
    selectivity_threshold=1.2
    adata,num_clz_selective_cells=calc_clz_selective_cell(adata,drug_list,selectivity_threshold)
    
    #save preprocessed adata 
    file_root, file_extension = os.path.splitext(adata_path)
    # Append '_processed' to the root and add the extension back
    processed_file_path = f"{file_root}_processed{file_extension}"
    adata.write(processed_file_path)
  
    return adata,GPCR_df

def preprocess_adata_in_batch(adata_path,max_cells):
    import dask
    from dask_cuda import initialize, LocalCUDACluster
    from dask.distributed import Client, default_client
    import rmm
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    def set_mem():
        rmm.reinitialize(managed_memory=True)
        cp.cuda.set_allocator(rmm_cupy_allocator)

    preprocess_start = time.time()
    #Set `preprocessing_gpus` below to specify the GPUs to use for preprocessing. 
    # For example, numbers 0-7 can be used on a machine with 8 gpus. Specifying a specific number,
    #  such as 5, will use only the 6th GPU on the machine. In practice, 
    # it's often a good idea to use GPUs 1-7 for pre-processing and GPU0 for downstream clustering, 
    # visualization, and differential gene expression steps. 
    preprocessing_gpus="1, 2, 3"
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=preprocessing_gpus)
    client = Client(cluster)    

    set_mem()
    client.run(set_mem)
    client
    n_workers = len(client.scheduler_info()['workers'])
    #load parameters
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    #set parameters
    params=set_parameters_for_preprocess(GPCR_list)
    
    #preprocess in batch
    print("preprocess_in_batches")

    #Below, we load the sparse count matrix from the `.h5ad` file into GPU using a custom function. 
    # While reading the dataset, filters are applied on the count matrix to remove cells with 
    # an extreme number of genes expressed. Genes will zero expression in all cells are also eliminated. 

    #The custom function uses [Dask](https://dask.org) to partition data. 
    # The above mentioned filters are applied on individual partitions. 
    # Usage of Dask along with cupy provides the following benefits:
    #- Parallelized data loading when multiple GPUs are available
    #- Ability to partition the data allows pre-processing large datasets

    #Filters are applied on individual batches of cells. 
    # Elementwise or cell-level normalization operations are also performed while reading. 
    # For this example, the following two operations are performed:
    #- Normalize the count matrix so that the total counts in each cell sum to 1e4.
    #- Log transform the count matrix.

    def partial_post_processor(partial_data):
        partial_data = rapids_scanpy_funcs.normalize_total(partial_data, target_sum=1e4)
        return partial_data.log1p()

    dask_sparse_arr, genes, query = rapids_scanpy_funcs.read_with_filter(client,
                                                        adata_path,
                                                        min_genes_per_cell=params['min_genes_per_cell'],
                                                        max_genes_per_cell=params['max_genes_per_cell'],
                                                        partial_post_processor=partial_post_processor)
    dask_sparse_arr = dask_sparse_arr.persist()
    dask_sparse_arr.shape

    # Select Most Variable Genes
    markers=params['markers']
    marker_genes_raw = {}
    i = 0
    for index in genes[genes.isin(markers)].index.to_arrow().to_pylist():
        marker_genes_raw[markers[i]] = dask_sparse_arr[:, index].compute().toarray().ravel()
        i += 1

    #Filter the count matrix to retain only the most variable genes.
    hvg = rapids_scanpy_funcs.highly_variable_genes_filter(client, dask_sparse_arr, genes, n_top_genes=n_top_genes)

    genes = genes[hvg]
    dask_sparse_arr = dask_sparse_arr[:, hvg]
    sparse_gpu_array = dask_sparse_arr.compute()

    del hvg

    print("marker_genes_raw")
    print(marker_genes_raw)
    ## Regress out confounding factors (number of counts, mitochondrial gene expression)
    # calculate the total counts and the percentage of mitochondrial counts for each cell
    mito_genes = genes.str.startswith(params['MITO_GENE_PREFIX']).values

    n_counts = sparse_gpu_array.sum(axis=1)
    percent_mito = (sparse_gpu_array[:,mito_genes].sum(axis=1) / n_counts).ravel()

    n_counts = cp.array(n_counts).ravel()
    percent_mito = cp.array(percent_mito).ravel()
    
    # regression
    print("perform regression")
    n_rows = dask_sparse_arr.shape[0]
    n_cols = dask_sparse_arr.shape[1]
    cols_per_worker = int(n_cols / n_workers)
    dask_sparse_arr = dask_sparse_arr.map_blocks(lambda x: x.todense(), dtype="float32", meta=cp.array(cp.zeros((0,)))).T
    dask_sparse_arr = dask_sparse_arr.rechunk((cols_per_worker, n_rows)).persist()
    dask_sparse_arr.compute_chunk_sizes()

    import math
    dask_sparse_arr = dask_sparse_arr.map_blocks(lambda x: rapids_scanpy_funcs.regress_out(x.T, n_counts, percent_mito).T, dtype="float32", meta=cp.array(cp.zeros(0,))).T
    dask_sparse_arr = dask_sparse_arr.rechunk((math.ceil(n_rows/n_workers), n_cols)).persist()
    dask_sparse_arr.compute_chunk_sizes()

    # scale
    print("perform scale")
    mean = dask_sparse_arr.mean(axis=0)
    dask_sparse_arr -= mean
    stddev = cp.sqrt(dask_sparse_arr.var(axis=0).compute())
    dask_sparse_arr /= stddev
    dask_sparse_arr = dask.array.clip(dask_sparse_arr, -10, 10).persist()
    del mean, stddev
    
    preprocess_time = time.time()
    print("Total Preprocessing time: %s" % (preprocess_time-preprocess_start))
    
    ## Cluster and visualize
    adata = anndata.AnnData(sparse_gpu_array.get())
    adata.var_names = genes.to_pandas()
    del sparse_gpu_array, genes
    print(f"shape of adata: {adata.X.shape}")
    
    GPCR_df=pd.DataFrame()
    for name, data in marker_genes_raw.items():
        print(len(adata.obs[name]))
        print(len(data.get()))
        adata.obs[name] = data.get()
        if   name[:-4] in GPCR_list:
            GPCR_df[name]=data.get()
        
    # Deminsionality reduction
    #We use PCA to reduce the dimensionality of the matrix to its top 50 principal components.
    from cuml.dask.decomposition import PCA
    pca_data = PCA(n_components=50).fit_transform(dask_sparse_arr)
    pca_data.compute_chunk_sizes()

    #We store the preprocessed count matrix as an AnnData object, 
    # which is currently in host memory. We also add the expression levels of 
    # the marker genes as observations to the annData object.
    local_pca = pca_data.compute()
    X = dask_sparse_arr.compute().get()

    adata = anndata.AnnData(X=X)
    adata.var_names = genes.to_numpy()
    adata.obsm["X_pca"] = local_pca.get()

    del pca_data
    del dask_sparse_arr
    
    #UMAP + Graph clustering
    print("UMAP")
    adata=UMAP_adata(adata,params["n_neighbors"],params["knn_n_pcs"],
                     params["umap_min_dist"],params["umap_spread"])
   
    #calculate response to antipsychotics
    print("calc drug response")
    default_drug_conc=100
    adata=calc_drug_response(adata,GPCR_df,GPCR_type_df,drug_list,D_R_mtx,default_drug_conc)
    
    #calculate clz selectivity
    selectivity_threshold=1.2
    adata,num_clz_selective_cells=calc_clz_selective_cell(adata,drug_list,selectivity_threshold)
    
    #save preprocessed adata 
    file_root, file_extension = os.path.splitext(adata_path)
    # Append '_processed' to the root and add the extension back
    processed_file_path = f"{file_root}_processed{file_extension}"
    adata.write(processed_file_path)
    client.shutdown()
    cluster.close()
    
    return adata,GPCR_df

def tsne_kmeans(adata,tsne_n_pcs,k):
    adata.obsm['X_tsne'] = TSNE().fit_transform(adata.obsm["X_pca"][:,:tsne_n_pcs])
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str) 
    print("t-sne + k-means")       
    sc.pl.tsne(adata, color=["kmeans"])
    return adata

def UMAP_adata(adata,n_neighbors,knn_n_pcs,umap_min_dist,umap_spread):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=knn_n_pcs,
                    method='rapids')
    sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread,
               method='rapids')
    sc.tl.louvain(adata, flavor='rapids')
    print("UMAP louvain")
    sc.pl.umap(adata, color=["louvain"])
    adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata)
    print("UMAP leiden")
    sc.pl.umap(adata, color=["leiden"])
    return adata

def calc_drug_response(adata,GPCR_df,GPCR_type_df,drug_list,D_R_mtx,drug_conc):
    #normalize GPCR expression levels
    GPCR_adata=anndata.AnnData(X=GPCR_df)
    GPCR_adata_norm=sc.pp.normalize_total(GPCR_adata,target_sum=1e4,inplace=False)['X']
    GPCR_adata_norm_df=pd.DataFrame(GPCR_adata_norm,columns=GPCR_adata.var.index)
    norm_df=pd.DataFrame(GPCR_adata_norm)
    norm_col=[str[:-4] for str in GPCR_df.columns]
    norm_df.columns=norm_col
    
    GPCR_type_df=GPCR_type_df[GPCR_type_df.receptor_name.isin(norm_col)]
    
    Gs=GPCR_type_df[GPCR_type_df.type=="Gs"]["receptor_name"].values
    Gi=GPCR_type_df[GPCR_type_df.type=="Gi"]["receptor_name"].values
    #Gq=GPCR_type_df[GPCR_type_df.type=="Gq"]["receptor_name"].values
    
    cAMP_df=pd.DataFrame(columns=drug_list)
    #Ca_df=pd.DataFrame(columns=drug_list)
    for drug in drug_list:
        Gs_effect=(norm_df.loc[:,Gs]/(1+drug_conc/D_R_mtx.loc[drug,Gs])).sum(axis=1) #TODO ki値で割り算するときにlog換算すべきか
        Gi_effect=(norm_df.loc[:,Gi]/(1+drug_conc/D_R_mtx.loc[drug,Gi])).sum(axis=1)
        basal_cAMP=(norm_df.loc[:,Gs]-norm_df.loc[:,Gi]).sum(axis=1)
        #Gq_effect=(norm_df.loc[:,Gq]/D_R_mtx.loc[drug,Gq]).sum(axis=1)
        cAMPmod=(Gs_effect-Gi_effect)-basal_cAMP #Giの阻害→cAMP上昇、Gsの阻害→cAMP低下
        cAMP_df[drug]=cAMPmod
    cAMP_df.index=adata.obs_names
    #Ca_df.index=adata.obs_names
    #Ca_df=Ca_df+10**(-4)
    for drug in drug_list:
        adata.obs['cAMP_%s'%drug]=cAMP_df[drug]
        #adata.obs['Ca_%s'%drug]=Ca_df[drug]
        
    return adata

def calc_clz_selective_cell(adata,drug_list,selectivity_threshold):
    adata.obs["is_clz_activated"]=np.zeros(len(adata.obs))
    adata.obs["is_clz_activated"][adata.obs["cAMP_CLOZAPINE"]>10]=1
    adata.obs["is_clz_activated"]=adata.obs["is_clz_activated"].astype("category")
    
    adata.obs["is_clz_inhibited"]=np.zeros(len(adata.obs))
    adata.obs["is_clz_inhibited"][adata.obs["cAMP_CLOZAPINE"]<-10]=1
    adata.obs["is_clz_inhibited"]=adata.obs["is_clz_inhibited"].astype("category")

    # 「CLOZAPINE」以外の薬に対応するカラム名のリストを作成
    drug_cols = [f"cAMP_{drug}" for drug in drug_list if drug != "CLOZAPINE"]

    # メモリ使用量削減のため、必要に応じて計算対象カラムを float32 にキャスト
    for col in drug_cols + ["cAMP_CLOZAPINE"]:
        adata.obs[col] = adata.obs[col].astype("float32")


    # ゼロ除算を避けるための小さな定数
    epsilon = 1e-9

    # 薬ごとの cAMP 値の平均（ゼロ除算防止のため epsilon を加算）
    adata.obs["cAMP_mean_other_than_czp"] = adata.obs[drug_cols].mean(axis=1) + epsilon


    # クロザピンに対するセレクティビティの計算（各細胞ごとにベクトル演算）
    adata.obs["cAMP_clz_selectivity"] = (adata.obs["cAMP_CLOZAPINE"] ** 2) / (adata.obs["cAMP_mean_other_than_czp"] ** 2)

    # selectivity_threshold と cAMP_CLOZAPINE > 0 の条件を満たす細胞をカテゴリ型でラベル付け
    adata.obs["is_clz_selective"] = (((adata.obs["cAMP_clz_selectivity"] > selectivity_threshold) & 
                                    (adata.obs["cAMP_CLOZAPINE"] > 0))
                                    ).astype("category")
    
    print("clz selective cells")
    print("# of clz selective cells:",adata.obs["is_clz_selective"].value_counts())
    num_clz_selective = adata.obs["is_clz_selective"].value_counts()[True]
    sc.pl.umap(adata, color=["is_clz_selective"],palette=["gray", "red"])
    return adata,num_clz_selective

def create_GPCR_pattern(n_pattern):
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    # 重複を避けるために使用するセット
    unique_patterns_set = set()

    # 結果を保存するための辞書
    pattern_dict = {}

    # 1万種類の独自の活性化パターンを生成
    i = 0
    while len(unique_patterns_set) < n_pattern:
        # ランダムな活性化パターンを生成（0はFalse、1はTrueとする）
        random_pattern = np.random.randint(2, size=len(GPCR_list2))
        # パターンを文字列に変換してハッシュ可能にする
        pattern_str = ''.join(map(str, random_pattern))

        # このパターンがまだ見つかっていない場合は保存
        if pattern_str not in unique_patterns_set:
            unique_patterns_set.add(pattern_str)
            pattern_dict[f"Pattern_{i+1}"] = {gpcr: bool(val) for gpcr, val in zip(GPCR_list2, random_pattern)}
            i += 1
            
    # pattern_dictをデータフレームに変換
    pattern_df = pd.DataFrame.from_dict(pattern_dict, orient='index').reset_index(drop=True)
    return pattern_df

def drug_titeration(adata, GPCR_df, GPCR_type_df, drug_list, D_R_mtx):
    import bisect
    import matplotlib.pyplot as plt
    # べき指数が -3 から +5 までのリスト（必要に応じて変更）
    powers = [i for i in range(-3, 6)]
    bisect.insort(powers, 0.2)
    bisect.insort(powers, 0.35)
    bisect.insort(powers, 0.5)
    bisect.insort(powers, 0.75)
    #bisect.insort(powers, 1.2)
    bisect.insort(powers, 1.5)
    bisect.insort(powers, 2.5)

    # 10のべき乗の値をリストにする（薬剤濃度リスト）
    drug_conc_list = [10**i for i in powers]

    # 各濃度におけるクロザピン選択細胞数を格納するリスト
    num_clz_list = []

    for drug_conc in drug_conc_list:
        print("Drug concentration:", drug_conc)
        # 薬剤反応の計算（関数の実装に依存）
        adata = calc_drug_response(adata, GPCR_df, GPCR_type_df, drug_list, D_R_mtx, drug_conc)
        # クロザピン選択細胞の計算（この関数は adata と num_clz_selective を返すと仮定）
        adata, num_clz_selective = calc_clz_selective_cell(adata, drug_list, selectivity_threshold=1.5)
        num_clz_list.append(num_clz_selective)

    # プロット
    plt.figure(figsize=(8, 6))
    plt.plot(drug_conc_list, num_clz_list, marker='o', linestyle='-')
    plt.xscale('log')  # x軸を対数スケールに設定
    plt.xlabel("Drug Concentration (nM)")
    plt.ylabel("Number of Clozapine Selective Cells")
    plt.title("Clozapine Selectivity vs. Drug Concentration")
    plt.ylim(bottom=0) 
    #plt.grid(True)
    plt.show()

def sim_inhibit_pattern(adata,GPCR_adata_norm_df,GPCR_type_df,drug_conc,n_pattern=10000):
    
    # 前提：以下の変数は既に定義されているものとする
    # adata: シングルセル解析の AnnData オブジェクト（obs に "is_clz_selective" などが含まれる）
    # GPCR_adata_norm_df: 正規化済み GPCR 発現データの DataFrame（行=細胞, 列=受容体名）
    # GPCR_type_df: 受容体タイプの DataFrame（列: receptor_name, type）; type は "Gs", "Gi" 等
    # drug_conc: 薬剤濃度（scalar）
    # ※ D_R_mtx は本コードでは使用せず、effective Ki 値によりシミュレーションする
    from tqdm import tqdm  # 追加：進捗バー用ライブラリ
    # 1. adata.obs の "is_clz_selective" に基づき、グループ分けするためのマスクを作成
    mask = adata.obs['is_clz_selective'].astype(bool) == True
    mask.index = pd.RangeIndex(start=0, stop=adata.obs['is_clz_selective'].shape[0], step=1)
    # 2. GPCRのリストおよび GPCR_type_df のフィルタリング
    # "Unnamed: 0" を除外したカラムリストを作成
    GPCR_list2 = [col for col in GPCR_adata_norm_df.columns if col != "Unnamed: 0"]
    # 全細胞の GPCR 発現データ（正規化済み）の DataFrame を用意
    # ※ GPCR_adata_norm_df の index と adata.obs_names が整合している前提
    all_expr = pd.DataFrame(GPCR_adata_norm_df, index=GPCR_adata_norm_df.index, columns=GPCR_list2)

    #GPCR_type_df = GPCR_type_df[GPCR_type_df.receptor_name.isin(GPCR_list2)]
    Gs = GPCR_type_df[GPCR_type_df.type == "Gs"]["receptor_name"].values
    Gi = GPCR_type_df[GPCR_type_df.type == "Gi"]["receptor_name"].values
    # expression_df に存在し、かつ effective_Ki にも存在する GPCR のみを抽出
    Gs_filtered = [gene for gene in Gs if (gene + '_raw' in all_expr.columns)]
    Gi_filtered = [gene for gene in Gi if (gene + '_raw' in all_expr.columns)]

    # フィルタ後のリストから、expression_df の列名用リストを作成
    Gs_cols = [gene + '_raw' for gene in Gs_filtered]
    Gi_cols = [gene + '_raw' for gene in Gi_filtered]

    # 4. ランダムな受容体阻害パターンを 10,000 パターン生成
    unique_patterns_set = set()
    pattern_dict = {}
    i = 0
    #n_pattern=10
    while len(unique_patterns_set) < n_pattern:
        random_pattern = np.random.randint(2, size=len(GPCR_list2))
        pattern_str = ''.join(map(str, random_pattern))
        if pattern_str not in unique_patterns_set:
            unique_patterns_set.add(pattern_str)
            # 各パターンは、受容体ごとに True (阻害する) / False (阻害しない) の辞書とする
            pattern_dict[f"Pattern_{i+1}"] = {gpcr: bool(val) for gpcr, val in zip(GPCR_list2, random_pattern)}
            i += 1

    # オプション：最初の5パターンを確認
    for key in list(pattern_dict.keys())[:5]:
        print(f"{key}: {pattern_dict[key]}")

    def simulate_response_all(expression_df, pattern, drug_conc, Gs_cols, Gi_cols):
        """
        expression_df: 各細胞の受容体発現 (DataFrame, 行=細胞, 列=受容体)
        pattern: 受容体阻害パターン（辞書, receptor -> bool, True=阻害する）
        drug_list: 薬剤名のリスト
        drug_conc: 薬剤濃度（scalar）
        Gs, Gi: Gs, Gi タイプ受容体名の配列
        """
        # 阻害パターンに応じた effective Ki の設定
        # 阻害する受容体は Ki = 0.01、阻害しない受容体は Ki = 10000
        #effective_Ki = pd.Series({receptor: (0.01 if pattern[receptor] else 10000)
        #                          for receptor in expression_df.columns})
        effective_Ki = pd.Series({
        receptor: (0.01 if pattern[receptor] else 10000)
        for receptor in expression_df.columns
        })
        #print(effective_Ki)
        responses = {}
        # Gs 効果・Gi 効果を計算
        gs_effect = (expression_df[Gs_cols].divide(1 + drug_conc / effective_Ki[Gs_cols])).sum(axis=1)
        gi_effect = (expression_df[Gi_cols].divide(1 + drug_conc / effective_Ki[Gi_cols])).sum(axis=1)
        basal_cAMP = (expression_df[Gs_cols] - expression_df[Gi_cols]).sum(axis=1)
        cAMPmod = (gs_effect - gi_effect) - basal_cAMP
        #print("cAMPmod")
        #print(cAMPmod.sum())
        
        # 細胞ごとの cAMPmod の Series とする
        responses= cAMPmod
        return responses

    # 6. 各阻害パターンについて、全細胞でシミュレーションした後、clz_selective と非選択細胞間の差分を算出
    results = []
    # tqdm を用いて進捗状況を表示
    for pattern_name, pattern in tqdm(pattern_dict.items(), total=len(pattern_dict), desc="Simulating drug responses"):
        # 全細胞でのシミュレーション結果を得る
        all_responses = simulate_response_all(all_expr, pattern, drug_conc, Gs_cols, Gi_cols)
        #print(all_responses)
        # 各薬剤について、mask_aligned を用いて clz_selective と非選択細胞群の平均値を計算
        selective_mean = all_responses[mask].mean()
        nonselective_mean = all_responses[~mask].mean()
        diff= selective_mean - nonselective_mean
        #print(diff)
        results.append({
            'pattern_name':pattern_name,
            'pattern': pattern,
            'diff': diff
        })

    # 7. 結果を DataFrame に変換し、diff の大きい順にソート
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by='diff', ascending=False)

    # 上位のパターンを確認（例：上位5件）
    print(results_df_sorted.head())

    return results_df_sorted,all_responses


def sim_inhibit_pattern_3r(adata, GPCR_adata_norm_df, GPCR_type_df, drug_conc):
    # 前提：以下の変数は既に定義されているものとする
    # adata: シングルセル解析の AnnData オブジェクト（obs に "is_clz_selective" などが含まれる）
    # GPCR_adata_norm_df: 正規化済み GPCR 発現データの DataFrame（行=細胞, 列=受容体名）
    # GPCR_type_df: 受容体タイプの DataFrame（列: receptor_name, type）; type は "Gs", "Gi" 等
    # drug_conc: 薬剤濃度（scalar）
    
    import itertools
    import numpy as np
    import pandas as pd
    from tqdm import tqdm  # 進捗バー用ライブラリ

    # 1. adata.obs の "is_clz_selective" に基づき、グループ分けするためのマスクを作成
    mask = adata.obs['is_clz_selective'].astype(bool) == True
    mask.index = pd.RangeIndex(start=0, stop=adata.obs['is_clz_selective'].shape[0], step=1)
    
    # 2. GPCR のリストおよび GPCR_type_df のフィルタリング
    # "Unnamed: 0" を除外したカラムリストを作成
    GPCR_list2 = [col for col in GPCR_adata_norm_df.columns if col != "Unnamed: 0"]
    # 全細胞の GPCR 発現データ（正規化済み）の DataFrame を用意
    # ※ GPCR_adata_norm_df の index と adata.obs_names が整合している前提
    all_expr = pd.DataFrame(GPCR_adata_norm_df, index=GPCR_adata_norm_df.index, columns=GPCR_list2)

    # GPCR_type_df から、各タイプごとに受容体名の配列を取得
    Gs = GPCR_type_df[GPCR_type_df.type == "Gs"]["receptor_name"].values
    Gi = GPCR_type_df[GPCR_type_df.type == "Gi"]["receptor_name"].values
    # expression_df に存在する受容体に限定
    Gs_filtered = [gene for gene in Gs if (gene + '_raw' in all_expr.columns)]
    Gi_filtered = [gene for gene in Gi if (gene + '_raw' in all_expr.columns)]

    # フィルタ後のリストから、expression_df の列名用リストを作成
    Gs_cols = [gene + '_raw' for gene in Gs_filtered]
    Gi_cols = [gene + '_raw' for gene in Gi_filtered]

    # 3. GPCR_list2 の中から3つの受容体だけを阻害するパターンを全パターン生成
    # itertools.combinations を用いて、全ての組み合わせを列挙する
    pattern_dict = {}
    all_combinations = list(itertools.combinations(GPCR_list2, 3))
    for i, inhibited_receptors in enumerate(tqdm(all_combinations, desc="Generating inhibition patterns")):
        # inhibited_receptors に含まれる受容体のみ True、それ以外 False とする
        pattern = {gpcr: (gpcr in inhibited_receptors) for gpcr in GPCR_list2}
        pattern_dict[f"Pattern_{i+1}"] = pattern

    # オプション：最初の5パターンを確認
    for key in list(pattern_dict.keys())[:5]:
        print(f"{key}: {pattern_dict[key]}")

    def simulate_response_all(expression_df, pattern, drug_conc, Gs_cols, Gi_cols):
        """
        各細胞の受容体発現 DataFrame に対して、各受容体の effective Ki をパターンに基づき設定し
        薬剤の濃度 drug_conc に応じた受容体応答をシミュレーションする関数
        """
        # 阻害する受容体は Ki = 0.01、阻害しない受容体は Ki = 10000
        effective_Ki = pd.Series({receptor: (0.01 if pattern[receptor] else 10000)
                                  for receptor in expression_df.columns})
        # Gs 効果・Gi 効果を計算
        gs_effect = (expression_df[Gs_cols].divide(1 + drug_conc / effective_Ki[Gs_cols])).sum(axis=1)
        gi_effect = (expression_df[Gi_cols].divide(1 + drug_conc / effective_Ki[Gi_cols])).sum(axis=1)
        basal_cAMP = (expression_df[Gs_cols] - expression_df[Gi_cols]).sum(axis=1)
        cAMPmod = (gs_effect - gi_effect) - basal_cAMP
        
        responses = cAMPmod  # 各細胞ごとの cAMPmod の Series
        return responses

    # 4. 各阻害パターンについてシミュレーションを実施し、
    # clz_selective と非選択細胞群の平均応答差分（diff）を算出
    results = []
    for pattern_name, pattern in tqdm(pattern_dict.items(), total=len(pattern_dict), desc="Simulating drug responses"):
        # 全細胞でのシミュレーション結果を取得
        all_responses = simulate_response_all(all_expr, pattern, drug_conc, Gs_cols, Gi_cols)
        selective_mean = all_responses[mask].mean()
        nonselective_mean = all_responses[~mask].mean()
        diff = selective_mean - nonselective_mean
        results.append({
            'pattern_name': pattern_name,
            'pattern': pattern,
            'diff': diff
        })

    # 5. 結果を DataFrame に変換し、diff の大きい順にソート
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by='diff', ascending=False)

    # 上位のパターンを確認（例：上位5件）
    print(results_df_sorted.head())

    return results_df_sorted, all_responses

def visualize_patterns(results_df_sorted, top_n=None, top_n_for_heatmap=None, scatter_n=None):
    """
    Parameters:
      results_df_sorted: DataFrame with columns 'pattern_name', 'pattern', 'diff'
                         'pattern' は {'HTR1A_raw': True/False, ...} の辞書形式とする
      top_n: ヒートマップ（従来版）および棒グラフで表示する上位パターン数（Noneの場合は全パターン）
      top_n_for_heatmap: 拡大版ヒートマップで表示する上位パターン数（Noneの場合は表示しない）
      scatter_n: 散布図にプロットするパターン数（Noneの場合は全パターン）
    
    Display:
      1. ヒートマップ（従来版）:
         - X軸: 受容体名（"_raw" を除去）、X軸ラベルは90°回転
         - Y軸: diff が大きい順のパターン番号（1,2,3,...）
         - 二値（True/False）の離散カラーマップを使用し、legend を右側に配置（余白を確保）
      2. ヒートマップ（拡大版）:
         - top_n_for_heatmap で指定した上位パターンを表示（従来版と同様の設定）
      3. 棒グラフ:
         - top_n パターン中の各受容体の True の割合 (%) を表示
         - ヒートマップと同じ横幅、X軸ラベルは90°回転、右側に空の legend を配置
      4. 散布図:
         - scatter_n に指定したパターン数（または全パターン）の diff 値をプロット
         - X軸はパターン番号（diff が大きい順、1～）、ラベルは90°回転
    """
    # 1,2,3 用のデータ（ヒートマップ（従来版）および棒グラフ）は top_n を使用（top_n が None の場合は全パターン）
    if top_n is not None:
        df_subset = results_df_sorted.head(top_n).reset_index(drop=True)
    else:
        df_subset = results_df_sorted.copy().reset_index(drop=True)
    n_patterns_subset = df_subset.shape[0]
    
    # ヒートマップ描画用のヘルパー関数
    def plot_heatmap(df, version_label):
        n_patterns = df.shape[0]
        # すべてのパターンで同じ受容体キーが使われていると仮定し、最初のパターンからキーを取得
        first_pattern = df.iloc[0]['pattern']
        receptor_keys = list(first_pattern.keys())
        receptors = [key.replace('_raw', '') for key in receptor_keys]
        
        # 各パターンの辞書をバイナリ行列に変換（True→1, False→0）
        pattern_matrix = np.zeros((n_patterns, len(receptors)), dtype=int)
        for i, row in df.iterrows():
            pat = row['pattern']
            for j, key in enumerate(receptor_keys):
                pattern_matrix[i, j] = 1 if pat.get(key, False) else 0
        
        # パターン番号ラベル（1～）
        pattern_labels = [str(i + 1) for i in range(n_patterns)]
        
        # 二値用離散カラーマップ（False: white, True: steelblue）
        cmap = ListedColormap(['white', 'steelblue'])
        
        # ヒートマップ描画
        fig, ax = plt.subplots(figsize=(12, max(6, n_patterns * 0.3)))
        im = ax.imshow(pattern_matrix, aspect='auto', cmap=cmap)
        
        # X軸：受容体名（90°回転）
        ax.set_xticks(np.arange(len(receptors)))
        ax.set_xticklabels(receptors, rotation=90, ha='center')
        ax.set_xlabel('Receptor Name')
        
        # Y軸：パターン番号
        ax.set_yticks(np.arange(n_patterns))
        ax.set_yticklabels(pattern_labels)
        ax.set_ylabel('Pattern (sorted by diff descending)')
        ax.set_title(f'Pattern Visualization ({version_label}) (Top {n_patterns} Patterns)')
        
        # legend を右側に配置
        false_patch = mpatches.Patch(color=cmap(0), label='False')
        true_patch = mpatches.Patch(color=cmap(1), label='True')
        ax.legend(handles=[false_patch, true_patch], title='Value',
                  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        # legend 分の余白を確保
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    
    # 1. ヒートマップ（従来版）：df_subset を使用
    plot_heatmap(df_subset, version_label="")
    
    # 2. ヒートマップ（拡大版）：top_n_for_heatmap が指定されている場合
    if top_n_for_heatmap is not None:
        df_enlarge = results_df_sorted.head(top_n_for_heatmap).reset_index(drop=True)
        plot_heatmap(df_enlarge, version_label="")
    
    # 3. 棒グラフ: df_subset をもとに各受容体の True の割合 (%) を計算
    first_pattern = df_subset.iloc[0]['pattern']
    receptor_keys = list(first_pattern.keys())
    receptors = [key.replace('_raw', '') for key in receptor_keys]
    
    pattern_matrix = np.zeros((n_patterns_subset, len(receptors)), dtype=int)
    for i, row in df_subset.iterrows():
        pat = row['pattern']
        for j, key in enumerate(receptor_keys):
            pattern_matrix[i, j] = 1 if pat.get(key, False) else 0
            
    true_counts = pattern_matrix.sum(axis=0)
    true_percentage = (true_counts / n_patterns_subset) * 100
    
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(np.arange(len(receptors)), true_percentage)
    ax2.set_xlabel('Receptor Name')
    ax2.set_ylabel('True Percentage (%)')
    ax2.set_title(f'True Percentage per Receptor (Top {n_patterns_subset} Patterns)')
    ax2.set_xticks(np.arange(len(receptors)))
    ax2.set_xticklabels(receptors, rotation=90, ha='center')
    
    # 空の legend を追加して右側の余白を確保（ダミーパッチを追加）
    dummy_patch = mpatches.Patch(color='none', label='')
    ax2.legend(handles=[dummy_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    
    # 4. 散布図: scatter_n に指定があればその上位パターン、指定がなければ全パターン
    if scatter_n is not None:
        df_scatter = results_df_sorted.head(scatter_n).reset_index(drop=True)
    else:
        df_scatter = results_df_sorted.copy().reset_index(drop=True)
    total_patterns = df_scatter.shape[0]
    scatter_labels = [str(i + 1) for i in range(total_patterns)]

    # 50刻みのtickを設定
    ticks = np.arange(0, total_patterns, 50)
    tick_labels = [str(tick+1) for tick in ticks]
    
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.scatter(np.arange(total_patterns), df_scatter['diff'], s=10)
    ax3.set_xlabel('Pattern (sorted by diff descending)')
    ax3.set_ylabel('Diff')
    ax3.set_title(f'Diff Values for Top {total_patterns} Patterns (sorted descending)')
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tick_labels, rotation=90, ha='center')
    
    plt.tight_layout()
    plt.show()
