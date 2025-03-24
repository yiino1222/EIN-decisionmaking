import numpy as np
import scanpy as sc
import anndata

import time
import os, wget

import cupy as cp
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
    params['pca_train_ratio'] = 0.1
    params['n_pca_batches'] = 40
    
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


def preprocess_adata_in_bulk(adata_path,label=None,add_markers=None):
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
                  gpu=True)
    
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

def sim_inhibit_pattern(adata,GPCR_adata_norm_df,GPCR_type_df,drug_list,drug_conc):
    # 前提：以下の変数は既に定義されているものとする
    # adata: シングルセル解析の AnnData オブジェクト（obs に "is_clz_selective" などが含まれる）
    # GPCR_adata_norm_df: 正規化済み GPCR 発現データの DataFrame（行=細胞, 列=受容体名）
    # GPCR_type_df: 受容体タイプの DataFrame（列: receptor_name, type）; type は "Gs", "Gi" 等
    # drug_list: 薬剤名のリスト（例: ["drugA", "drugB", ...]）
    # drug_conc: 薬剤濃度（scalar）
    # ※ D_R_mtx は本コードでは使用せず、effective Ki 値によりシミュレーションする

    # 1. adata.obs の "is_clz_selective" に基づき、グループ分けするためのマスクを作成
    mask = adata.obs['is_clz_selective'] == True

    # 2. GPCRのリストおよび GPCR_type_df のフィルタリング
    GPCR_list2 = GPCR_adata_norm_df.columns
    GPCR_type_df = GPCR_type_df[GPCR_type_df.receptor_name.isin(GPCR_list2)]
    Gs = GPCR_type_df[GPCR_type_df.type == "Gs"]["receptor_name"].values
    Gi = GPCR_type_df[GPCR_type_df.type == "Gi"]["receptor_name"].values

    # 3. ランダムな受容体阻害パターンを 10,000 パターン生成
    unique_patterns_set = set()
    pattern_dict = {}
    i = 0
    while len(unique_patterns_set) < 10000:
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

    # 4. 全細胞の GPCR 発現データ（正規化済み）の DataFrame を用意
    # ※ GPCR_adata_norm_df の index と adata.obs_names が整合している前提
    all_expr = pd.DataFrame(GPCR_adata_norm_df, index=GPCR_adata_norm_df.index, columns=GPCR_list2)

    # 5. 全細胞に対する cAMP 変化（cAMP modulation）をシミュレーションする関数の定義
    def simulate_drug_response_all(expression_df, pattern, drug_list, drug_conc, Gs, Gi):
        """
        expression_df: 各細胞の受容体発現 (DataFrame, 行=細胞, 列=受容体)
        pattern: 受容体阻害パターン（辞書, receptor -> bool, True=阻害する）
        drug_list: 薬剤名のリスト
        drug_conc: 薬剤濃度（scalar）
        Gs, Gi: Gs, Gi タイプ受容体名の配列
        """
        # 阻害パターンに応じた effective Ki の設定
        # 阻害する受容体は Ki = 0.01、阻害しない受容体は Ki = 10000
        effective_Ki = pd.Series({receptor: (0.01 if pattern[receptor] else 10000)
                                for receptor in expression_df.columns})
        
        responses = {}
        for drug in drug_list:
            # 各薬剤について、Gs 効果・Gi 効果を計算
            gs_effect = (expression_df[Gs].divide(1 + drug_conc / effective_Ki[Gs])).sum(axis=1)
            gi_effect = (expression_df[Gi].divide(1 + drug_conc / effective_Ki[Gi])).sum(axis=1)
            basal_cAMP = (expression_df[Gs] - expression_df[Gi]).sum(axis=1)
            cAMPmod = (gs_effect - gi_effect) - basal_cAMP
            # 各薬剤の結果は、細胞ごとの cAMPmod の Series とする
            responses[drug] = cAMPmod
        return responses

    # 6. 各阻害パターンについて、全細胞でシミュレーションした後、clz_selective と非選択細胞間の差分を算出
    results = []
    for pattern_name, pattern in pattern_dict.items():
        # 全細胞でのシミュレーション結果を得る
        all_responses = simulate_drug_response_all(all_expr, pattern, drug_list, drug_conc, Gs, Gi)
        
        diff = {}
        for drug in drug_list:
            # clz_selective 細胞群の平均 cAMPmod
            selective_mean = all_responses[drug][mask].mean()
            # 非選択細胞群の平均 cAMPmod
            nonselective_mean = all_responses[drug][~mask].mean()
            diff[drug] = selective_mean - nonselective_mean
        # 複数薬剤の場合、ここでは平均差を評価指標とする
        mean_diff = np.mean(list(diff.values()))
        results.append({
            'pattern': pattern_name,
            'diff_per_drug': diff,
            'mean_diff': mean_diff
        })

    # 7. 結果を DataFrame に変換し、mean_diff の大きい順にソート
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by='mean_diff', ascending=False)

    # 上位のパターンを確認（例：上位5件）
    print(results_df_sorted.head())

"""
def preprocess_adata_in_batch(adata_path,max_cells):
    preprocess_start = time.time()
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    #set parameters
    params=set_parameters_for_preprocess(GPCR_list)
    
    #preprocess in batch
    print("preprocess_in_batches")
    sparse_gpu_array, genes, marker_genes_raw = \
    rapids_scanpy_funcs.preprocess_in_batches(adata_path, 
                                              params['markers'], 
                                              min_genes_per_cell=params['min_genes_per_cell'], 
                                              max_genes_per_cell=params['max_genes_per_cell'], 
                                              min_cells_per_gene=params['min_cells_per_gene'], 
                                              target_sum=1e4, 
                                              n_top_genes=params['n_top_genes'],
                                              max_cells=max_cells)#params["USE_FIRST_N_CELLS"]
    
    print("marker_genes_raw")
    print(marker_genes_raw)
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
    
    # scale
    print("perform scale")
    mean = sparse_gpu_array.mean(axis=0)
    sparse_gpu_array -= mean
    stddev = cp.sqrt(sparse_gpu_array.var(axis=0))
    sparse_gpu_array /= stddev
    sparse_gpu_array = sparse_gpu_array.clip(None,10)
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
    #If the number of cells was smaller, we would use the command 
    # `adata.obsm["X_pca"] = cuml.decomposition.PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)` 
    # to perform PCA on all the cells.
    #However, we cannot perform PCA on the complete dataset using a single GPU. 
    # Therefore, we use the batched PCA function in `utils.py`, which uses only a fraction 
    # of the total cells to train PCA.
    adata = utils.pca(adata, n_components=params["n_components"], 
                  train_ratio=params["pca_train_ratio"], 
                  n_batches=params["n_pca_batches"],
                  gpu=True)
    
    #t-sne + k-means
    adata.obsm['X_tsne'] = TSNE().fit_transform(adata.obsm["X_pca"][:,:params['tsne_n_pcs']])
    kmeans = KMeans(n_clusters=params['k'], init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str)        
    sc.pl.tsne(adata, color=["kmeans"])
    #sc.pl.tsne(adata, color=["SNAP25_raw"], color_map="Blues", vmax=1, vmin=-0.05)
    
    #UMAP + Graph clustering
    sc.pp.neighbors(adata, n_neighbors=params["n_neighbors"], n_pcs=params["knn_n_pcs"],
                    method='rapids')
    sc.tl.umap(adata, min_dist=params["umap_min_dist"], spread=params["umap_spread"],
               method='rapids')
    sc.tl.louvain(adata, flavor='rapids')
    sc.pl.umap(adata, color=["louvain"])
    adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata)
    sc.pl.umap(adata, color=["leiden"])
    #sc.pl.umap(adata, color=["SNAP25_raw"], color_map="Blues", vmax=1, vmin=-0.05)
    
    #calculate response to antipsychotics
    #noramlize GPCR expression levels
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
        Gs_effect=(norm_df.loc[:,Gs]/D_R_mtx.loc[drug,Gs]).sum(axis=1) #TODO ki値で割り算するときにlog換算すべきか
        Gi_effect=(norm_df.loc[:,Gi]/D_R_mtx.loc[drug,Gi]).sum(axis=1)
        #Gq_effect=(norm_df.loc[:,Gq]/D_R_mtx.loc[drug,Gq]).sum(axis=1)
        cAMPmod=Gi_effect-Gs_effect #Giの阻害→cAMP上昇、Gsの阻害→cAMP低下
        Camod=-Gq_effect #Gq阻害→Ca低下
        cAMP_df[drug]=cAMPmod
        Ca_df[drug]=Camod
        
    cAMP_df.index=adata.obs_names
    #Ca_df.index=adata.obs_names
    #Ca_df=Ca_df+10**(-4)
    for drug in drug_list:
        adata.obs['cAMP_%s'%drug]=cAMP_df[drug]
        #adata.obs['Ca_%s'%drug]=Ca_df[drug]
    
    #save preprocessed adata 
    file_root, file_extension = os.path.splitext(adata_path)
    # Append '_processed' to the root and add the extension back
    processed_file_path = f"{file_root}_processed{file_extension}"
    adata.write(processed_file_path)
    
    return adata
"""