"""Portable local LAI workbench; reference algorithms are retained unchanged."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time, os
# 小样本模型单线程即可；避免本机遥感与数值库线程运行时冲突。
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
conda_bin=Path(sys.prefix)/'Library'/'bin'
if conda_bin.exists():
    os.environ['PATH']=str(conda_bin)+os.pathsep+os.environ.get('PATH','')
    if hasattr(os,'add_dll_directory'): _dll_handle=os.add_dll_directory(str(conda_bin))
gdal_data=Path(sys.prefix)/'Library'/'share'/'gdal'
if gdal_data.exists():os.environ.setdefault('GDAL_DATA',str(gdal_data))
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SETTINGS=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
FEATURES=['NDVI','RVI','DVI','TNDVI','RDVI','NGRDI','RI','MSR','MSAVI','TVI','WDRVI','GRVI','NLI','MTVI2','CRI','GNDVI','OSAVI','SAVI','PVI','MCARI','RGRI','NDRE','TCARI_OSAVI','RECI']
FEATURES += ['TexNIR_'+p+'_Avg' for p in ['Contrast','Dissimilarity','Homogeneity','Energy','Correlation','ASM','Entropy','Mean']]
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def resolve_input(path, base=ROOT):
    if not str(path).strip(): raise ValueError('请输入文件路径。')
    p=Path(str(path).strip()).expanduser()
    return p.resolve() if p.is_absolute() else (base/p).resolve()

def number(params, name, default, minimum, maximum, integer=False):
    try: value=float(params.get(name, default))
    except (TypeError, ValueError): raise ValueError(f'{name} 必须是数值。')
    if not np.isfinite(value) or not minimum<=value<=maximum or (integer and not value.is_integer()):
        raise ValueError(f'{name} 须在 {minimum}～{maximum} 之间'+('，且为整数。' if integer else '。'))
    return int(value) if integer else value

def read_table(path):
    p=resolve_input(path)
    if not p.is_file() or p.suffix.lower() not in ['.csv','.xlsx']:
        raise ValueError('请选择存在的 CSV 或 XLSX 数据表。')
    df=pd.read_excel(p) if p.suffix.lower()=='.xlsx' else pd.read_csv(p)
    return df.rename(columns={'ID':'Tree_ID','E':'X','N':'Y'})

def fingerprint(path):
    h=hashlib.sha256()
    with open(resolve_input(path),'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def valid_features(df,features,label=False):
    required=features+(['Tree_ID','LAI_Observed'] if label else [])
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError('缺少字段：'+', '.join(missing))
    vals=df[features+(['LAI_Observed'] if label else [])].apply(pd.to_numeric,errors='coerce')
    ok=np.isfinite(vals).all(axis=1)
    if label: ok &= df.Tree_ID.notna() & (vals.LAI_Observed>=0)
    if not ok.any(): raise ValueError('没有包含完整有效特征的记录，请检查取样状态和缺失值。')
    return vals,ok

def train(params,dest,progress):
    path=resolve_input(params['table']); df=read_table(path)
    features=FEATURES
    vals,ok=valid_features(df,features,label=True)
    clean=df.loc[ok].copy(); X=vals.loc[ok,features]; y=vals.loc[ok,'LAI_Observed']
    if clean.Tree_ID.nunique()<10: raise ValueError('按树编号十折验证至少需要 10 个不同编号；不会自动改成随机拆分。')
    pipeline=params.get('pipeline','public_reference_v1')
    if pipeline not in ['archived_v1','public_reference_v1']:raise ValueError('未知特征流程版本。')
    defaults=SETTINGS.get('model_defaults',{})
    model=GradientBoostingRegressor(
        n_estimators=number(params,'n_estimators',defaults.get('n_estimators',100),10,1000,True),
        learning_rate=number(params,'learning_rate',defaults.get('learning_rate',.05),.0001,1),
        max_depth=number(params,'max_depth',defaults.get('max_depth',3),1,20,True),random_state=42)
    progress('正在按树编号进行十折验证；同一树的多期记录保持在同一折。')
    oof=cross_val_predict(model,X,y,cv=GroupKFold(10),groups=clean.Tree_ID,n_jobs=1)
    rmse=float(np.sqrt(mean_squared_error(y,oof)))
    metrics={'R2':float(r2_score(y,oof)),'RMSE':rmse,'MAE':float(mean_absolute_error(y,oof)),
             'RPD':float(np.std(y,ddof=1)/rmse) if rmse>0 else None,'observations':len(clean),'tree_ids':int(clean.Tree_ID.nunique()),'excluded_rows':int((~ok).sum()),
             'validation':'GroupKFold(10), grouped by Tree_ID; fixed 32 features',
             'parameters':model.get_params()}
    model.fit(X,y)
    joblib.dump(model,dest/'model.joblib')
    meta={'features':features,'pipeline':pipeline,'data_sha256':fingerprint(path),'source':str(Path(path).resolve()),
          'training_lai_range':[float(y.min()),float(y.max())],'metrics':metrics,'parameters':model.get_params()}
    write_json(dest/'model_meta.json',meta)
    clean['LAI_CV_Predicted']=oof; clean['LAI_CV_Residual']=oof-y
    clean.to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    write_json(dest/'metrics.json',metrics)
    fig,ax=plt.subplots(figsize=(6,4.6));ax.scatter(y,oof,s=20,color='#13795b',alpha=.75)
    lo=min(float(y.min()),float(oof.min()));hi=max(float(y.max()),float(oof.max()))
    ax.plot([lo,hi],[lo,hi],color='#334155',linestyle='--');ax.set(xlabel='参考 LAI',ylabel='十折验证预测 LAI',title=f'分组十折验证  R²={metrics["R2"]:.3f}  RMSE={rmse:.3f}')
    fig.tight_layout();fig.savefig(dest/'validation.png',dpi=160);plt.close(fig)
    return {'message':f'已验证 {len(clean)} 条记录，模型及结果已存档。','metrics':metrics,'preview':'validation.png'}

def predict(params,dest,progress):
    model_id=params.get('model_id','')
    if not model_id or Path(model_id).name!=model_id:raise ValueError('请先选择由本应用训练的模型。')
    folder=OUT/model_id
    if not (folder/'model_meta.json').is_file():raise ValueError('模型不存在。')
    meta=json.loads((folder/'model_meta.json').read_text(encoding='utf-8'))
    pipeline=params.get('pipeline','public_reference_v1')
    if pipeline!=meta['pipeline']:raise ValueError('模型和输入的特征流程版本不一致，请用同一流程重新提取并标定。')
    df=read_table(params['table']); vals,ok=valid_features(df,meta['features'])
    model=joblib.load(folder/'model.joblib') # 仅加载本应用输出目录内的自建模型
    pred=np.full(len(df),np.nan);pred[ok]=model.predict(vals.loc[ok,meta['features']])
    df['LAI_Predicted']=pred;df['Prediction_Status']=np.where(ok,'有效','特征缺失或非有限值')
    lo,hi=meta['training_lai_range'];df['Outside_Training_LAI_Range']=np.where(ok,(pred<lo)|(pred>hi),False)
    df.to_csv(dest/'lai_predictions.csv',index=False,encoding='utf-8-sig')
    result={'message':f'已预测 {int(ok.sum())} 条记录；{int((~ok).sum())} 条记录保留为空并注明原因。',
            'valid':int(ok.sum()),'invalid':int((~ok).sum()),'mean_lai':float(np.nanmean(pred)),'model_id':model_id,
            'table_preview':json.loads(df.head(20).to_json(orient='records',force_ascii=False))}
    if {'X','Y'}<=set(df.columns):
        fig,ax=plt.subplots(figsize=(7,4.5));m=ax.scatter(df.X,df.Y,c=pred,cmap='RdYlGn',s=42)
        fig.colorbar(m,ax=ax,label='预测 LAI');ax.set(xlabel='X（输入坐标）',ylabel='Y（输入坐标）',title='样点 LAI 分布（非连续栅格）');ax.set_aspect('equal',adjustable='datalim')
        fig.tight_layout();fig.savefig(dest/'sample_lai.png',dpi=160);plt.close(fig);result['preview']='sample_lai.png'
    return result

def extract(params,dest,progress):
    sample_path=resolve_input(params['samples']); config_path=resolve_input(params['imagery_config']);sample_crs=params.get('sample_crs',SETTINGS['sample_crs'])
    # 动态半径必须在米制投影中计算。原始影像为 EPSG:4326 时使用虚拟重投影，避免整幅复制。
    path=ROOT/'reference'/'code'/'02_extract_vi_texture_features.py'
    spec=importlib.util.spec_from_file_location('lai_ref_features',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    from pyproj import CRS
    src_sample=CRS.from_user_input(sample_crs)
    if not src_sample.is_projected or src_sample.axis_info[0].unit_name!='metre':raise ValueError('样点坐标系须为米制投影，请填写坐标表实际使用的 CRS。')
    cfg=mod.dyn.load_imagery_config(config_path)
    if len(cfg)>12:raise ValueError('一次最多处理 12 个日期，请分批运行。')
    # 保留日期原文，减少 2024-8-7 和 2024-08-07 不一致导致的静默漏样。
    sdf=read_table(sample_path)
    if 'Date' in sdf:
        a=pd.to_datetime(sdf.Date,errors='raise').dt.strftime('%Y-%m-%d');b=pd.to_datetime(cfg.Date,errors='raise').dt.strftime('%Y-%m-%d')
        if not set(a)<=set(b):raise ValueError('样点日期与影像配置不匹配。')
        date_map=dict(zip(b,cfg.Date.astype(str)));sdf['Date']=a.map(date_map)
    sdf.to_csv(dest/'normalized_samples.csv',index=False,encoding='utf-8-sig')
    original_open=mod.dyn.open_band_sources
    opened=[]
    def metric_sources(row):
        raws=[]; result={}
        try:
            for band in mod.dyn.BAND_NAMES:
                raw=rasterio.open(resolve_input(row[band],config_path.parent));raws.append(raw)
            ref=raws[0]
            if ref.crs is None:raise ValueError('影像缺少坐标系。')
            if any(s.crs!=ref.crs or s.transform!=ref.transform or s.width!=ref.width or s.height!=ref.height for s in raws):raise ValueError('四个波段的坐标系、网格或范围不一致，请先配准。')
            for band,raw in zip(mod.dyn.BAND_NAMES,raws):
                # 未指定标定系数时仅允许反射率范围；DN 影像不能默默当反射率。
                coeff=mod.dyn.get_calibration_coefficients(row)
                if band not in coeff:
                    v=raw.read(1,out_shape=(100,100),masked=True).compressed();v=v[np.isfinite(v)]
                    if len(v) and np.quantile(v,.95)>1.5:raise ValueError(f'{band} 看似 DN 值，请填写该期的 a、b 辐射系数。')
                vrt=WarpedVRT(raw,crs=src_sample.to_string(),resampling=Resampling.bilinear,dtype='float32',nodata=np.nan)
                result[band]=vrt;opened.extend([vrt,raw])
            return result
        except Exception:
            for s in result.values():s.close()
            for s in raws:s.close()
            raise
    mod.dyn.open_band_sources=metric_sources
    defaults=SETTINGS.get('roi_defaults',{})
    q=number(params,'q',defaults.get('q',70),.01,99.99)
    p=number(params,'p',defaults.get('p',90),.01,100)
    roi=mod.dyn.ROIParameters(
        q_percentile=q,p_percentile=p,
        exploration_radius_m=number(params,'exploration_radius_m',defaults.get('exploration_radius_m',2),.01,30),
        r_min_m=number(params,'r_min_m',defaults.get('r_min_m',.2),.01,30),
        r_max_m=number(params,'r_max_m',defaults.get('r_max_m',2.6),.01,30),
        fallback_radius_m=number(params,'fallback_radius_m',defaults.get('fallback_radius_m',1.5),.01,30),
        alpha=number(params,'alpha',defaults.get('alpha',1),.01,10),
        min_valid_pixels=number(params,'min_valid_pixels',defaults.get('min_valid_pixels',30),1,100000,True),
        morphology_close_radius=number(params,'morphology_close_radius',defaults.get('morphology_close_radius',2),0,20,True),
        morphology_open_radius=number(params,'morphology_open_radius',defaults.get('morphology_open_radius',1),0,20,True))
    if roi.r_min_m>roi.r_max_m:raise ValueError('最小半径不能大于最大半径。')
    if not roi.r_min_m<=roi.fallback_radius_m<=roi.r_max_m:raise ValueError('回退半径须在最小半径和最大半径之间。')
    progress('正在读取影像并转换到米制投影，逐样点提取动态 ROI 和特征。')
    try:
        mod.run_feature_extraction(dest/'normalized_samples.csv',config_path,dest/'extracted_features.csv',sample_crs,roi)
    finally:
        mod.dyn.open_band_sources=original_open
        for s in opened:s.close()
    df=pd.read_csv(dest/'extracted_features.csv')
    if df.empty:raise ValueError('没有匹配到任何记录，请检查日期和样点。')
    valid=np.isfinite(df[FEATURES].apply(pd.to_numeric,errors='coerce')).all(axis=1)
    write_json(dest/'feature_meta.json',{'pipeline':'public_reference_v1','q':q,'p':p,'sample_crs':sample_crs,
               'roi_parameters':vars(roi),
               'samples_sha256':fingerprint(sample_path),'config_sha256':fingerprint(config_path)})
    summary=df.reindex(columns=['Tree_ID','Date','ROI_Radius_m','Pixel_Count','Status'])
    summary.to_csv(dest/'roi_summary.csv',index=False,encoding='utf-8-sig')
    if not valid.any():raise ValueError('全部样点提取失败，请检查坐标系、影像覆盖与取样状态。详细记录已保存在特征表中。')
    radii=pd.to_numeric(df.loc[valid,'ROI_Radius_m'],errors='coerce')
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.plot(np.arange(1,len(radii)+1),radii,color='#13795b',marker='o',markersize=3,linewidth=1)
    ax.set(xlabel='完整特征样点序号',ylabel='观测半径（米）',title='动态 ROI 观测半径')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(dest/'roi_radius.png',dpi=160);plt.close(fig)
    return {'message':f'已提取 {len(df)} 条记录，其中 {int(valid.sum())} 条特征完整。','observations':len(df),'valid':int(valid.sum()),'pipeline':'public_reference_v1',
            'preview':'roi_radius.png','table_preview':json.loads(summary.head(20).to_json(orient='records',force_ascii=False)),
            'roi_parameters':vars(roi)}

def preview_map(params,dest,progress):
    if params.get('map_mode')=='figure':
        import shutil
        source=resolve_input(params['figure'])
        if not source.is_file() or source.suffix.lower() not in ['.png','.jpg','.jpeg','.webp']:raise ValueError('请选择已有的 PNG、JPG 或 WebP 成果图片。')
        filename='spatial_figure'+source.suffix.lower()
        shutil.copyfile(source,dest/filename)
        meta={'kind':'figure','source':str(source),'source_sha256':fingerprint(source),'display_title':'空间制图成果'}
        write_json(dest/'figure_source.json',meta)
        return {'message':'已打开空间成果图。','preview':filename,'metadata':meta}
    p=resolve_input(params['raster'])
    if not p.is_file() or p.suffix.lower() not in ['.tif','.tiff']:raise ValueError('请选择 LAI GeoTIFF。')
    with rasterio.open(p) as src:
        scale=min(1,1000/max(src.height,src.width));h=max(1,int(src.height*scale));w=max(1,int(src.width*scale))
        arr=src.read(1,out_shape=(h,w),masked=True,resampling=Resampling.nearest)
        vals=arr.compressed();vals=vals[np.isfinite(vals)]
        if not len(vals):raise ValueError('栅格没有有效值。')
        meta={'source':str(p.resolve()),'width':src.width,'height':src.height,'crs':str(src.crs),
              'preview_mean':float(np.mean(vals)),'preview_min':float(np.min(vals)),'preview_max':float(np.max(vals)),
              'preview_width':w,'preview_height':h,'statistics_scope':'preview pixels'}
        fig,ax=plt.subplots(figsize=(8,5.2));low,high=np.quantile(vals,[.02,.98]);im=ax.imshow(arr,cmap='RdYlGn',vmin=low,vmax=max(high,low+1e-4))
        ax.set_title('LAI 空间分布');ax.axis('off');fig.colorbar(im,ax=ax,label='LAI',shrink=.75)
        fig.tight_layout();fig.savefig(dest/'map_preview.png',dpi=160);plt.close(fig)
    write_json(dest/'raster_metadata.json',meta)
    return {'message':'已生成 LAI 栅格预览。','preview':'map_preview.png','metadata':meta}

ACTIONS={'train':train,'predict':predict,'extract':extract,'map':preview_map}
