"""Rebuild the small synthetic example dataset. No real orchard data is used."""
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Import engine first to configure the GDAL DLL path in Conda on Windows.
import engine
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin


def generate():
    folder=ROOT/'examples'
    imagery=folder/'imagery'
    imagery.mkdir(parents=True,exist_ok=True)
    width,height,resolution=128,96,.15
    origin_x,origin_y=400000.0,3800000.0
    transform=from_origin(origin_x,origin_y+height*resolution,resolution,resolution)
    cols,rows=np.meshgrid(np.arange(width),np.arange(height))
    x=(cols+.5)*resolution
    y=(height-rows-.5)*resolution
    positions=[(3+4*col,3+4*row) for row in range(3) for col in range(4)]
    dates=['2024-08-01','2024-09-01']
    samples,config=[],[]

    def write_raster(path,array):
        with rasterio.open(path,'w',driver='GTiff',height=height,width=width,
                           count=1,dtype='float32',crs='EPSG:32649',transform=transform,
                           compress='DEFLATE',predictor=3,nodata=-9999.0) as dst:
            dst.write(np.round(array,4).astype('float32'),1)

    for season,date in enumerate(dates):
        canopy=np.zeros((height,width))
        density=np.zeros_like(canopy)
        for i,(cx,cy) in enumerate(positions):
            radius=1.05+.055*i+.06*season
            local=np.exp(-((x-cx)**2+(y-cy)**2)/(radius**2*.68))
            canopy=np.maximum(canopy,local)
            density=np.maximum(density,local*(.74+.018*i+.045*season))
            samples.append(dict(Tree_ID=f'DEMO_{i+1:02d}',Date=date,
                                X=origin_x+cx,Y=origin_y+cy,
                                LAI_Observed=round(1.05+.075*i+.12*season+.015*np.sin(i),4)))
        texture=.008*np.sin(x*9.1)*np.cos(y*6.7)*canopy
        bands={
            'Green':.085+.105*density+texture*.6,
            'Red':.115-.068*density+texture*.2,
            'RedEdge':.125+.245*density+texture*.8,
            'NIR':.165+.49*density+texture,
        }
        cfg={'Date':date}
        for band,array in bands.items():
            filename=f'{date}_{band.lower()}.tif'
            write_raster(imagery/filename,array)
            cfg[band]='imagery/'+filename
        config.append(cfg)
        if season==1:
            lai=1.0+1.0*density+.04*np.sin(x*.8)*canopy
            write_raster(folder/'lai_demo.tif',lai)

    pd.DataFrame(samples).to_csv(folder/'sample_locations.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(config).to_csv(folder/'imagery_config.csv',index=False,encoding='utf-8-sig')
    params=dict(samples='examples/sample_locations.csv',imagery_config='examples/imagery_config.csv',
                sample_crs='EPSG:32649',q=70,p=90)
    with tempfile.TemporaryDirectory(prefix='lai-example-') as temp:
        dest=Path(temp)
        result=engine.extract(params,dest,print)
        if result['valid']!=24:raise RuntimeError(f'Expected 24 complete rows: {result}')
        table=pd.read_csv(dest/'extracted_features.csv')
        table.to_csv(folder/'example_features.csv',index=False,encoding='utf-8-sig')
        table[table.Date==dates[1]].drop(columns=['LAI_Observed']).to_csv(
            folder/'example_prediction.csv',index=False,encoding='utf-8-sig')
        engine.preview_map(dict(raster='examples/lai_demo.tif'),dest,print)
        (folder/'lai_demo.png').write_bytes((dest/'map_preview.png').read_bytes())
    print('Generated: 12 synthetic trees, 2 dates, 8 small band GeoTIFFs, 24 feature rows.')


if __name__=='__main__':
    generate()
